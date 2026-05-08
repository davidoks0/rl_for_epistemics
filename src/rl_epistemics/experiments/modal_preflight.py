from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS_URL = "https://status.modal.com/"


def modal_status_is_healthy(status_html: str) -> bool:
    down_markers = [
        "Some services are down",
        "Downtime",
        "Internal data stores are failing",
        "functions and sandboxes are currently down",
    ]
    return not any(marker in status_html for marker in down_markers)


def fetch_modal_status(timeout: int = 20) -> dict[str, Any]:
    with urllib.request.urlopen(STATUS_URL, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="replace")
    return {
        "url": STATUS_URL,
        "healthy": modal_status_is_healthy(html),
        "contains_down_banner": "Some services are down" in html,
        "contains_internal_datastore_incident": "Internal data stores are failing" in html,
        "contains_functions_down_note": "functions and sandboxes are currently down" in html,
    }


def _run_modal_json(args: list[str], *, timeout: int = 30) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    command = [sys.executable, "-m", "modal", *args, "--json"]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    meta = {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode != 0:
        return [], meta
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        meta["json_error"] = str(exc)
        meta["stdout_prefix"] = completed.stdout[:500]
        return [], meta
    return data if isinstance(data, list) else [], meta


def _run_text(args: list[str], *, timeout: int = 30) -> tuple[str, dict[str, Any]]:
    command = [sys.executable, "-m", "modal", *args]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    return completed.stdout.strip(), {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def summarize_modal_state(
    *,
    profile: str,
    secrets: list[dict[str, Any]],
    containers: list[dict[str, Any]],
    apps: list[dict[str, Any]],
    provider_status: dict[str, Any] | None = None,
    required_secret: str = "tinker-api-key",
) -> dict[str, Any]:
    provider_status = provider_status or {}
    active_apps = [app for app in apps if str(app.get("State", "")).lower() != "stopped"]
    active_task_apps = [app for app in active_apps if _int_value(app.get("Tasks")) > 0]
    detached_task_apps = [
        app for app in active_task_apps if "detached" in str(app.get("State", "")).lower()
    ]
    zero_task_active_apps = [app for app in active_apps if _int_value(app.get("Tasks")) == 0]
    rl_for_epistemics_apps = [
        app
        for app in apps
        if str(app.get("Description", "")).startswith(("rl-for-epistemics", "rl-epistemics"))
    ]
    active_app_names = sorted({str(container.get("App Name", "")) for container in containers})
    secret_names = sorted(str(secret.get("Name", "")) for secret in secrets)
    return {
        "profile": profile,
        "provider_status": provider_status,
        "provider_status_healthy": provider_status.get("healthy"),
        "provider_outage_detected": provider_status.get("healthy") is False,
        "required_secret": required_secret,
        "required_secret_present": required_secret in secret_names,
        "active_container_count": len(containers),
        "active_container_app_names": active_app_names,
        "active_task_app_count": len(active_task_apps),
        "active_detached_task_app_count": len(detached_task_apps),
        "zero_task_active_app_count": len(zero_task_active_apps),
        "active_task_apps": active_task_apps,
        "detached_task_apps": detached_task_apps,
        "zero_task_active_apps": zero_task_active_apps,
        "recent_rl_for_epistemics_apps": rl_for_epistemics_apps[:12],
        "capacity_risk": bool(detached_task_apps or containers),
        "capacity_risk_note": (
            "New Modal calls may queue before container start if the workspace is saturated by "
            "active detached tasks. This is a scheduling preflight, not proof of a Modal outage."
        ),
        "scheduling_blocker": (
            "provider_status_outage"
            if provider_status.get("healthy") is False
            else "active_detached_tasks"
            if detached_task_apps or containers
            else None
        ),
    }


def collect_modal_preflight(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    provider_status: dict[str, Any]
    if config.get("skip_status_check"):
        provider_status = {"skipped": True}
    else:
        try:
            provider_status = fetch_modal_status(timeout=int(config.get("status_timeout", 20)))
        except Exception as exc:
            provider_status = {
                "url": STATUS_URL,
                "healthy": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    profile, profile_meta = _run_text(["profile", "current"], timeout=int(config.get("timeout", 30)))
    secrets, secrets_meta = _run_modal_json(["secret", "list"], timeout=int(config.get("timeout", 30)))
    containers, containers_meta = _run_modal_json(
        ["container", "list"], timeout=int(config.get("timeout", 30))
    )
    apps, apps_meta = _run_modal_json(["app", "list"], timeout=int(config.get("timeout", 30)))
    summary = summarize_modal_state(
        profile=profile,
        secrets=secrets,
        containers=containers,
        apps=apps,
        provider_status=provider_status,
        required_secret=str(config.get("required_secret", "tinker-api-key")),
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "raw": {
            "secrets": secrets,
            "containers": containers,
            "apps": apps,
        },
        "commands": {
            "profile": profile_meta,
            "secrets": secrets_meta,
            "containers": containers_meta,
            "apps": apps_meta,
        },
    }


def write_modal_preflight(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    output_path = Path(config.get("output_path", "outputs/report/modal_preflight.json"))
    result = collect_modal_preflight(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["output_path"] = str(output_path)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture Modal credential and scheduling preflight.")
    parser.add_argument("--output-path", default="outputs/report/modal_preflight.json")
    parser.add_argument("--required-secret", default="tinker-api-key")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--status-timeout", type=int, default=20)
    parser.add_argument("--skip-status-check", action="store_true")
    args = parser.parse_args(argv)
    result = write_modal_preflight(vars(args))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(result["output_path"])


if __name__ == "__main__":
    main()
