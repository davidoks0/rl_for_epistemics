from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_epistemics.experiments.modal_preflight import fetch_modal_status, modal_status_is_healthy

DEFAULT_VOLUME = "rl-epistemics-artifacts"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: list[str]
    timeout: int | None = None


def _modal_cmd(args: list[str], modal_env: str | None = None) -> list[str]:
    if not modal_env:
        return [sys.executable, "-m", "modal", *args]
    if args and args[0] == "run":
        args = ["run", "--env", modal_env, *args[1:]]
    elif len(args) >= 2 and args[0] == "volume" and args[1] in {"get", "put"}:
        args = ["volume", args[1], "--env", modal_env, *args[2:]]
    return [sys.executable, "-m", "modal", *args]


def _volume_put(
    volume: str,
    local_path: str,
    remote_path: str,
    *,
    modal_env: str | None = None,
) -> list[str]:
    return _modal_cmd(["volume", "put", "--force", volume, local_path, remote_path], modal_env)


def _volume_get(
    volume: str,
    remote_path: str,
    local_destination: str,
    *,
    modal_env: str | None = None,
) -> list[str]:
    return _modal_cmd(["volume", "get", "--force", volume, remote_path, local_destination], modal_env)


def _candidate_local_pythons(explicit: str | None = None) -> list[str]:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("RL_EPISTEMICS_PYTHON"):
        candidates.append(str(os.environ["RL_EPISTEMICS_PYTHON"]))
    candidates.extend(
        [
            sys.executable,
            "python3.11",
            "python3.12",
            "python3.10",
            "/opt/homebrew/bin/python3.11",
            "/opt/homebrew/bin/python3.12",
            "/opt/homebrew/bin/python3.10",
        ]
    )
    seen: set[str] = set()
    resolved: list[str] = []
    for candidate in candidates:
        path = (
            str(Path(candidate).expanduser())
            if "/" in candidate
            else shutil.which(candidate) or candidate
        )
        if path not in seen:
            seen.add(path)
            resolved.append(path)
    return resolved


def _python_supports_local_tinker(executable: str) -> tuple[bool, str]:
    probe = (
        "import sys; "
        "assert sys.version_info >= (3, 10), sys.version; "
        "import tinker"
    )
    try:
        completed = subprocess.run(
            [executable, "-c", probe],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if completed.returncode == 0:
        return True, "ok"
    detail = (completed.stderr or completed.stdout).strip().splitlines()
    return False, detail[-1] if detail else f"exit {completed.returncode}"


def resolve_local_tinker_python(explicit: str | None = None) -> str:
    failures: list[str] = []
    for executable in _candidate_local_pythons(explicit):
        ok, detail = _python_supports_local_tinker(executable)
        if ok:
            return executable
        failures.append(f"{executable}: {detail}")
        if explicit:
            break
    raise RuntimeError(
        "Local Tinker mode needs Python >=3.10 with the `tinker` package installed. "
        "Set --python-executable or RL_EPISTEMICS_PYTHON. Tried: "
        + "; ".join(failures)
    )


def build_post_gate_pipeline_steps(
    *,
    volume: str = DEFAULT_VOLUME,
    modal_env: str | None = None,
    include_preflight: bool = True,
    include_input_sync: bool = True,
    include_train: bool = True,
    include_train_sync: bool = True,
    include_eval: bool = True,
    include_eval_sync: bool = True,
    include_local_refresh: bool = True,
) -> list[PipelineStep]:
    steps: list[PipelineStep] = []
    if include_preflight:
        steps.append(
            PipelineStep(
                "modal-preflight",
                [sys.executable, "scripts/modal_preflight.py"],
                timeout=60,
            )
        )
    if include_input_sync:
        steps.extend(
            [
                PipelineStep(
                    "sync-verified-train-data",
                    _volume_put(volume, "data/verified_train", "data/", modal_env=modal_env),
                ),
                PipelineStep(
                    "sync-verified-eval-data",
                    _volume_put(volume, "data/verified_eval", "data/", modal_env=modal_env),
                ),
                PipelineStep(
                    "sync-rm-design-ablations",
                    _volume_put(
                        volume,
                        "outputs/rm_design_ablations",
                        "outputs/",
                        modal_env=modal_env,
                    ),
                ),
                PipelineStep(
                    "sync-gpt55-adjudication-csv",
                    _volume_put(
                        volume,
                        "outputs/blog/manual_review_queue_adjudicated_gpt55.csv",
                        "outputs/blog/",
                        modal_env=modal_env,
                    ),
                ),
                PipelineStep(
                    "sync-gpt55-adjudication-metrics",
                    _volume_put(
                        volume,
                        "outputs/blog/manual_review_queue_adjudicated_gpt55_metrics.json",
                        "outputs/blog/",
                        modal_env=modal_env,
                    ),
                ),
            ]
        )
    if include_train:
        steps.append(
            PipelineStep(
                "train-post-gate-tinker",
                _modal_cmd(["run", "modal_app.py::run_post_gate_tinker"], modal_env),
            )
        )
    if include_train_sync:
        steps.append(
            PipelineStep(
                "download-post-gate-train-output",
                _volume_get(
                    volume,
                    "outputs/tinker_rl_post_gate",
                    "outputs/tinker_rl_post_gate",
                    modal_env=modal_env,
                ),
            )
        )
    if include_eval:
        steps.append(
            PipelineStep(
                "eval-post-gate-tinker",
                _modal_cmd(
                    [
                        "run",
                        "modal_app.py::modal_run_experiment_suite",
                        "--suite-name",
                        "post_gate_eval",
                    ],
                    modal_env,
                ),
            )
        )
    if include_eval_sync:
        steps.extend(
            [
                PipelineStep(
                    "download-post-gate-verified-eval",
                    _volume_get(
                        volume,
                        "outputs/eval_tinker_verified_post_gate",
                        "outputs/eval_tinker_verified_post_gate",
                        modal_env=modal_env,
                    ),
                ),
                PipelineStep(
                    "download-post-gate-hard-eval",
                    _volume_get(
                        volume,
                        "outputs/eval_tinker_hard_post_gate",
                        "outputs/eval_tinker_hard_post_gate",
                        modal_env=modal_env,
                    ),
                ),
                PipelineStep(
                    "download-output-blog-artifacts",
                    _volume_get(volume, "outputs/blog", "outputs/blog", modal_env=modal_env),
                ),
                PipelineStep(
                    "download-report-artifacts",
                    _volume_get(volume, "outputs/report", "outputs/report", modal_env=modal_env),
                ),
                PipelineStep(
                    "download-blog",
                    _volume_get(volume, "blog", "blog", modal_env=modal_env),
                ),
            ]
        )
    if include_local_refresh:
        steps.extend(
            [
                PipelineStep(
                    "refresh-second-stage-audit",
                    [sys.executable, "scripts/audit_second_stage.py"],
                    timeout=60,
                ),
                PipelineStep(
                    "refresh-blog",
                    [
                        sys.executable,
                        "-c",
                        (
                            "from rl_epistemics.reporting.write_blog import write_blog; "
                            "write_blog('outputs', 'blog/rl_for_epistemic_humility.md')"
                        ),
                    ],
                    timeout=60,
                ),
            ]
        )
    return steps


def build_local_post_gate_pipeline_steps(
    *,
    tinker_api_key_file: str | None = None,
    python_executable: str | None = None,
    include_train: bool = True,
    include_eval: bool = True,
    include_local_refresh: bool = True,
) -> list[PipelineStep]:
    steps: list[PipelineStep] = []
    python = python_executable or sys.executable
    key_args = ["--tinker-api-key-file", tinker_api_key_file] if tinker_api_key_file else []
    if include_train:
        steps.append(
            PipelineStep(
                "local-train-post-gate-tinker",
                [
                    python,
                    "scripts/train_tinker_rl.py",
                    "--config",
                    "configs/tinker_rl_post_gate.yaml",
                    *key_args,
                ],
            )
        )
    if include_eval:
        steps.append(
            PipelineStep(
                "local-eval-post-gate-tinker",
                [
                    python,
                    "scripts/eval_post_gate_tinker.py",
                    "--train-config",
                    "configs/tinker_rl_post_gate.yaml",
                    *key_args,
                ],
            )
        )
    if include_local_refresh:
        steps.extend(
            [
                PipelineStep(
                    "refresh-report-tables",
                    [
                        python,
                        "-c",
                        (
                            "from rl_epistemics.reporting.tables import write_report_tables; "
                            "write_report_tables('outputs', 'outputs/report')"
                        ),
                    ],
                    timeout=60,
                ),
                PipelineStep(
                    "refresh-failure-reports",
                    [
                        python,
                        "-c",
                        (
                            "from rl_epistemics.reporting.failures import write_failure_reports; "
                            "write_failure_reports('outputs', 'outputs/report', 'outputs/blog')"
                        ),
                    ],
                    timeout=60,
                ),
                PipelineStep(
                    "refresh-blog",
                    [
                        python,
                        "-c",
                        (
                            "from rl_epistemics.reporting.write_blog import write_blog; "
                            "write_blog('outputs', 'blog/rl_for_epistemic_humility.md')"
                        ),
                    ],
                    timeout=60,
                ),
                PipelineStep(
                    "refresh-second-stage-audit",
                    [python, "scripts/audit_second_stage.py"],
                    timeout=60,
                ),
            ]
        )
    return steps


def _run_step(step: PipelineStep, *, dry_run: bool = False) -> dict[str, Any]:
    record = {"name": step.name, "command": step.command, "dry_run": dry_run}
    print(f"\n== {step.name} ==")
    print(" ".join(step.command))
    if dry_run:
        record["returncode"] = None
        return record
    completed = subprocess.run(step.command, check=False, timeout=step.timeout)
    record["returncode"] = completed.returncode
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, step.command)
    return record


def wait_for_modal_status(
    *,
    poll_seconds: int = 60,
    max_wait_seconds: int | None = None,
    status_timeout: int = 20,
) -> dict[str, Any]:
    start = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        status = fetch_modal_status(timeout=status_timeout)
        status["attempts"] = attempts
        if status.get("healthy"):
            return status
        if max_wait_seconds is not None and time.monotonic() - start >= max_wait_seconds:
            raise RuntimeError(
                f"Timed out waiting for Modal status to recover after {attempts} attempts: {status}"
            )
        print(f"Modal status unhealthy on attempt {attempts}: {status}", file=sys.stderr)
        time.sleep(poll_seconds)


def run_post_gate_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config.get("root_dir", ".")).resolve()
    previous_cwd = Path.cwd()
    os.chdir(root)
    try:
        if config.get("local_tinker"):
            status = {"skipped": True, "reason": "local_tinker"}
            needs_local_python = not (
                config.get("skip_train", False)
                and config.get("skip_eval", False)
                and config.get("skip_local_refresh", False)
            )
            local_python = (
                resolve_local_tinker_python(config.get("python_executable"))
                if needs_local_python
                else config.get("python_executable")
            )
            if not config.get("dry_run") and not os.environ.get("TINKER_API_KEY"):
                key_file = config.get("tinker_api_key_file")
                if not key_file:
                    raise RuntimeError(
                        "Local Tinker mode needs TINKER_API_KEY or --tinker-api-key-file."
                    )
                key_path = Path(str(key_file)).expanduser()
                if not key_path.exists() or key_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"Local Tinker key file is missing or empty: {key_path}. "
                        "Run `python3 scripts/write_tinker_key_file.py` first."
                    )
        elif not config.get("skip_status_check"):
            if config.get("wait_for_modal"):
                max_wait = config.get("max_wait_seconds")
                status = wait_for_modal_status(
                    poll_seconds=int(config.get("poll_seconds", 60)),
                    max_wait_seconds=int(max_wait) if max_wait is not None else None,
                    status_timeout=int(config.get("status_timeout", 20)),
                )
            else:
                status = fetch_modal_status(timeout=int(config.get("status_timeout", 20)))
            if not status["healthy"]:
                raise RuntimeError(
                    f"Modal status page reports an active incident; refusing launch: {status}"
                )
        else:
            status = {"skipped": True}

        if config.get("local_tinker"):
            steps = build_local_post_gate_pipeline_steps(
                tinker_api_key_file=config.get("tinker_api_key_file"),
                python_executable=local_python,
                include_train=not config.get("skip_train", False),
                include_eval=not config.get("skip_eval", False),
                include_local_refresh=not config.get("skip_local_refresh", False),
            )
        else:
            steps = build_post_gate_pipeline_steps(
                volume=str(config.get("volume", DEFAULT_VOLUME)),
                modal_env=config.get("modal_env"),
                include_preflight=not config.get("skip_preflight", False),
                include_input_sync=not config.get("skip_input_sync", False),
                include_train=not config.get("skip_train", False),
                include_train_sync=not config.get("skip_train_sync", False),
                include_eval=not config.get("skip_eval", False),
                include_eval_sync=not config.get("skip_eval_sync", False),
                include_local_refresh=not config.get("skip_local_refresh", False),
            )
        records = [_run_step(step, dry_run=bool(config.get("dry_run"))) for step in steps]
        return {"root_dir": str(root), "status": status, "steps": records}
    finally:
        os.chdir(previous_cwd)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the post-gate Tinker train/eval/retrieval pipeline after Modal recovers."
    )
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--modal-env", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-tinker", action="store_true", help="Run train/eval locally instead of through Modal.")
    parser.add_argument("--tinker-api-key-file", default=None)
    parser.add_argument(
        "--python-executable",
        default=None,
        help="Python executable for local Tinker steps; defaults to an auto-detected Python >=3.10 with `tinker` installed.",
    )
    parser.add_argument("--skip-status-check", action="store_true")
    parser.add_argument("--status-timeout", type=int, default=20)
    parser.add_argument("--wait-for-modal", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-seconds", type=int, default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-input-sync", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-train-sync", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-eval-sync", action="store_true")
    parser.add_argument("--skip-local-refresh", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_post_gate_pipeline(vars(args))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
