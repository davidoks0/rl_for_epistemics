import pytest

import rl_epistemics.experiments.run_post_gate_pipeline as post_gate_pipeline
from rl_epistemics.experiments.run_post_gate_pipeline import (
    DEFAULT_VOLUME,
    build_local_post_gate_pipeline_steps,
    build_post_gate_pipeline_steps,
    resolve_local_tinker_python,
    wait_for_modal_status,
)


def test_build_post_gate_pipeline_steps_contains_train_eval_and_sync_commands():
    steps = build_post_gate_pipeline_steps(volume=DEFAULT_VOLUME, modal_env="main")
    by_name = {step.name: step for step in steps}

    assert "train-post-gate-tinker" in by_name
    train_command = " ".join(by_name["train-post-gate-tinker"].command)
    eval_command = " ".join(by_name["eval-post-gate-tinker"].command)
    assert "modal run --env main modal_app.py::run_post_gate_tinker" in train_command
    assert "modal run --env main modal_app.py::modal_run_experiment_suite" in eval_command
    assert "post_gate_eval" in eval_command
    assert "outputs/tinker_rl_post_gate" in " ".join(
        by_name["download-post-gate-train-output"].command
    )
    assert "outputs/eval_tinker_verified_post_gate" in " ".join(
        by_name["download-post-gate-verified-eval"].command
    )
    assert "scripts/audit_second_stage.py" in " ".join(
        by_name["refresh-second-stage-audit"].command
    )


def test_build_post_gate_pipeline_steps_can_skip_external_execution():
    steps = build_post_gate_pipeline_steps(
        include_input_sync=False,
        include_train=False,
        include_train_sync=False,
        include_eval=False,
        include_eval_sync=False,
    )
    names = [step.name for step in steps]

    assert names == ["modal-preflight", "refresh-second-stage-audit", "refresh-blog"]


def test_build_local_post_gate_pipeline_steps_uses_key_file_and_refreshes_reports():
    steps = build_local_post_gate_pipeline_steps(
        tinker_api_key_file="/tmp/key",
        python_executable="/opt/project-python",
    )
    by_name = {step.name: step for step in steps}

    train_command = " ".join(by_name["local-train-post-gate-tinker"].command)
    eval_command = " ".join(by_name["local-eval-post-gate-tinker"].command)
    assert train_command.startswith("/opt/project-python ")
    assert "scripts/train_tinker_rl.py --config configs/tinker_rl_post_gate.yaml" in train_command
    assert "--tinker-api-key-file /tmp/key" in train_command
    assert "scripts/eval_post_gate_tinker.py" in eval_command
    assert "--tinker-api-key-file /tmp/key" in eval_command
    assert "refresh-report-tables" in by_name
    assert "refresh-failure-reports" in by_name
    assert "refresh-second-stage-audit" in by_name


def test_run_post_gate_pipeline_local_tinker_skips_modal_status(monkeypatch, tmp_path):
    called = {"status": False}

    def fake_fetch_modal_status(timeout=20):
        called["status"] = True
        return {"healthy": False}

    monkeypatch.setattr(post_gate_pipeline, "fetch_modal_status", fake_fetch_modal_status)
    result = post_gate_pipeline.run_post_gate_pipeline(
        {
            "root_dir": tmp_path,
            "local_tinker": True,
            "dry_run": True,
            "skip_train": True,
            "skip_eval": True,
            "skip_local_refresh": True,
        }
    )

    assert called["status"] is False
    assert result["status"] == {"skipped": True, "reason": "local_tinker"}
    assert result["steps"] == []


def test_run_post_gate_pipeline_local_tinker_requires_key_file(tmp_path):
    with pytest.raises(RuntimeError, match="Local Tinker key file is missing or empty"):
        post_gate_pipeline.run_post_gate_pipeline(
            {
                "root_dir": tmp_path,
                "local_tinker": True,
                "tinker_api_key_file": str(tmp_path / "missing-key"),
                "skip_train": True,
                "skip_eval": True,
                "skip_local_refresh": True,
            }
        )


def test_run_post_gate_pipeline_local_tinker_resolves_python(monkeypatch, tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text("tk-test", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(
        post_gate_pipeline,
        "resolve_local_tinker_python",
        lambda explicit=None: "/opt/project-python",
    )

    def fake_run_step(step, *, dry_run=False):
        captured.setdefault("commands", []).append(step.command)
        return {"name": step.name, "command": step.command, "dry_run": dry_run, "returncode": 0}

    monkeypatch.setattr(post_gate_pipeline, "_run_step", fake_run_step)
    result = post_gate_pipeline.run_post_gate_pipeline(
        {
            "root_dir": tmp_path,
            "local_tinker": True,
            "tinker_api_key_file": str(key_file),
            "dry_run": True,
            "skip_train": False,
            "skip_eval": True,
            "skip_local_refresh": True,
        }
    )

    assert result["status"] == {"skipped": True, "reason": "local_tinker"}
    assert captured["commands"][0][0] == "/opt/project-python"


def test_resolve_local_tinker_python_rejects_bad_explicit(monkeypatch):
    monkeypatch.setattr(
        post_gate_pipeline,
        "_python_supports_local_tinker",
        lambda executable: (False, "No module named tinker"),
    )

    with pytest.raises(RuntimeError, match="Local Tinker mode needs Python"):
        resolve_local_tinker_python("/bad-python")


@pytest.mark.parametrize("skip", [True, False])
def test_build_post_gate_pipeline_steps_preflight_toggle(skip):
    steps = build_post_gate_pipeline_steps(include_preflight=not skip)
    names = [step.name for step in steps]

    assert ("modal-preflight" in names) is (not skip)


def test_wait_for_modal_status_returns_when_provider_is_healthy(monkeypatch):
    calls = []

    def fake_fetch_modal_status(timeout=20):
        calls.append(timeout)
        return {"healthy": len(calls) == 2}

    monkeypatch.setattr(post_gate_pipeline, "fetch_modal_status", fake_fetch_modal_status)
    monkeypatch.setattr(post_gate_pipeline.time, "sleep", lambda seconds: None)

    status = wait_for_modal_status(poll_seconds=1, max_wait_seconds=10, status_timeout=3)

    assert status["healthy"] is True
    assert status["attempts"] == 2
    assert calls == [3, 3]


def test_wait_for_modal_status_times_out(monkeypatch):
    times = iter([0, 2, 4])

    monkeypatch.setattr(post_gate_pipeline, "fetch_modal_status", lambda timeout=20: {"healthy": False})
    monkeypatch.setattr(post_gate_pipeline.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(post_gate_pipeline.time, "monotonic", lambda: next(times))

    with pytest.raises(RuntimeError, match="Timed out waiting for Modal status"):
        wait_for_modal_status(poll_seconds=1, max_wait_seconds=3, status_timeout=3)
