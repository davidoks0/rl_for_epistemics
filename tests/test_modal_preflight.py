from rl_epistemics.experiments.modal_preflight import modal_status_is_healthy, summarize_modal_state


def test_modal_status_is_healthy_detects_outage_markers():
    assert modal_status_is_healthy("# All services are online")
    assert not modal_status_is_healthy("# Some services are down")
    assert not modal_status_is_healthy("Internal data stores are failing")


def test_summarize_modal_state_flags_secret_and_active_detached_tasks():
    summary = summarize_modal_state(
        profile="davidoks0",
        secrets=[{"Name": "tinker-api-key"}],
        containers=[
            {
                "Container ID": "ta-1",
                "App ID": "ap-1",
                "App Name": "kurrent-pylaia-training",
            }
        ],
        apps=[
            {
                "App ID": "ap-1",
                "Description": "kurrent-pylaia-training",
                "State": "ephemeral (detached)",
                "Tasks": "1",
            },
            {
                "App ID": "ap-2",
                "Description": "rl-for-epistemics",
                "State": "stopped",
                "Tasks": "0",
            },
        ],
        provider_status={
            "healthy": False,
            "contains_internal_datastore_incident": True,
        },
    )

    assert summary["profile"] == "davidoks0"
    assert summary["provider_outage_detected"] is True
    assert summary["required_secret_present"] is True
    assert summary["active_container_count"] == 1
    assert summary["active_detached_task_app_count"] == 1
    assert summary["capacity_risk"] is True
    assert summary["scheduling_blocker"] == "provider_status_outage"
    assert summary["recent_rl_for_epistemics_apps"][0]["App ID"] == "ap-2"


def test_summarize_modal_state_handles_no_active_apps():
    summary = summarize_modal_state(
        profile="davidoks0",
        secrets=[],
        containers=[],
        apps=[{"App ID": "ap-1", "Description": "rl-for-epistemics", "State": "stopped"}],
        provider_status={"healthy": True},
    )

    assert summary["provider_outage_detected"] is False
    assert summary["required_secret_present"] is False
    assert summary["active_container_count"] == 0
    assert summary["active_task_app_count"] == 0
    assert summary["capacity_risk"] is False
    assert summary["scheduling_blocker"] is None
