from rl_epistemics.eval.llm_judge import maybe_judge_rows


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _row():
    return {
        "id": "fake-api",
        "prompt_type": "fake",
        "domain": "software_api",
        "entity_name": "requests.quantum_retry",
        "prompt": "How do I use requests.quantum_retry?",
        "completion": "I cannot verify that API.",
        "label": "clean_uncertainty",
        "verification_status": "negative_retrieval_required",
        "verification_evidence": "Requests docs search: no result",
    }


def test_openai_judge_provider_uses_responses_shape(monkeypatch):
    calls = []

    def fake_post(endpoint, headers, json, timeout):
        calls.append({"endpoint": endpoint, "headers": headers, "json": json, "timeout": timeout})
        return _FakeResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    '{"label": "clean_uncertainty", "confidence": "high", '
                                    '"needs_human_review": false, "notes": "No invented details."}'
                                ),
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("rl_epistemics.eval.llm_judge.requests.post", fake_post)

    judged = maybe_judge_rows(
        [_row()],
        {
            "llm_judge": {
                "enabled": True,
                "provider": "openai",
                "api_key_env": "TEST_OPENAI_API_KEY",
                "model": "gpt-5.5",
                "max_tokens": 128,
                "reasoning_effort": "high",
            }
        },
    )

    assert judged[0]["llm_judge_label"] == "clean_uncertainty"
    assert judged[0]["llm_judge_confidence"] == "high"
    assert judged[0]["llm_judge_needs_human_review"] is False
    assert judged[0]["llm_judge_provider"] == "openai"
    assert calls[0]["endpoint"] == "https://api.openai.com/v1/responses"
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert "Rubric:" in calls[0]["json"]["instructions"]
    assert "hedged_confabulation rather than clean_uncertainty" in calls[0]["json"]["instructions"]
    assert calls[0]["json"]["text"]["format"]["type"] == "json_schema"
    assert calls[0]["json"]["text"]["format"]["strict"] is True
    assert calls[0]["json"]["reasoning"] == {"effort": "high"}
    assert calls[0]["json"]["max_output_tokens"] == 128


def test_openai_judge_still_supports_chat_completions_endpoint(monkeypatch):
    calls = []

    def fake_post(endpoint, headers, json, timeout):
        calls.append({"endpoint": endpoint, "headers": headers, "json": json, "timeout": timeout})
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"label": "clean_uncertainty", "confidence": "medium", '
                                '"needs_human_review": false, "notes": "No invented details."}'
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("rl_epistemics.eval.llm_judge.requests.post", fake_post)

    judged = maybe_judge_rows(
        [_row()],
        {
            "llm_judge": {
                "enabled": True,
                "provider": "openai",
                "api_key_env": "TEST_OPENAI_API_KEY",
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "model": "gpt-4o-mini",
                "max_tokens": 128,
            }
        },
    )

    assert judged[0]["llm_judge_label"] == "clean_uncertainty"
    assert judged[0]["llm_judge_confidence"] == "medium"
    assert calls[0]["endpoint"] == "https://api.openai.com/v1/chat/completions"
    assert calls[0]["json"]["messages"][0]["role"] == "system"
    assert calls[0]["json"]["response_format"]["type"] == "json_schema"
    assert calls[0]["json"]["response_format"]["json_schema"]["strict"] is True
    assert calls[0]["json"]["max_tokens"] == 128


def test_openai_judge_accepts_lowercase_env_alias(monkeypatch):
    calls = []

    def fake_post(endpoint, headers, json, timeout):
        calls.append(headers)
        return _FakeResponse(
            {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    '{"label": "clean_uncertainty", "confidence": "high", '
                                    '"needs_human_review": false, "notes": "No invented details."}'
                                ),
                            }
                        ]
                    }
                ]
            }
        )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("openai_api_key", "lowercase-test-key")
    monkeypatch.setattr("rl_epistemics.eval.llm_judge.requests.post", fake_post)

    judged = maybe_judge_rows(
        [_row()],
        {"llm_judge": {"enabled": True, "provider": "openai", "model": "gpt-5.5"}},
    )

    assert judged[0]["llm_judge_label"] == "clean_uncertainty"
    assert calls[0]["Authorization"] == "Bearer lowercase-test-key"


def test_openai_judge_can_read_api_key_file(tmp_path, monkeypatch):
    calls = []
    key_file = tmp_path / "openai_key"
    key_file.write_text("file-test-key\n", encoding="utf-8")

    def fake_post(endpoint, headers, json, timeout):
        calls.append(headers)
        return _FakeResponse(
            {
                "output_text": (
                    '{"label": "clean_uncertainty", "confidence": "high", '
                    '"needs_human_review": false, "notes": "No invented details."}'
                ),
                "output": [
                    {
                        "content": []
                    }
                ]
            }
        )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("openai_api_key", raising=False)
    monkeypatch.setattr("rl_epistemics.eval.llm_judge.requests.post", fake_post)

    judged = maybe_judge_rows(
        [_row()],
        {
            "llm_judge": {
                "enabled": True,
                "provider": "openai",
                "model": "gpt-5.5",
                "api_key_file": str(key_file),
            }
        },
    )

    assert judged[0]["llm_judge_label"] == "clean_uncertainty"
    assert calls[0]["Authorization"] == "Bearer file-test-key"


def test_openai_judge_no_output_marks_unclear_for_human_review(monkeypatch):
    def fake_post(endpoint, headers, json, timeout):
        return _FakeResponse({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}})

    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("rl_epistemics.eval.llm_judge.requests.post", fake_post)

    judged = maybe_judge_rows(
        [_row()],
        {
            "llm_judge": {
                "enabled": True,
                "provider": "openai",
                "api_key_env": "TEST_OPENAI_API_KEY",
                "model": "gpt-5.5",
            }
        },
    )

    assert judged[0]["llm_judge_label"] == "unclear"
    assert judged[0]["llm_judge_confidence"] == "low"
    assert judged[0]["llm_judge_needs_human_review"] is True
    assert "no output text" in judged[0]["llm_judge_notes"]


def test_anthropic_judge_provider_uses_messages_shape(monkeypatch):
    calls = []

    def fake_post(endpoint, headers, json, timeout):
        calls.append({"endpoint": endpoint, "headers": headers, "json": json, "timeout": timeout})
        return _FakeResponse(
            {
                "content": [
                    {
                        "type": "text",
                        "text": "```json\n{\"label\": \"false_premise_correction\", \"notes\": \"Corrected the premise.\"}\n```",
                    }
                ]
            }
        )

    monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("rl_epistemics.eval.llm_judge.requests.post", fake_post)

    judged = maybe_judge_rows(
        [_row()],
        {
            "llm_judge": {
                "enabled": True,
                "provider": "anthropic",
                "api_key_env": "TEST_ANTHROPIC_API_KEY",
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 96,
            }
        },
    )

    assert judged[0]["llm_judge_label"] == "false_premise_correction"
    assert judged[0]["llm_judge_provider"] == "anthropic"
    assert calls[0]["endpoint"] == "https://api.anthropic.com/v1/messages"
    assert calls[0]["headers"]["x-api-key"] == "test-key"
    assert calls[0]["headers"]["anthropic-version"] == "2023-06-01"
    assert calls[0]["json"]["system"]
    assert calls[0]["json"]["messages"] == [{"role": "user", "content": calls[0]["json"]["messages"][0]["content"]}]
    assert calls[0]["json"]["max_tokens"] == 96


def test_anthropic_provider_can_be_inferred_from_model_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    judged = maybe_judge_rows(
        [_row()],
        {"llm_judge": {"enabled": True, "model": "claude-sonnet-4-20250514"}},
    )

    assert judged[0]["llm_judge_label"] is None
    assert "ANTHROPIC_API_KEY" in judged[0]["llm_judge_notes"]
