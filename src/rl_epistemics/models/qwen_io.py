from __future__ import annotations

from typing import Any


def require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for model IO. Install with `uv pip install -e .`."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_id: str) -> Any:
    _, auto_tokenizer = require_transformers()
    tokenizer = auto_tokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_id: str,
    *,
    torch_dtype: Any = "auto",
    device_map: str | dict[str, Any] | None = "auto",
    output_hidden_states: bool = False,
) -> Any:
    auto_model, _ = require_transformers()
    if isinstance(torch_dtype, str) and torch_dtype not in {"auto"}:
        import torch

        dtype_aliases = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_aliases.get(torch_dtype.lower(), torch_dtype)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = auto_model.from_pretrained(model_id, **kwargs)
    if output_hidden_states:
        model.config.output_hidden_states = True
    return model


def apply_chat_template(
    tokenizer: Any,
    prompt: str,
    completion: str | None = None,
    *,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    if completion is not None:
        messages.append({"role": "assistant", "content": completion})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def normalize_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)
