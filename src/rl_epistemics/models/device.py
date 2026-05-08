from __future__ import annotations

from typing import Any

import torch


def select_torch_device(requested: str | None = "auto") -> torch.device:
    name = str(requested or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested, but CUDA is not available.")
    if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("device='mps' requested, but MPS is not available.")
    return torch.device(name)


def model_device_map(requested: str | None, selected: torch.device) -> str | dict[str, Any] | None:
    name = str(requested or "auto").lower()
    if selected.type in {"cpu", "mps"}:
        return None
    if name in {"cpu", "mps", "cuda"}:
        return None
    return name


def model_load_dtype(torch_dtype: Any, selected: torch.device) -> Any:
    if selected.type == "mps" and (torch_dtype is None or str(torch_dtype).lower() == "auto"):
        return "float16"
    return "auto" if torch_dtype is None else torch_dtype
