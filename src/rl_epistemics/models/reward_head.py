from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class RewardHeadConfig:
    hidden_size: int
    head_type: str = "matrix_sum"
    scale: float = 0.01

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardHeadConfig":
        return cls(
            hidden_size=int(data["hidden_size"]),
            head_type=str(data.get("head_type", data.get("type", "matrix_sum"))),
            scale=float(data.get("scale", 0.01)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "head_type": self.head_type,
            "scale": self.scale,
        }


class RewardHead(nn.Module):
    def __init__(self, config: RewardHeadConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        if config.head_type == "matrix_sum":
            self.net = nn.Linear(d, d)
        elif config.head_type == "scalar_linear":
            self.net = nn.Linear(d, 1)
        elif config.head_type == "mlp":
            self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        else:
            raise ValueError(f"Unknown reward head type: {config.head_type}")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        out = self.net(hidden)
        if self.config.head_type == "matrix_sum":
            return self.config.scale * out.sum(dim=-1)
        return out.squeeze(-1)

    def save(self, output_dir: str | Path) -> None:
        import json

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target / "reward_head.pt")
        with (target / "reward_head_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def load(cls, output_dir: str | Path, map_location: str | torch.device = "cpu") -> "RewardHead":
        import json

        source = Path(output_dir)
        with (source / "reward_head_config.json").open("r", encoding="utf-8") as f:
            config = RewardHeadConfig.from_dict(json.load(f))
        head = cls(config)
        head.load_state_dict(torch.load(source / "reward_head.pt", map_location=map_location))
        return head

