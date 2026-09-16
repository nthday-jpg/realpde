"""submission_v3 — CNO baseline with NO adaptation (predict-only ablation).

Same CNO base weights as the adapted variants, but ``ttt_step`` never updates
anything: it predicts the current input under ``no_grad`` and reports
``adapt_loss: None``. Use it to isolate how much test-time adaptation actually
adds over the frozen fine-tuned baseline.

Base-model resolution for ``model.pth`` (whose filename carries no
architecture hint): ``policy.yaml: base_model`` > ``BASE_MODEL`` env >
default ``"cno"``. Without ``model.pth`` it falls back to a tiny residual
conv net so the smoke test still runs.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

try:  # optional convenience base class (duck-typed evaluation)
    from ttt_model import TTTModel
except Exception:  # pragma: no cover
    TTTModel = nn.Module  # type: ignore

IN_STEP = 20
CHANNELS = 3  # [u, v, p]


class TinyForecaster(nn.Module):
    """Fallback base model (same as the template): near-identity residual conv."""

    def __init__(self, channels: int = CHANNELS, hidden: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = x.shape
        z = x.reshape(b * t, h, w, c).permute(0, 3, 1, 2)
        z = z + self.conv2(torch.relu(self.conv1(z)))
        return z.permute(0, 2, 3, 1).reshape(b, t, h, w, c)


def _read_base_model(submission_dir: str) -> str:
    """Resolve the baseline architecture hint for model.pth."""
    policy_path = os.path.join(submission_dir, "policy.yaml")
    if os.path.exists(policy_path):
        with open(policy_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if ":" not in line or line.startswith((" ", "\t")):
                    continue
                k, v = (s.strip() for s in line.split(":", 1))
                if k == "base_model" and v:
                    return v.lower()
    return os.environ.get("BASE_MODEL", "cno").strip().lower() or "cno"


class NoAdaptModel(TTTModel):
    """Frozen forecaster: predict-only, no test-time updates."""

    def __init__(self, base: nn.Module, device: str):
        super().__init__()
        self.base = base.to(device)
        self.device = device
        self._init_state = copy.deepcopy(self.base.state_dict())

    def reset_ttt_state(self) -> None:
        self.base.load_state_dict(copy.deepcopy(self._init_state))

    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        input_norm = torch.as_tensor(input_norm).to(self.device)
        self.base.eval()
        with torch.no_grad():
            pred_norm = self.base(input_norm)
        return pred_norm, {"adapt_loss": None}


def _build_base(submission_dir: str, device: str) -> nn.Module:
    ckpt = os.path.join(submission_dir, "model.pth")
    if os.path.exists(ckpt):
        import sys
        if submission_dir not in sys.path:
            sys.path.insert(0, submission_dir)
        from load_baseline import load_baseline  # type: ignore
        base, meta = load_baseline(_read_base_model(submission_dir), ckpt, device=device)
        print(f"[v3] loaded baseline: {meta}")
        return base
    print("[v3] no model.pth — TinyForecaster fallback (smoke-test only)")
    return TinyForecaster()


def get_ttt_model(submission_dir: str, device: str):
    """Entry point called once by the evaluator (construction is not timed)."""
    return NoAdaptModel(_build_base(submission_dir, device), device)
