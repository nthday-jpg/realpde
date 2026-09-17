"""submission_v6 — v3 frozen CNO + FIXED relative band (minimal SPS probe).

Same predict-only path as ``submission_v3`` (frozen baseline, ``adapt_loss:
None``), plus a per-element interval returned on EVERY step so the scorer
grades our band instead of its default:

    lower, upper = pred -+ bound_frac * |pred|   # normalized units, pred shape

``bound_frac`` is the half-width fraction (0.05 reproduces the scorer's
default ``±5%`` band; 0.10 / 0.20 / 0.30 test whether the default is too
narrow for this predictor). Set it in ``policy.yaml`` or override at pack
time without editing code::

    python scripts/make_submission_zip.py submission_v6 \\
        --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \\
        --set bound_frac=0.20 --out dist/submission_v6_w20.zip

Bounds live in the prediction's normalized space (same tensor, same device);
the harness denormalizes them exactly like ``pred_norm``. Returned on every
step — the ingestion program enforces all-or-none for the whole run.
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


def _read_policy(submission_dir: str) -> Dict[str, Any]:
    """Flat policy reader: base_model (str), bound_frac (float half-width)."""
    policy: Dict[str, Any] = {"base_model": "cno", "bound_frac": 0.05}
    path = os.path.join(submission_dir, "policy.yaml")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if ":" not in line or line.startswith((" ", "\t")):
                    continue
                k, v = (s.strip() for s in line.split(":", 1))
                if k == "base_model" and v:
                    policy[k] = v.lower()
                elif k == "bound_frac":
                    policy[k] = float(v)
    if not (0.0 < policy["bound_frac"] < 1.0):
        raise ValueError(f"bound_frac must be in (0, 1), got {policy['bound_frac']}")
    return policy


class FixedBandModel(TTTModel):
    """Frozen forecaster + fixed ``pred ± bound_frac*|pred|`` band, every step."""

    def __init__(self, base: nn.Module, device: str, bound_frac: float = 0.05):
        super().__init__()
        self.base = base.to(device)
        self.device = device
        self.bound_frac = bound_frac
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
        w = self.bound_frac * pred_norm.abs()
        return pred_norm, {
            "adapt_loss": None,
            "lower": pred_norm - w,
            "upper": pred_norm + w,
        }


def _build_base(submission_dir: str, device: str, policy: Dict[str, Any]) -> nn.Module:
    ckpt = os.path.join(submission_dir, "model.pth")
    if os.path.exists(ckpt):
        import sys
        if submission_dir not in sys.path:
            sys.path.insert(0, submission_dir)
        from load_baseline import load_baseline  # type: ignore
        base, meta = load_baseline(policy["base_model"], ckpt, device=device)
        print(f"[v6] loaded baseline: {meta}")
        return base
    print("[v6] no model.pth — TinyForecaster fallback (smoke-test only)")
    return TinyForecaster()


def get_ttt_model(submission_dir: str, device: str):
    """Entry point called once by the evaluator (construction is not timed)."""
    policy = _read_policy(submission_dir)
    return FixedBandModel(
        _build_base(submission_dir, device, policy), device, policy["bound_frac"]
    )
