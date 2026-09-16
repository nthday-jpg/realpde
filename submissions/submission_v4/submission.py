"""submission_v4 — CNO + online quantile calibration (frozen weights + intervals).

v3's frozen CNO predictions plus a per-step uncertainty band for the SPS
subscore. Each step (after the first) does a table lookup:

    residual = |prev_target - prev_pred|      # revealed previous pair, normalized units
    q90[channel] = quantile(residual, 0.90)   # per-channel table over recent history
    lower, upper = pred - q90, pred + q90      # band for the CURRENT prediction

Only genuinely revealed data is used (previous pair); the current target is
never touched. Weights stay frozen — no gradient steps, so the Time cost over
v3 is one small quantile per step. Bounds ride in ``info["lower"/"upper"]``
(normalized space, model device); when the table is empty (first step) the
keys are omitted and the scorer falls back to its default ±5% band.

Knobs in ``policy.yaml``: ``base_model`` (ckpt hint, default ``cno``),
``coverage`` (quantile level, default 0.90), ``history`` (residual windows
kept, default 5).
"""

from __future__ import annotations

import copy
import os
from collections import deque
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

try:  # optional convenience base class (duck-typed evaluation)
    from ttt_model import TTTModel
except Exception:  # pragma: no cover
    TTTModel = nn.Module  # type: ignore

IN_STEP = 20
CHANNELS = 3  # [u, v, p]; calibration covers measured u, v only
MEASURED = 2


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
    """Flat policy reader: base_model (str), coverage (float), history (int)."""
    policy: Dict[str, Any] = {"base_model": "cno", "coverage": 0.90, "history": 5}
    path = os.path.join(submission_dir, "policy.yaml")
    if not os.path.exists(path):
        return policy
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if ":" not in line or line.startswith((" ", "\t")):
                continue
            k, v = (s.strip() for s in line.split(":", 1))
            if k == "base_model" and v:
                policy[k] = v.lower()
            elif k == "coverage":
                policy[k] = float(v)
            elif k == "history":
                policy[k] = max(int(v), 1)
    if not (0.0 < policy["coverage"] < 1.0):
        raise ValueError(f"coverage must be in (0, 1), got {policy['coverage']}")
    return policy


class CalibratedTTTModel(TTTModel):
    """Frozen forecaster with an online residual-quantile interval table."""

    def __init__(self, base: nn.Module, device: str, coverage: float = 0.90,
                 history: int = 5):
        super().__init__()
        self.base = base.to(device)
        self.device = device
        self.coverage = coverage
        self.history = history
        self._init_state = copy.deepcopy(self.base.state_dict())
        self._resid: deque[torch.Tensor] = deque()
        self._prev_pred: Optional[torch.Tensor] = None

    def reset_ttt_state(self) -> None:
        self.base.load_state_dict(copy.deepcopy(self._init_state))
        self._resid.clear()
        self._prev_pred = None

    def _update_table(self, prev_target_norm: torch.Tensor) -> Optional[torch.Tensor]:
        """Fold the revealed previous pair into the residual table.

        Returns per-channel quantile half-widths, or None when no table yet.
        """
        if self._prev_pred is None:
            return None
        resid = (prev_target_norm - self._prev_pred).abs()[..., :MEASURED]
        self._resid.append(resid.detach().reshape(-1, MEASURED).cpu())
        while len(self._resid) > self.history:
            self._resid.popleft()
        pooled = torch.cat(list(self._resid), dim=0)
        return torch.quantile(pooled, self.coverage, dim=0)

    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        input_norm = torch.as_tensor(input_norm).to(self.device)

        # (1) Calibrate on the PREVIOUS (pred, target) pair, both revealed.
        half_width: Optional[torch.Tensor] = None
        if prev_target_norm is not None:
            prev_target_norm = torch.as_tensor(prev_target_norm).to(self.device)
            half_width = self._update_table(prev_target_norm)

        # (2) Predict the current window -- no gradients, no current target.
        self.base.eval()
        with torch.no_grad():
            pred_norm = self.base(input_norm)

        # (3) Table lookup: band around the current prediction.
        info: Dict[str, Any] = {"adapt_loss": None}
        if half_width is not None:
            hw = half_width.to(self.device).view(1, 1, 1, 1, MEASURED)
            lo_uv = pred_norm[..., :MEASURED] - hw
            hi_uv = pred_norm[..., :MEASURED] + hw
            lo_p = hi_p = pred_norm[..., MEASURED:]
            info["lower"] = torch.cat([lo_uv, lo_p], dim=-1)
            info["upper"] = torch.cat([hi_uv, hi_p], dim=-1)

        self._prev_pred = pred_norm.detach()

        return pred_norm, info


def _build_base(submission_dir: str, device: str, policy: Dict[str, Any]) -> nn.Module:
    ckpt = os.path.join(submission_dir, "model.pth")
    if os.path.exists(ckpt):
        import sys
        if submission_dir not in sys.path:
            sys.path.insert(0, submission_dir)
        from load_baseline import load_baseline  # type: ignore
        base, meta = load_baseline(policy["base_model"], ckpt, device=device)
        print(f"[v4] loaded baseline: {meta}")
        return base
    print("[v4] no model.pth — TinyForecaster fallback (smoke-test only)")
    return TinyForecaster()


def get_ttt_model(submission_dir: str, device: str):
    """Entry point called once by the evaluator (construction is not timed)."""
    policy = _read_policy(submission_dir)
    base = _build_base(submission_dir, device, policy)
    return CalibratedTTTModel(base, device, policy["coverage"], policy["history"])
