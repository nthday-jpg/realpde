"""submission_v5 — CNO + online RELATIVE quantile calibration.

v4's absolute band (``pred ± q90``) under-covers fast-flow regions and wastes
width near stagnation. v5 scales the band by local magnitude, per channel:

    rel_res = |prev_target - prev_pred| / (|prev_pred| + eps)   # revealed pair
    q90[channel] = quantile(rel_res, 0.90)                     # table over history
    width = q90 * |pred|;  lower, upper = pred - width, pred + width

All in normalized units. Weights stay frozen (``adapt_loss: None``); bounds
ride in ``info["lower"/"upper"]`` and are returned on EVERY step (ingestion
enforces all-or-none). When the relative table is empty (trajectory start) it
falls back to ``pred ± fallback_frac*|pred|`` (default 0.05, same width as the
scorer's default band).

Knobs in ``policy.yaml``: ``base_model`` (ckpt hint, default ``cno``),
``coverage`` (quantile level, default 0.90), ``history`` (residual windows
kept, default 5), ``table_frames`` (first T time-frames of each window kept
in the table; T=20, default 5), ``fallback_frac`` (relative half-width when
the table is empty, default 0.05). ``eps = 1e-6`` guards the division.
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
EPS = 1e-6


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
    policy: Dict[str, Any] = {
        "base_model": "cno", "coverage": 0.90, "history": 5,
        "table_frames": 5, "fallback_frac": 0.05,
    }
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
            elif k == "table_frames":
                policy[k] = max(int(v), 1)
            elif k == "fallback_frac":
                policy[k] = float(v)
    if not (0.0 < policy["coverage"] < 1.0):
        raise ValueError(f"coverage must be in (0, 1), got {policy['coverage']}")
    return policy


class RelativeCalibratedTTTModel(TTTModel):
    """Frozen forecaster with an online relative-residual quantile table."""

    def __init__(self, base: nn.Module, device: str, coverage: float = 0.90,
                 history: int = 5, table_frames: int = 5,
                 fallback_frac: float = 0.05):
        super().__init__()
        self.base = base.to(device)
        self.device = device
        self.coverage = coverage
        self.history = history
        self.table_frames = table_frames
        self.fallback_frac = fallback_frac
        self._init_state = copy.deepcopy(self.base.state_dict())
        self._resid: deque[torch.Tensor] = deque()
        self._prev_pred: Optional[torch.Tensor] = None

    def _make_bounds(self, pred_norm: torch.Tensor,
                     qrel: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a (lower, upper) pair in pred_norm's shape on every step.

        Uses the calibrated relative factor when the table is ready; otherwise
        falls back to ``pred ± fallback_frac*|pred|`` so bounds are never
        omitted (enforcement: all steps or none).
        """
        f = self.fallback_frac
        if qrel is not None:
            factor = qrel.to(self.device).view(1, 1, 1, 1, MEASURED)
            w_uv = factor * pred_norm[..., :MEASURED].abs()
        else:
            w_uv = f * pred_norm[..., :MEASURED].abs()
        # Unmeasured p channel gets a nominal width (scorer grades u, v only).
        w_p = f * pred_norm[..., MEASURED:].abs()
        lo = torch.cat([pred_norm[..., :MEASURED] - w_uv,
                        pred_norm[..., MEASURED:] - w_p], dim=-1)
        hi = torch.cat([pred_norm[..., :MEASURED] + w_uv,
                        pred_norm[..., MEASURED:] + w_p], dim=-1)
        return lo, hi

    def reset_ttt_state(self) -> None:
        self.base.load_state_dict(copy.deepcopy(self._init_state))
        self._resid.clear()
        self._prev_pred = None

    def _update_table(self, prev_target_norm: torch.Tensor) -> Optional[torch.Tensor]:
        """Fold the revealed previous pair into the relative-residual table.

        Returns per-channel quantile factors, or None when no table yet.
        """
        if self._prev_pred is None:
            return None
        prev_pred = self._prev_pred[..., :MEASURED]
        prev_tgt = prev_target_norm[..., :MEASURED]
        rel = (prev_tgt - prev_pred).abs() / (prev_pred.abs() + EPS)
        # Keep only the first `table_frames` time-frames of the window (T=20)
        # so the quantile table draws on the leading portion of each window.
        rel = rel[..., :self.table_frames, :, :, :MEASURED]
        self._resid.append(rel.detach().reshape(-1, MEASURED).cpu())
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
        qrel: Optional[torch.Tensor] = None
        if prev_target_norm is not None:
            prev_target_norm = torch.as_tensor(prev_target_norm).to(self.device)
            qrel = self._update_table(prev_target_norm)

        # (2) Predict the current window -- no gradients, no current target.
        self.base.eval()
        with torch.no_grad():
            pred_norm = self.base(input_norm)

        # (3) Build a band around the current prediction on EVERY step.
        info: Dict[str, Any] = {"adapt_loss": None}
        lo, hi = self._make_bounds(pred_norm, qrel)
        info["lower"] = lo
        info["upper"] = hi

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
        print(f"[v5] loaded baseline: {meta}")
        return base
    print("[v5] no model.pth — TinyForecaster fallback (smoke-test only)")
    return TinyForecaster()


def get_ttt_model(submission_dir: str, device: str):
    """Entry point called once by the evaluator (construction is not timed)."""
    policy = _read_policy(submission_dir)
    base = _build_base(submission_dir, device, policy)
    return RelativeCalibratedTTTModel(base, device, policy["coverage"], policy["history"],
                                      policy["table_frames"], policy["fallback_frac"])
