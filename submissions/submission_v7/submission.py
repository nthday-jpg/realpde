"""submission_v7 — CNO + EMA quantile calibration (frozen weights + intervals).

v4's band with infinite memory instead of a fixed window: each step folds the
revealed previous pair's per-channel residual quantile into an exponential
moving average, and the band for the CURRENT prediction rides the EMA:

    q_step[channel] = quantile(|prev_target - prev_pred|, coverage)  # revealed pair
    ema   = q_step                     if first update
          = alpha*q_step + (1-alpha)*ema   otherwise
    lower, upper = pred -+ ema         # normalized units, pred shape

``alpha = 1`` is memoryless (current-step q only); smaller alpha smooths over
more history (effective window ~ ``1/alpha`` steps). Weights stay frozen
(``adapt_loss: None``); bounds ride in ``info["lower"/"upper"]`` on EVERY
step (ingestion enforces all-or-none). Trajectory start (no update yet) falls
back to ``pred ± fallback_frac*|pred|`` (default 0.05, scorer-default width).

Sweep without code edits::

    python scripts/make_submission_zip.py submission_v7 \\
        --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \\
        --set ema_alpha=0.30 --out dist/submission_v7_a030.zip

Knobs in ``policy.yaml``: ``base_model`` (ckpt hint, default ``cno``),
``coverage`` (quantile level, default 0.90), ``table_frames`` (first T
time-frames of each window entering the quantile; T=20, default 5),
``fallback_frac`` (relative half-width before the first update, default
0.05), ``ema_alpha`` (EMA weight on the current step, default 0.30).
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
    """Flat policy reader: base_model (str), coverage/table/EMA floats."""
    policy: Dict[str, Any] = {
        "base_model": "cno", "coverage": 0.90, "table_frames": 5,
        "fallback_frac": 0.05, "ema_alpha": 0.30,
    }
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
                elif k == "coverage":
                    policy[k] = float(v)
                elif k == "table_frames":
                    policy[k] = max(int(v), 1)
                elif k == "fallback_frac":
                    policy[k] = float(v)
                elif k == "ema_alpha":
                    policy[k] = float(v)
    if not (0.0 < policy["coverage"] < 1.0):
        raise ValueError(f"coverage must be in (0, 1), got {policy['coverage']}")
    if not (0.0 < policy["ema_alpha"] <= 1.0):
        raise ValueError(f"ema_alpha must be in (0, 1], got {policy['ema_alpha']}")
    return policy


class EmaCalibratedTTTModel(TTTModel):
    """Frozen forecaster with an EMA of per-step residual quantiles."""

    def __init__(self, base: nn.Module, device: str, coverage: float = 0.90,
                 table_frames: int = 5, fallback_frac: float = 0.05,
                 ema_alpha: float = 0.30):
        super().__init__()
        self.base = base.to(device)
        self.device = device
        self.coverage = coverage
        self.table_frames = table_frames
        self.fallback_frac = fallback_frac
        self.ema_alpha = ema_alpha
        self._init_state = copy.deepcopy(self.base.state_dict())
        self._ema: Optional[torch.Tensor] = None  # per-channel half-width, CPU
        self._prev_pred: Optional[torch.Tensor] = None

    def _make_bounds(self, pred_norm: torch.Tensor,
                     half_width: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a (lower, upper) pair in pred_norm's shape on every step."""
        f = self.fallback_frac
        if half_width is not None:
            w_uv = half_width.to(self.device).view(1, 1, 1, 1, MEASURED)
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
        self._ema = None
        self._prev_pred = None

    def _update_ema(self, prev_target_norm: torch.Tensor) -> Optional[torch.Tensor]:
        """Fold the revealed previous pair's quantile into the EMA."""
        if self._prev_pred is None:
            return self._ema
        resid = (prev_target_norm - self._prev_pred).abs()[..., :self.table_frames, :, :, :MEASURED]
        q = torch.quantile(resid.detach().reshape(-1, MEASURED).cpu(),
                           self.coverage, dim=0)
        self._ema = q if self._ema is None else self.ema_alpha * q + (1.0 - self.ema_alpha) * self._ema
        return self._ema

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
            half_width = self._update_ema(prev_target_norm)

        # (2) Predict the current window -- no gradients, no current target.
        self.base.eval()
        with torch.no_grad():
            pred_norm = self.base(input_norm)

        # (3) Build a band around the current prediction on EVERY step.
        info: Dict[str, Any] = {"adapt_loss": None}
        lo, hi = self._make_bounds(pred_norm, half_width)
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
        print(f"[v7] loaded baseline: {meta}")
        return base
    print("[v7] no model.pth — TinyForecaster fallback (smoke-test only)")
    return TinyForecaster()


def get_ttt_model(submission_dir: str, device: str):
    """Entry point called once by the evaluator (construction is not timed)."""
    policy = _read_policy(submission_dir)
    base = _build_base(submission_dir, device, policy)
    return EmaCalibratedTTTModel(base, device, policy["coverage"], policy["table_frames"],
                                policy["fallback_frac"], policy["ema_alpha"])
