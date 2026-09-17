"""submission_v10 — CNO + 1-step SGD + v4 calibration band.

v9's adaptation loop (one SGD step on the revealed previous pair, then
predict) plus v4's online per-channel residual-quantile interval on EVERY
step:

    residual = |prev_target - prev_pred|      # revealed previous pair
    q90[channel] = quantile(residual, 0.90)   # last `history` windows
    lower, upper = pred -+ q90                # CURRENT prediction, normalized

Only revealed data is used; the current target is never touched. Trajectory
start (empty table) falls back to ``pred ± fallback_frac*|pred|`` (default
0.05, scorer-default width) so bounds ship on every step (all-or-none).

Knobs in ``policy.yaml``: ``base_model`` (default ``cno``), ``ttt_lr``
(default 1e-3), ``coverage`` (default 0.90), ``history`` (default 2),
``table_frames`` (default 5), ``fallback_frac`` (default 0.05).
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
    """Flat policy reader: adaptation + calibration knobs."""
    policy: Dict[str, Any] = {
        "base_model": "cno", "ttt_lr": 1e-3, "coverage": 0.90, "history": 2,
        "table_frames": 5, "fallback_frac": 0.05,
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
                elif k == "ttt_lr":
                    policy[k] = float(v)
                elif k == "coverage":
                    policy[k] = float(v)
                elif k == "history":
                    policy[k] = max(int(v), 1)
                elif k == "table_frames":
                    policy[k] = max(int(v), 1)
                elif k == "fallback_frac":
                    policy[k] = float(v)
    if not policy["ttt_lr"] > 0.0:
        raise ValueError(f"ttt_lr must be positive, got {policy['ttt_lr']}")
    if not (0.0 < policy["coverage"] < 1.0):
        raise ValueError(f"coverage must be in (0, 1), got {policy['coverage']}")
    return policy


class AdaptCalibratedTTTModel(TTTModel):
    """One SGD step on the previous pair + q90 band on the current prediction."""

    def __init__(self, base: nn.Module, device: str, ttt_lr: float = 1e-3,
                 coverage: float = 0.90, history: int = 2,
                 table_frames: int = 5, fallback_frac: float = 0.05):
        super().__init__()
        self.base = base.to(device)
        self.device = device
        self.ttt_lr = ttt_lr
        self.coverage = coverage
        self.history = history
        self.table_frames = table_frames
        self.fallback_frac = fallback_frac
        self._init_state = copy.deepcopy(self.base.state_dict())
        self._opt = torch.optim.SGD(self.base.parameters(), lr=ttt_lr)
        self._prev_input: Optional[torch.Tensor] = None
        self._resid: deque[torch.Tensor] = deque()
        self._prev_pred: Optional[torch.Tensor] = None

    def reset_ttt_state(self) -> None:
        self.base.load_state_dict(copy.deepcopy(self._init_state))
        self._opt = torch.optim.SGD(self.base.parameters(), lr=self.ttt_lr)
        self._prev_input = None
        self._resid.clear()
        self._prev_pred = None

    def _update_table(self, prev_target_norm: torch.Tensor) -> Optional[torch.Tensor]:
        """Fold the revealed previous pair into the residual table."""
        if self._prev_pred is None:
            return None
        resid = (prev_target_norm - self._prev_pred).abs()[..., :self.table_frames, :, :, :MEASURED]
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

        # (1) ONE gradient step on the PREVIOUS (input, target) pair.
        adapt_loss: Optional[float] = None
        half_width: Optional[torch.Tensor] = None
        if prev_target_norm is not None and self._prev_input is not None:
            prev_target_norm = torch.as_tensor(prev_target_norm).to(self.device)
            half_width = self._update_table(prev_target_norm)
            self.base.train()
            self._opt.zero_grad()
            pred_prev = self.base(self._prev_input)
            loss = torch.mean((pred_prev - prev_target_norm) ** 2)
            loss.backward()
            self._opt.step()
            adapt_loss = float(loss.detach().cpu())

        # (2) Predict the current window -- no gradients, no current target.
        self.base.eval()
        with torch.no_grad():
            pred_norm = self.base(input_norm)

        # (3) Band around the current prediction on EVERY step.
        f = self.fallback_frac
        if half_width is not None:
            w_uv = half_width.to(self.device).view(1, 1, 1, 1, MEASURED)
        else:
            w_uv = f * pred_norm[..., :MEASURED].abs()
        # Unmeasured p channel gets a nominal width (scorer grades u, v only).
        w_p = f * pred_norm[..., MEASURED:].abs()
        info: Dict[str, Any] = {
            "adapt_loss": adapt_loss,
            "lower": torch.cat([pred_norm[..., :MEASURED] - w_uv,
                                pred_norm[..., MEASURED:] - w_p], dim=-1),
            "upper": torch.cat([pred_norm[..., :MEASURED] + w_uv,
                                pred_norm[..., MEASURED:] + w_p], dim=-1),
        }

        # (4) Cache the current input AND prediction for the next step.
        self._prev_input = input_norm.detach()
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
        print(f"[v10] loaded baseline: {meta}")
        return base
    print("[v10] no model.pth — TinyForecaster fallback (smoke-test only)")
    return TinyForecaster()


def get_ttt_model(submission_dir: str, device: str):
    """Entry point called once by the evaluator (construction is not timed)."""
    policy = _read_policy(submission_dir)
    base = _build_base(submission_dir, device, policy)
    return AdaptCalibratedTTTModel(base, device, policy["ttt_lr"], policy["coverage"],
                                  policy["history"], policy["table_frames"],
                                  policy["fallback_frac"])
