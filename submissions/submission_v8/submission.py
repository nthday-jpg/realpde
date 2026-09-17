"""submission_v8 — CNO + SPEED-CONDITIONED quantile band (frozen weights).

v4's residual table, but the q90 is conditioned on predicted flow speed: fast
pixels get the fast-pixel quantile, slow pixels the slow one.

    table (revealed previous pairs, last ``history`` windows):
        r   = |prev_target - prev_pred|            # (..., u, v), normalized
        s   = |prev_pred| magnitude (speed)        # (..., ), normalized
        edges   = quantiles(s, linspace(0, 1, n_bins+1))
        q[b, c] = quantile(r[c] where s in bin b, coverage)
    current step:
        s_cur = |pred| magnitude per pixel
        w[c]  = linear-interp(q[:, c] at s_cur over bin centers)
        lower, upper = pred -+ w

Separate u/v tables (shared speed bins). No EMA — the table is exactly the
last ``history`` windows, like v4. Weights stay frozen (``adapt_loss:
None``); bounds ride in ``info["lower"/"upper"]`` on EVERY step (ingestion
enforces all-or-none). Trajectory start (empty table) falls back to
``pred ± fallback_frac*|pred|`` (default 0.05, scorer-default width).

Knobs in ``policy.yaml``: ``base_model`` (ckpt hint, default ``cno``),
``coverage`` (default 0.90), ``history`` (windows kept, default 2),
``table_frames`` (first T time-frames per window, default 5),
``fallback_frac`` (default 0.05), ``n_bins`` (speed bins, default 10).
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
    """Flat policy reader for the speed-binned calibration knobs."""
    policy: Dict[str, Any] = {
        "base_model": "cno", "coverage": 0.90, "history": 2,
        "table_frames": 5, "fallback_frac": 0.05, "n_bins": 10,
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
                elif k == "history":
                    policy[k] = max(int(v), 1)
                elif k == "table_frames":
                    policy[k] = max(int(v), 1)
                elif k == "fallback_frac":
                    policy[k] = float(v)
                elif k == "n_bins":
                    policy[k] = max(int(v), 2)
    if not (0.0 < policy["coverage"] < 1.0):
        raise ValueError(f"coverage must be in (0, 1), got {policy['coverage']}")
    return policy


class SpeedBinnedTTTModel(TTTModel):
    """Frozen forecaster with a speed-conditioned residual-quantile table."""

    def __init__(self, base: nn.Module, device: str, coverage: float = 0.90,
                 history: int = 2, table_frames: int = 5,
                 fallback_frac: float = 0.05, n_bins: int = 10):
        super().__init__()
        self.base = base.to(device)
        self.device = device
        self.coverage = coverage
        self.history = history
        self.table_frames = table_frames
        self.fallback_frac = fallback_frac
        self.n_bins = n_bins
        self._init_state = copy.deepcopy(self.base.state_dict())
        self._resid: deque[torch.Tensor] = deque()  # (|r_uv|, speed) rows, CPU
        self._prev_pred: Optional[torch.Tensor] = None

    def reset_ttt_state(self) -> None:
        self.base.load_state_dict(copy.deepcopy(self._init_state))
        self._resid.clear()
        self._prev_pred = None

    def _update_table(self, prev_target_norm: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Fold the revealed pair into the table; return (centers, q90) or None."""
        if self._prev_pred is None:
            return None
        prev_pred_uv = self._prev_pred[..., :self.table_frames, :, :, :MEASURED]
        r = (prev_target_norm[..., :self.table_frames, :, :, :MEASURED] - prev_pred_uv).abs()
        s = torch.sqrt((prev_pred_uv ** 2).sum(dim=-1, keepdim=True))  # speed
        self._resid.append(torch.cat([r.reshape(-1, MEASURED),
                                      s.reshape(-1, 1)], dim=1).detach().cpu())
        while len(self._resid) > self.history:
            self._resid.popleft()
        pooled = torch.cat(list(self._resid), dim=0)
        spd = pooled[:, MEASURED]
        levels = torch.linspace(0.0, 1.0, self.n_bins + 1)
        edges = torch.quantile(spd, levels)
        edges[0], edges[-1] = -float("inf"), float("inf")  # cover out-of-range speeds
        # Pooled fallback for degenerate (e.g. zero-speed-tie) bins.
        pooled_q = torch.quantile(pooled[:, :MEASURED], self.coverage, dim=0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        centers = torch.where(torch.isfinite(centers), centers,
                              torch.zeros_like(centers))
        q = torch.empty(self.n_bins, MEASURED)
        for b in range(self.n_bins):
            in_bin = (spd > edges[b]) & (spd <= edges[b + 1])
            if int(in_bin.sum()) == 0:
                q[b] = pooled_q
            else:
                qb = torch.quantile(pooled[in_bin][:, :MEASURED], self.coverage, dim=0)
                q[b] = torch.where(torch.isfinite(qb), qb, pooled_q)
        return centers, q

    def _lookup(self, pred_norm: torch.Tensor,
                table: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Linear-interp per-channel half-width at each pixel's predicted speed."""
        centers, q = table
        centers = centers.to(self.device)
        q = q.to(self.device)  # (B, 2)
        s_cur = torch.sqrt((pred_norm[..., :MEASURED] ** 2).sum(dim=-1))  # (..., )
        s_cur = s_cur.clamp(centers[0], centers[-1])
        i1 = torch.bucketize(s_cur, centers).clamp(1, self.n_bins - 1)
        i0 = i1 - 1
        c0 = centers[i0]
        c1 = centers[i1]
        t = ((s_cur - c0) / (c1 - c0 + EPS)).unsqueeze(-1)  # (..., 1)
        return (1.0 - t) * q[i0] + t * q[i1]  # (..., 2)

    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        input_norm = torch.as_tensor(input_norm).to(self.device)

        # (1) Calibrate on the PREVIOUS (pred, target) pair, both revealed.
        table: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if prev_target_norm is not None:
            prev_target_norm = torch.as_tensor(prev_target_norm).to(self.device)
            table = self._update_table(prev_target_norm)

        # (2) Predict the current window -- no gradients, no current target.
        self.base.eval()
        with torch.no_grad():
            pred_norm = self.base(input_norm)

        # (3) Band around the current prediction on EVERY step.
        f = self.fallback_frac
        if table is not None:
            w_uv = self._lookup(pred_norm, table)
        else:
            w_uv = f * pred_norm[..., :MEASURED].abs()
        # Unmeasured p channel gets a nominal width (scorer grades u, v only).
        w_p = f * pred_norm[..., MEASURED:].abs()
        info: Dict[str, Any] = {
            "adapt_loss": None,
            "lower": torch.cat([pred_norm[..., :MEASURED] - w_uv,
                                pred_norm[..., MEASURED:] - w_p], dim=-1),
            "upper": torch.cat([pred_norm[..., :MEASURED] + w_uv,
                                pred_norm[..., MEASURED:] + w_p], dim=-1),
        }

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
        print(f"[v8] loaded baseline: {meta}")
        return base
    print("[v8] no model.pth — TinyForecaster fallback (smoke-test only)")
    return TinyForecaster()


def get_ttt_model(submission_dir: str, device: str):
    """Entry point called once by the evaluator (construction is not timed)."""
    policy = _read_policy(submission_dir)
    base = _build_base(submission_dir, device, policy)
    return SpeedBinnedTTTModel(base, device, policy["coverage"], policy["history"],
                               policy["table_frames"], policy["fallback_frac"],
                               policy["n_bins"])
