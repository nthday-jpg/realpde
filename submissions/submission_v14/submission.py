"""submission_v14 — CNO + FULL multi-step gradient update (MSE + temporal-difference).

Every ``ttt_step`` with a revealed previous pair runs ``ttt_steps`` full-parameter
gradient steps (all weights, not LoRA) on::

    loss = MSE(pred_prev, prev_target)
         + td_lambda * MSE(diff(pred_prev), diff(prev_target))

where ``diff`` is the 1-step temporal finite difference along the T axis
(``x[:, 1:] - x[:, :-1]``) on the measured (u, v) channels. MSE anchors the
level; the TD term anchors the dynamics so adaptation cannot buy a lower
instantaneous error by flattening temporal variability (which is exactly what
the TKE scalar hides).

Only revealed data is used (cached previous input vs revealed previous
target); the current target is never touched. After adapting, the current
input is predicted with no gradients. No uncertainty intervals are returned,
so the scorer grades the default ``pred ± 0.05*|pred|`` band (same setup as
v9, isolating what the MSE + TD update adds over plain 1-step SGD).

Knobs in ``policy.yaml``: ``base_model`` (default ``cno``), ``ttt_lr``
(default 1e-3), ``ttt_steps`` (inner gradient steps, default 3),
``optimizer`` (``sgd``|``adam``, default ``sgd``), ``td_lambda`` (default 0.1),
``grad_clip`` (global norm clip, default 1.0, 0 disables).

``info`` carries ``adapt_loss`` (final total loss, read by the evaluator)
plus ``mse_loss`` / ``td_loss`` components for logging (ignored by evaluator).

TKE spatial logging lives in ``tke_maps.py`` next to this file: it replays the
streaming loop, computes per-window KE(x) = 1/2[Var_t(u) + Var_t(v)] maps for
prediction and target, and writes ``KE_TTA - KE_target`` difference maps.
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
CHANNELS = 3  # [u, v, p]; TD covers measured u, v only
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
    """Flat policy reader for adaptation + TD knobs."""
    policy: Dict[str, Any] = {
        "base_model": "cno",
        "ttt_lr": 1e-3,
        "ttt_steps": 3,
        "optimizer": "sgd",
        "td_lambda": 0.1,
        "grad_clip": 1.0,
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
                elif k == "ttt_steps":
                    policy[k] = max(int(v), 1)
                elif k == "optimizer" and v:
                    policy[k] = v.lower()
                elif k == "td_lambda":
                    policy[k] = float(v)
                elif k == "grad_clip":
                    policy[k] = float(v)
    if not policy["ttt_lr"] > 0.0:
        raise ValueError(f"ttt_lr must be positive, got {policy['ttt_lr']}")
    if policy["optimizer"] not in ("sgd", "adam"):
        raise ValueError(f"optimizer must be sgd|adam, got {policy['optimizer']}")
    if not policy["td_lambda"] >= 0.0:
        raise ValueError(f"td_lambda must be >= 0, got {policy['td_lambda']}")
    return policy


def td_mismatch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between 1-step temporal differences on the measured channels.

    pred/target: (B, T, H, W, C) with T >= 2. Returns a scalar tensor.
    """
    dp = pred[..., 1:, :, :, :MEASURED] - pred[..., :-1, :, :, :MEASURED]
    dt = target[..., 1:, :, :, :MEASURED] - target[..., :-1, :, :, :MEASURED]
    return torch.mean((dp - dt) ** 2)


class FullGradMsTdTTTModel(TTTModel):
    """Full-parameter multi-step TTT with MSE + temporal-difference loss."""

    def __init__(
        self,
        base: nn.Module,
        device: str,
        ttt_lr: float = 1e-3,
        ttt_steps: int = 3,
        optimizer: str = "sgd",
        td_lambda: float = 0.1,
        grad_clip: float = 1.0,
    ):
        super().__init__()
        self.base = base.to(device)
        self.device = device
        self.ttt_lr = ttt_lr
        self.ttt_steps = ttt_steps
        self.optimizer_name = optimizer
        self.td_lambda = td_lambda
        self.grad_clip = grad_clip
        self._init_state = copy.deepcopy(self.base.state_dict())
        self._opt = self._make_opt()
        self._prev_input: Optional[torch.Tensor] = None

    def _make_opt(self) -> torch.optim.Optimizer:
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.base.parameters(), lr=self.ttt_lr)
        return torch.optim.SGD(self.base.parameters(), lr=self.ttt_lr)

    def reset_ttt_state(self) -> None:
        self.base.load_state_dict(copy.deepcopy(self._init_state))
        self._opt = self._make_opt()
        self._prev_input = None

    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        input_norm = torch.as_tensor(input_norm).to(self.device)

        # (1) FULL update on the PREVIOUS (input, target) pair: ttt_steps
        # gradient steps on all parameters with MSE + TD regularization.
        adapt_loss: Optional[float] = None
        mse_f: Optional[float] = None
        td_f: Optional[float] = None
        if prev_target_norm is not None and self._prev_input is not None:
            prev_target_norm = torch.as_tensor(prev_target_norm).to(self.device)
            self.base.train()
            for _ in range(self.ttt_steps):
                self._opt.zero_grad()
                pred_prev = self.base(self._prev_input)
                mse = torch.mean((pred_prev - prev_target_norm) ** 2)
                td = td_mismatch(pred_prev, prev_target_norm)
                loss = mse + self.td_lambda * td
                loss.backward()
                if self.grad_clip and self.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(self.base.parameters(), self.grad_clip)
                self._opt.step()
            adapt_loss = float(loss.detach().cpu())
            mse_f = float(mse.detach().cpu())
            td_f = float(td.detach().cpu())

        # (2) Predict the current window -- no gradients, no current target.
        self.base.eval()
        with torch.no_grad():
            pred_norm = self.base(input_norm)

        # (3) Cache the current input for the next step's adaptation.
        self._prev_input = input_norm.detach()

        return pred_norm, {
            "adapt_loss": adapt_loss,
            "mse_loss": mse_f,
            "td_loss": td_f,
        }


def _build_base(submission_dir: str, device: str, policy: Dict[str, Any]) -> nn.Module:
    ckpt = os.path.join(submission_dir, "model.pth")
    if os.path.exists(ckpt):
        import sys
        if submission_dir not in sys.path:
            sys.path.insert(0, submission_dir)
        from load_baseline import load_baseline  # type: ignore
        base, meta = load_baseline(policy["base_model"], ckpt, device=device)
        print(f"[v14] loaded baseline: {meta}")
        return base
    print("[v14] no model.pth — TinyForecaster fallback (smoke-test only)")
    return TinyForecaster()


def get_ttt_model(submission_dir: str, device: str):
    """Entry point called once by the evaluator (construction is not timed)."""
    policy = _read_policy(submission_dir)
    base = _build_base(submission_dir, device, policy)
    return FullGradMsTdTTTModel(
        base,
        device,
        policy["ttt_lr"],
        policy["ttt_steps"],
        policy["optimizer"],
        policy["td_lambda"],
        policy["grad_clip"],
    )
