"""submission_v9 — CNO + ONE gradient step per step (plain TTT ablation).

The reference adaptation loop from ``submission_template.py`` on the real
baseline: each ``ttt_step`` takes a single SGD step supervising the cached
previous input against the revealed previous target (a genuine input->target
pair, never the current target), then predicts the current input.

No uncertainty intervals are returned, so the scorer grades the default
``pred ± 0.05*|pred|`` band. Pair with v10 (same adaptation + v4 band) to
isolate what adaptation adds over v3 (frozen) and what calibration adds over
adaptation alone.

Knobs in ``policy.yaml``: ``base_model`` (ckpt hint, default ``cno``),
``ttt_lr`` (SGD learning rate, default 1e-3). Override at pack time::

    python scripts/make_submission_zip.py submission_v9 \\
        --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \\
        --set ttt_lr=0.0005 --out dist/submission_v9_lr05.zip
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
    """Flat policy reader: base_model (str), ttt_lr (float)."""
    policy: Dict[str, Any] = {"base_model": "cno", "ttt_lr": 1e-3}
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
    if not policy["ttt_lr"] > 0.0:
        raise ValueError(f"ttt_lr must be positive, got {policy['ttt_lr']}")
    return policy


class OneStepTTTModel(TTTModel):
    """One SGD step on the previous pair, then predict (no intervals)."""

    def __init__(self, base: nn.Module, device: str, ttt_lr: float = 1e-3):
        super().__init__()
        self.base = base.to(device)
        self.device = device
        self.ttt_lr = ttt_lr
        self._init_state = copy.deepcopy(self.base.state_dict())
        self._opt = torch.optim.SGD(self.base.parameters(), lr=ttt_lr)
        self._prev_input: Optional[torch.Tensor] = None

    def reset_ttt_state(self) -> None:
        self.base.load_state_dict(copy.deepcopy(self._init_state))
        self._opt = torch.optim.SGD(self.base.parameters(), lr=self.ttt_lr)
        self._prev_input = None

    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        input_norm = torch.as_tensor(input_norm).to(self.device)

        # (1) ONE gradient step on the PREVIOUS (input, target) pair.
        adapt_loss: Optional[float] = None
        if prev_target_norm is not None and self._prev_input is not None:
            prev_target_norm = torch.as_tensor(prev_target_norm).to(self.device)
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

        # (3) Cache the current input for the next step's adaptation.
        self._prev_input = input_norm.detach()

        return pred_norm, {"adapt_loss": adapt_loss}


def _build_base(submission_dir: str, device: str, policy: Dict[str, Any]) -> nn.Module:
    ckpt = os.path.join(submission_dir, "model.pth")
    if os.path.exists(ckpt):
        import sys
        if submission_dir not in sys.path:
            sys.path.insert(0, submission_dir)
        from load_baseline import load_baseline  # type: ignore
        base, meta = load_baseline(policy["base_model"], ckpt, device=device)
        print(f"[v9] loaded baseline: {meta}")
        return base
    print("[v9] no model.pth — TinyForecaster fallback (smoke-test only)")
    return TinyForecaster()


def get_ttt_model(submission_dir: str, device: str):
    """Entry point called once by the evaluator (construction is not timed)."""
    policy = _read_policy(submission_dir)
    return OneStepTTTModel(
        _build_base(submission_dir, device, policy), device, policy["ttt_lr"]
    )
