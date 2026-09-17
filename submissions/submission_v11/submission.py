"""submission_v11 — CNO + test-time LoRA (1-step SGD on adapters only).

v9's loop (one gradient step on the revealed previous pair, then predict),
but the update touches ONLY low-rank adapters instead of all 32 MB:

    y = conv(x) + scaling * B(A(x))     # A: (r, in, k,k,k), B: (out, r, 1,1,1)

Every ``nn.Conv3d`` in the loaded baseline is wrapped once at construction
(base weights frozen, ``B`` zero-initialized so the net starts identical to
the checkpoint). ``reset_ttt_state`` restores the wrapped checkpoint state
(base + zeroed adapters) and rebuilds the optimizer over LoRA params only.

Pure ``torch`` (no ``peft`` — nothing is installed at evaluation time, so the
~30-line adapter lives in this file and ships inside the zip).

No intervals are returned (scorer default band), so v9 -> v11 isolates the
parameterization: full-model SGD vs LoRA-SGD, same data, same lr, same loss.

Knobs in ``policy.yaml``: ``base_model`` (ckpt hint, default ``cno``),
``ttt_lr`` (default 1e-3), ``lora_rank`` (default 4), ``lora_alpha``
(default 8, effective scale ``alpha / rank``). Sweep rank without edits::

    python scripts/make_submission_zip.py submission_v11 \\
        --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \\
        --set lora_rank=8 --out dist/submission_v11_r8.zip
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Dict, Iterator, Optional, Tuple

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


class LoRAConv3d(nn.Module):
    """Frozen Conv3d + trainable low-rank side path (B zero-init => identity)."""

    def __init__(self, conv: nn.Conv3d, rank: int, alpha: float):
        super().__init__()
        self.conv = conv
        self.conv.weight.requires_grad_(False)
        if self.conv.bias is not None:
            self.conv.bias.requires_grad_(False)
        self.scaling = alpha / max(rank, 1)
        # Down-projection mirrors the base geometry; up-projection is 1x1x1.
        self.lora_A = nn.Conv3d(conv.in_channels, rank, kernel_size=conv.kernel_size,
                                stride=conv.stride, padding=conv.padding,
                                dilation=conv.dilation, groups=1, bias=False)
        self.lora_B = nn.Conv3d(rank, conv.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.scaling * self.lora_B(self.lora_A(x))


def _wrap_conv3d(module: nn.Module, rank: int, alpha: float) -> int:
    """Replace every nn.Conv3d with LoRAConv3d, in place. Returns wrap count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRAConv3d):
            continue  # already wrapped (never recurse into adapters)
        if isinstance(child, nn.Conv3d):
            setattr(module, name, LoRAConv3d(child, rank, alpha))
            count += 1
        else:
            count += _wrap_conv3d(child, rank, alpha)
    return count


def _lora_params(module: nn.Module) -> Iterator[torch.Tensor]:
    for m in module.modules():
        if isinstance(m, LoRAConv3d):
            yield m.lora_A.weight
            yield m.lora_B.weight


def _read_policy(submission_dir: str) -> Dict[str, Any]:
    """Flat policy reader: base_model (str), ttt_lr / rank / alpha."""
    policy: Dict[str, Any] = {
        "base_model": "cno", "ttt_lr": 1e-3, "lora_rank": 4, "lora_alpha": 8.0,
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
                elif k == "lora_rank":
                    policy[k] = max(int(v), 1)
                elif k == "lora_alpha":
                    policy[k] = float(v)
    if not policy["ttt_lr"] > 0.0:
        raise ValueError(f"ttt_lr must be positive, got {policy['ttt_lr']}")
    return policy


class LoRATTTModel(TTTModel):
    """One SGD step on LoRA params only, then predict (no intervals)."""

    def __init__(self, base: nn.Module, device: str, ttt_lr: float = 1e-3,
                 rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base.to(device)
        self.device = device
        self.ttt_lr = ttt_lr
        n = _wrap_conv3d(self.base, rank, alpha)
        self.base.to(device)  # move freshly created adapters
        lora = list(_lora_params(self.base))
        if not lora:  # fallback base has Conv2d, not Conv3d: adapt all (== v9)
            print("[v11] no Conv3d found — adapting all params (v9-equivalent)")
            lora = [p for p in self.base.parameters() if p.requires_grad]
        else:
            print(f"[v11] wrapped {n} Conv3d, {sum(p.numel() for p in lora)} LoRA params")
        self._lora = lora
        self._init_state = copy.deepcopy(self.base.state_dict())
        self._opt = torch.optim.SGD(lora, lr=ttt_lr)
        self._prev_input: Optional[torch.Tensor] = None

    def reset_ttt_state(self) -> None:
        self.base.load_state_dict(copy.deepcopy(self._init_state))
        self._opt = torch.optim.SGD(list(_lora_params(self.base))
                                    if any(isinstance(m, LoRAConv3d) for m in self.base.modules())
                                    else [p for p in self.base.parameters() if p.requires_grad],
                                    lr=self.ttt_lr)
        self._prev_input = None

    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        input_norm = torch.as_tensor(input_norm).to(self.device)

        # (1) ONE gradient step on the PREVIOUS pair -- LoRA params only.
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
        print(f"[v11] loaded baseline: {meta}")
        return base
    print("[v11] no model.pth — TinyForecaster fallback (smoke-test only)")
    return TinyForecaster()


def get_ttt_model(submission_dir: str, device: str):
    """Entry point called once by the evaluator (construction is not timed)."""
    policy = _read_policy(submission_dir)
    return LoRATTTModel(
        _build_base(submission_dir, device, policy), device,
        policy["ttt_lr"], policy["lora_rank"], policy["lora_alpha"],
    )
