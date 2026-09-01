from __future__ import annotations

import copy
import os
from typing import Optional

import torch
import torch.nn as nn


def _load_state(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> dict:
    """Load a checkpoint into *model*, handling all known formats.

    Returns metadata dict from the checkpoint.
    """
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # fp16-packed format (from pack_ckpt_fp16.py)
    if isinstance(obj, dict) and "state_fp16" in obj:
        complex_keys = set(obj.get("complex_keys", []))
        sd = {}
        for k, v in obj["state_fp16"].items():
            if k in complex_keys:
                sd[k] = torch.view_as_complex(v.float())
            elif torch.is_tensor(v) and v.dtype == torch.float16:
                sd[k] = v.float()
            else:
                sd[k] = v
        model.load_state_dict(sd, strict=True)
        return {"format": "fp16_packed"}

    # Training checkpoint with metadata
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"], strict=True)
        meta = {}
        for k in ("iteration", "best_iteration", "best_val_loss"):
            if k in obj:
                v = obj[k]
                meta[k] = v.item() if torch.is_tensor(v) else v
        meta["format"] = "train_ckpt"
        return meta

    # Raw state dict
    model.load_state_dict(obj, strict=True)
    return {"format": "raw_state_dict"}


class ModelAdapter(nn.Module):
    """Uniform wrapper around any model (baseline or pretrained).

    Wraps a model so that ``forward(x)`` always accepts and returns tensors of
    shape ``(B, T, H, W, C)`` regardless of the underlying model's conventions.

    Usage::

        # From a baseline checkpoint (FNO / CNO / Transolver)
        adapter = ModelAdapter.from_baseline("sim_real_cno.pth", device="cuda")

        # From a pretrained model
        from models.unet import UNet, UNetConfig
        adapter = ModelAdapter.from_pretrained(UNet(UNetConfig()), "checkpoint.pth")
    """

    def __init__(self, model: nn.Module, initial_state: Optional[dict] = None):
        super().__init__()
        self.model = model
        self._initial_state = copy.deepcopy(initial_state or model.state_dict())

    @classmethod
    def from_baseline(
        cls,
        ckpt_path: str,
        device: str = "cpu",
    ) -> ModelAdapter:
        """Load a baseline checkpoint (FNO / CNO / Transolver).

        Uses the existing ``load_baseline.build_model`` + checkpoint loading
        from ``load_baseline.load_checkpoint_state``.
        """
        import sys
        _HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)

        from load_baseline import detect_model_type, build_model, load_checkpoint_state

        model_type = detect_model_type(ckpt_path)
        model = build_model(model_type, device=device)
        state, meta = load_checkpoint_state(ckpt_path)
        model.load_state_dict(state, strict=True)
        model.eval()
        return cls(model, model.state_dict())

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ) -> ModelAdapter:
        """Wrap a pretrained model, optionally loading a checkpoint."""
        model = model.to(device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            _load_state(model, checkpoint_path, device)
        model.eval()
        return cls(model, model.state_dict())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Model returned {type(out)}, expected torch.Tensor")
        return out

    def reset_weights(self) -> None:
        """Restore the model to its initial (post-checkpoint) weights."""
        self.model.load_state_dict(copy.deepcopy(self._initial_state))

    @property
    def initial_state(self) -> dict:
        return copy.deepcopy(self._initial_state)
