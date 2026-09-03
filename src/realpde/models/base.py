from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn

IN_STEP = 20
HEIGHT = 32
WIDTH = 64
CHANNELS = 3
EXPECTED_SHAPE = (1, IN_STEP, HEIGHT, WIDTH, CHANNELS)


class PretrainModel(nn.Module, ABC):
    """Abstract base class for pretrained PDE models.

    All models accept and return tensors of shape ``(B, T, H, W, C)`` where
    ``C == 3`` (u, v, p channels).  The concrete ``T``, ``H``, ``W`` are
    determined by the training configuration (``in_step`` / ``out_step``) but
    must remain compatible with the competition evaluation shapes at inference
    time (``T=20, H=32, W=64``).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T_in, H, W, C)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, T_out, H, W, C)``.
        """

    def load_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> dict:
        """Load a training checkpoint and apply weights.

        Parameters
        ----------
        checkpoint_path : str
            Path to a ``.pth`` file containing ``{'model_state_dict': ...}``.
        device : str
            Target device.

        Returns
        -------
        dict
            Metadata from the checkpoint (``iteration``, ``best_val_loss``, …).
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.load_state_dict(state, strict=True)
        meta = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        return meta
