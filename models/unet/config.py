from dataclasses import dataclass


@dataclass
class UNetConfig:
    """Configuration for the per-frame 2D UNet."""

    in_step: int = 20
    out_step: int = 20
    in_channels: int = 3
    out_channels: int = 3
    channels: int = 16
    n_layers: int = 2

    @property
    def name(self) -> str:
        return "unet"
