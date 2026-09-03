from realpde.models.base import PretrainModel
from realpde.models.adapter import ModelAdapter
from realpde.models.unet import UNet, UNetConfig

_MODEL_REGISTRY: dict[str, type] = {
    "unet": UNet,
}


def get_model(name: str, **kwargs) -> PretrainModel:
    """Instantiate a model by name.

    Parameters
    ----------
    name : str
        Model name (e.g. ``"unet"``).
    **kwargs
        Forwarded to the model constructor as keyword arguments.
        A ``config`` keyword containing a dataclass is also supported.

    Returns
    -------
    PretrainModel
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}")
    cls = _MODEL_REGISTRY[name]
    return cls(**kwargs)


__all__ = ["get_model", "ModelAdapter", "PretrainModel", "UNet", "UNetConfig"]
