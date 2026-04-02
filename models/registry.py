"""
Model registry — discover and instantiate models by name.
"""
from typing import Type

from .base import BaseModel

# Registry mapping name -> class
_REGISTRY: dict[str, Type[BaseModel]] = {}


def register_model(cls: Type[BaseModel]) -> Type[BaseModel]:
    """
    Decorator to register a model class.

    Usage
    -----
    @register_model
    class MyModel(BaseModel):
        ...
    """
    # Instantiate temporarily to get the name
    # Models must support no-arg construction for registration
    instance = cls()
    _REGISTRY[instance.name] = cls
    return cls


def get_model(name: str, **kwargs) -> BaseModel:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name : str
        Model identifier (e.g. "gbm", "garch_1_1", "regime_block_bootstrap")
    **kwargs
        Parameters passed to the model constructor

    Returns
    -------
    BaseModel
        Instantiated model

    Raises
    ------
    KeyError
        If model name is not registered
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(_REGISTRY.keys())


