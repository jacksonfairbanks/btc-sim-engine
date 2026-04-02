"""Simulation models for BTC price paths."""
from .base import BaseModel, SimulationResult
from .registry import get_model, list_models, register_model

# Import models to trigger registration (order doesn't matter)
from . import gbm  # noqa: F401
from . import garch  # noqa: F401
from . import regime_block_bootstrap  # noqa: F401

__all__ = [
    "BaseModel",
    "SimulationResult",
    "get_model",
    "list_models",
    "register_model",
]
