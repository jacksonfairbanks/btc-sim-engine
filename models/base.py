"""
Abstract base class for all price path simulation models.

THE contract that every model must implement. No exceptions.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class SimulationResult:
    """
    Standard output from any model.

    This is the universal interface between models and validation.
    The validation layer never knows which model generated the paths—
    it only receives SimulationResult objects.
    """
    paths: np.ndarray  # Shape: (n_simulations, n_steps) — price levels
    log_returns: np.ndarray  # Shape: (n_simulations, n_steps-1) — log returns
    params_used: dict  # Exact parameters used for reproducibility
    model_name: str  # e.g. "gbm", "garch_1_1", "regime_block_bootstrap"
    metadata: dict = field(default_factory=dict)  # Model-specific diagnostics

    def __post_init__(self):
        """Validate shapes and data integrity."""
        if self.paths.ndim != 2:
            raise ValueError(f"paths must be 2D array, got shape {self.paths.shape}")

        if self.log_returns.ndim != 2:
            raise ValueError(f"log_returns must be 2D array, got shape {self.log_returns.shape}")

        n_sims, n_steps = self.paths.shape
        expected_returns_shape = (n_sims, n_steps - 1)

        if self.log_returns.shape != expected_returns_shape:
            raise ValueError(
                f"log_returns shape {self.log_returns.shape} inconsistent with "
                f"paths shape {self.paths.shape}. Expected {expected_returns_shape}"
            )

        # Check for NaN or inf
        if np.isnan(self.paths).any():
            raise ValueError("paths contains NaN values")
        if np.isinf(self.paths).any():
            raise ValueError("paths contains inf values")
        if np.isnan(self.log_returns).any():
            raise ValueError("log_returns contains NaN values")
        if np.isinf(self.log_returns).any():
            raise ValueError("log_returns contains inf values")


class BaseModel(ABC):
    """
    Contract for all price path generators.

    Every model must implement this interface exactly.
    Models are black boxes to the validation layer.
    """

    def __init__(self):
        """Initialize model. Subclasses may add model-specific parameters."""
        self._is_fitted = False
        self._fitted_params = {}

    @abstractmethod
    def fit(self, historical_returns: np.ndarray) -> None:
        """
        Fit model to historical log returns.

        Parameters
        ----------
        historical_returns : np.ndarray
            1D array of historical log returns

        Notes
        -----
        After calling fit(), the model must be ready to simulate.
        Store fitted parameters in self._fitted_params for reproducibility.
        Set self._is_fitted = True when fitting is complete.
        """
        pass

    @abstractmethod
    def simulate(
        self,
        n_simulations: int,
        n_steps: int,
        initial_price: float,
        seed: int | None = None
    ) -> SimulationResult:
        """
        Generate simulated price paths.

        Parameters
        ----------
        n_simulations : int
            Number of price paths to simulate
        n_steps : int
            Number of time steps in each path (days)
        initial_price : float
            Starting price for all paths
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        SimulationResult
            Standardized output containing paths, returns, and metadata

        Raises
        ------
        RuntimeError
            If model has not been fitted
        """
        pass

    @abstractmethod
    def get_default_params(self) -> dict[str, Any]:
        """
        Return default parameter dict for this model.

        Returns
        -------
        dict
            Parameter name -> default value

        Examples
        --------
        {"mu": None, "sigma": None}  # None means "estimate from data"
        {"p": 1, "q": 1, "dist": "normal"}
        """
        pass

    @abstractmethod
    def get_param_space(self) -> dict[str, dict[str, Any]]:
        """
        Return parameter search space for optimization.

        Returns
        -------
        dict
            Parameter name -> parameter spec dict

        Parameter spec format:
        - For continuous: {"type": "float", "low": 0.01, "high": 0.99}
        - For integer: {"type": "int", "low": 1, "high": 10}
        - For categorical: {"type": "categorical", "choices": ["a", "b", "c"]}

        Examples
        --------
        {
            "p": {"type": "int", "low": 1, "high": 3},
            "q": {"type": "int", "low": 1, "high": 3},
            "dist": {"type": "categorical", "choices": ["normal", "t", "skewt"]}
        }
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique model identifier string.

        Returns
        -------
        str
            Model name (e.g., "gbm", "garch_1_1", "regime_block_bootstrap")
        """
        pass

    def set_params(self, **params) -> None:
        """
        Set model parameters (used by optimization).

        Parameters
        ----------
        **params
            Parameter name-value pairs
        """
        for key, value in params.items():
            setattr(self, key, value)

    def get_params(self) -> dict[str, Any]:
        """
        Get current model parameters.

        Returns
        -------
        dict
            Current parameter values
        """
        return {
            key: getattr(self, key)
            for key in self.get_default_params().keys()
            if hasattr(self, key)
        }

    def _check_fitted(self) -> None:
        """Raise error if model hasn't been fitted yet."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.name} must be fitted before calling simulate(). "
                "Call fit() with historical returns first."
            )
