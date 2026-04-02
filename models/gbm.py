"""
Geometric Brownian Motion — baseline control model.

The simplest stochastic model: log-normal random walk with constant
drift and volatility. If other models can't beat GBM on the scoring
system, they are not worth their complexity.

S(t+1) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
where Z ~ N(0,1)

Parameters are estimated via MLE from training data on every fit() call.
There is no hyperparameter optimization for GBM — mu and sigma are fully
determined by the data.
"""
import numpy as np
from typing import Any

from .base import BaseModel, SimulationResult
from .registry import register_model


@register_model
class GBMModel(BaseModel):
    """
    Geometric Brownian Motion price path generator.

    Mu and sigma are always estimated from training data via MLE.
    get_param_space() returns an empty dict — no optimization needed.
    """

    def __init__(self):
        super().__init__()
        self.mu: float | None = None
        self.sigma: float | None = None
        self._dt = 1 / 365  # Daily time step

    @property
    def name(self) -> str:
        return "gbm"

    def get_default_params(self) -> dict[str, Any]:
        return {"mu": None, "sigma": None}

    def get_param_space(self) -> dict[str, dict[str, Any]]:
        # GBM has no hyperparameters to optimize.
        # mu and sigma are estimated from training data via MLE.
        return {}

    def fit(self, historical_returns: np.ndarray) -> None:
        """
        Estimate mu and sigma from historical daily log returns via MLE.

        Every call re-estimates from the provided data — parameters are
        never carried over from a previous fit.

        Parameters
        ----------
        historical_returns : np.ndarray
            1D array of daily log returns
        """
        if historical_returns.ndim != 1:
            raise ValueError("historical_returns must be 1D")

        daily_mu = np.mean(historical_returns)
        daily_sigma = np.std(historical_returns, ddof=1)

        # Annualize: crypto trades 365 days/year
        self.mu = float(daily_mu * 365)
        self.sigma = float(daily_sigma * np.sqrt(365))

        self._fitted_params = {
            "mu_annual": self.mu,
            "sigma_annual": self.sigma,
            "mu_daily": daily_mu,
            "sigma_daily": daily_sigma,
        }
        self._is_fitted = True

    def simulate(
        self,
        n_simulations: int,
        n_steps: int,
        initial_price: float,
        seed: int | None = None,
    ) -> SimulationResult:
        """
        Generate GBM price paths.

        Parameters
        ----------
        n_simulations : int
            Number of paths to generate
        n_steps : int
            Number of daily time steps per path
        initial_price : float
            Starting price
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        SimulationResult
        """
        self._check_fitted()

        rng = np.random.default_rng(seed)

        # Convert annualized params to daily
        dt = self._dt
        mu_daily = self.mu * dt
        sigma_daily = self.sigma * np.sqrt(dt)

        # Generate random innovations
        Z = rng.standard_normal((n_simulations, n_steps))

        # Daily log returns: (mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z
        log_returns = (mu_daily - 0.5 * sigma_daily**2) + sigma_daily * Z

        # Build price paths from cumulative returns
        cum_returns = np.cumsum(log_returns, axis=1)
        paths = np.empty((n_simulations, n_steps + 1))
        paths[:, 0] = initial_price
        paths[:, 1:] = initial_price * np.exp(cum_returns)

        return SimulationResult(
            paths=paths,
            log_returns=log_returns,
            params_used={
                "mu": self.mu,
                "sigma": self.sigma,
            },
            model_name=self.name,
            metadata={
                "mu_annual": self.mu,
                "sigma_annual": self.sigma,
                "mu_daily": self._fitted_params["mu_daily"],
                "sigma_daily": self._fitted_params["sigma_daily"],
                "initial_price": initial_price,
                "n_steps": n_steps,
            },
        )
