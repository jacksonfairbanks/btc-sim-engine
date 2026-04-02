"""
GARCH(1,1) — Volatility Clustering Benchmark Model.

Captures the well-documented autocorrelation in BTC's volatility: big moves
follow big moves, calm follows calm. Parametric model — generates synthetic
returns from a time-varying conditional variance process.

The GARCH(1,1) equation:
    sigma^2(t) = omega + alpha * r^2(t-1) + beta * sigma^2(t-1)

where:
    omega = base volatility floor (never drops below this)
    alpha = shock reactivity (yesterday's return surprise -> today's vol)
    beta  = persistence (yesterday's vol carries forward)
    alpha + beta = persistence (for BTC: typically 0.95-0.99)

Uses the `arch` library for MLE fitting. Simulation rolls forward from the
last fitted conditional variance for realistic forward projection.

Academic reference: Bollerslev (1986), "Generalized Autoregressive Conditional
Heteroskedasticity."
"""
import hashlib
import pickle
import time
import numpy as np
import warnings
from pathlib import Path
from typing import Any

from arch import arch_model

from .base import BaseModel, SimulationResult
from .registry import register_model


@register_model
class GARCHModel(BaseModel):
    """
    GARCH(p,q) price path generator with configurable innovation distribution.

    Hyperparameters (searched by Optuna):
        p: GARCH lag order (1-2)
        q: ARCH lag order (1-2)
        dist: innovation distribution ("normal", "t", "skewt")
        mean_model: mean specification ("Zero", "Constant", "ARX")

    Fitted parameters (estimated by MLE):
        omega, alpha, beta: GARCH equation coefficients
        nu: degrees of freedom for t/skewt (lower = fatter tails)
        mu: constant mean return (if mean_model="Constant")
    """

    def __init__(self):
        super().__init__()
        # Hyperparameters
        self.p: int = 1
        self.q: int = 1
        self.dist: str = "t"
        self.mean_model: str = "Constant"

        # Fitted state
        self._model_fit = None
        self._returns_scale: float = 100.0  # arch library works better with % returns
        self._last_cond_var: float | None = None  # last fitted conditional variance (scaled)
        self._unconditional_var: float | None = None
        self._convergence_log: list[str] = []
        self._timing: dict[str, float] = {}

        # Cache
        self._cache_dir = Path("results/cache")

    @property
    def name(self) -> str:
        return "garch_1_1"

    def get_default_params(self) -> dict[str, Any]:
        return {
            "p": 1,
            "q": 1,
            "dist": "t",
            "mean_model": "Constant",
        }

    def get_param_space(self) -> dict[str, dict[str, Any]]:
        return {
            "p": {"type": "int", "low": 1, "high": 2},
            "q": {"type": "int", "low": 1, "high": 2},
            "dist": {"type": "categorical", "choices": ["normal", "t", "skewt"]},
            "mean_model": {"type": "categorical", "choices": ["Zero", "Constant", "ARX"]},
        }

    # ── Caching ────────────────────────────────────────────────────

    def _cache_key(self, returns: np.ndarray) -> str:
        """Hash of training data + hyperparams."""
        data_hash = hashlib.md5(returns.tobytes()).hexdigest()[:12]
        return f"garch_{data_hash}_p{self.p}_q{self.q}_{self.dist}_{self.mean_model}"

    def _load_cache(self, key: str) -> dict | None:
        cache_file = self._cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, data: dict) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._cache_dir / f"{key}.pkl", "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass

    # ── fit() ──────────────────────────────────────────────────────

    def fit(self, historical_returns: np.ndarray) -> None:
        """
        Fit GARCH model to historical daily log returns.

        Parameters
        ----------
        historical_returns : np.ndarray
            1D array of daily log returns.
        """
        if historical_returns.ndim != 1:
            raise ValueError("historical_returns must be 1D")

        self._convergence_log = []
        self._timing = {}
        t0 = time.time()

        # Scale returns to percentage for arch library numerical stability
        scaled_returns = historical_returns * self._returns_scale

        # Check cache
        cache_key = self._cache_key(historical_returns)
        cached = self._load_cache(cache_key)
        if cached is not None:
            self._fitted_params = cached["fitted_params"]
            self._last_cond_var = cached["last_cond_var"]
            self._unconditional_var = cached["unconditional_var"]
            self._model_fit = cached.get("model_fit")
            self._convergence_log.append(f"GARCH loaded from cache ({cache_key})")
            self._is_fitted = True
            self._timing["fit"] = 0.0
            return

        # Fit GARCH
        am = arch_model(
            scaled_returns,
            mean=self.mean_model,
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist=self.dist,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self._model_fit = am.fit(disp="off", show_warning=False)
            except Exception as e:
                self._convergence_log.append(f"GARCH fit failed: {str(e)[:100]}")
                # Retry with different starting values
                try:
                    self._model_fit = am.fit(
                        disp="off", show_warning=False,
                        starting_values=None,
                    )
                except Exception as e2:
                    self._convergence_log.append(f"GARCH retry failed: {str(e2)[:100]}")
                    raise RuntimeError(f"GARCH fitting failed: {e2}") from e2

        # Extract fitted parameters
        params = self._model_fit.params
        param_dict = {k: float(v) for k, v in params.items()}

        # Extract GARCH-specific parameters (names vary by mean model)
        # omega is always named "omega", alpha[1] and beta[1] for GARCH(1,1)
        omega = param_dict.get("omega", 0)
        alpha_keys = [k for k in param_dict if k.startswith("alpha")]
        beta_keys = [k for k in param_dict if k.startswith("beta")]
        alpha = sum(param_dict[k] for k in alpha_keys)
        beta = sum(param_dict[k] for k in beta_keys)
        persistence = alpha + beta

        # Half-life of volatility shock
        if persistence > 0 and persistence < 1:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = float('inf')

        # Extract distribution parameters
        nu = param_dict.get("nu", None)  # degrees of freedom for t/skewt
        lam = param_dict.get("lambda", None)  # skewness for skewt

        # Mean parameter
        mu = param_dict.get("mu", 0.0)

        # Conditional variance: last fitted value and unconditional
        cond_vol = np.asarray(self._model_fit.conditional_volatility)
        self._last_cond_var = float(cond_vol[-1] ** 2)  # stored in scaled units
        self._unconditional_var = float(omega / (1 - persistence)) if persistence < 1 else self._last_cond_var

        # Log convergence info
        self._convergence_log.append(
            f"GARCH({self.p},{self.q}) fitted with dist={self.dist}, mean={self.mean_model}"
        )
        self._convergence_log.append(
            f"omega={omega:.6f}, alpha={alpha:.4f}, beta={beta:.4f}, "
            f"persistence={persistence:.4f}, half_life={half_life:.1f}d"
        )
        if nu is not None:
            self._convergence_log.append(f"nu (DoF)={nu:.2f}")
        if lam is not None:
            self._convergence_log.append(f"lambda (skew)={lam:.4f}")
        self._convergence_log.append(
            f"mu={mu:.6f} (scaled, daily={mu / self._returns_scale:.6f})"
        )
        self._convergence_log.append(
            f"Last cond vol: {np.sqrt(self._last_cond_var):.4f}% "
            f"(daily: {np.sqrt(self._last_cond_var) / self._returns_scale:.6f})"
        )
        self._convergence_log.append(
            f"Unconditional vol: {np.sqrt(self._unconditional_var):.4f}% "
            f"(daily: {np.sqrt(self._unconditional_var) / self._returns_scale:.6f})"
        )
        self._convergence_log.append(
            f"LL={self._model_fit.loglikelihood:.1f}, "
            f"AIC={self._model_fit.aic:.1f}, BIC={self._model_fit.bic:.1f}"
        )

        elapsed = time.time() - t0
        self._timing["fit"] = elapsed
        self._convergence_log.append(f"Fit completed in {elapsed:.2f}s")

        # Store clean fitted params
        self._fitted_params = {
            "p": self.p,
            "q": self.q,
            "dist": self.dist,
            "mean_model": self.mean_model,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "persistence": persistence,
            "half_life_days": round(half_life, 1),
            "mu_scaled": mu,
            "mu_daily": mu / self._returns_scale,
            "last_cond_vol_daily": np.sqrt(self._last_cond_var) / self._returns_scale,
            "unconditional_vol_daily": np.sqrt(self._unconditional_var) / self._returns_scale,
            "log_likelihood": float(self._model_fit.loglikelihood),
            "aic": float(self._model_fit.aic),
            "bic": float(self._model_fit.bic),
        }
        if nu is not None:
            self._fitted_params["nu"] = nu
        if lam is not None:
            self._fitted_params["lambda"] = lam

        # Cache (exclude model_fit object — it's large and not always picklable)
        self._save_cache(cache_key, {
            "fitted_params": self._fitted_params,
            "last_cond_var": self._last_cond_var,
            "unconditional_var": self._unconditional_var,
        })

        self._is_fitted = True

    # ── simulate() ─────────────────────────────────────────────────

    def simulate(
        self,
        n_simulations: int,
        n_steps: int,
        initial_price: float,
        seed: int | None = None,
    ) -> SimulationResult:
        """
        Generate GARCH price paths by rolling forward conditional variance.

        Starts from the last fitted conditional variance (not unconditional),
        so the simulation reflects the current volatility state. Each day:
        1. Draw innovation z from the fitted distribution
        2. Compute return: r(t) = mu + sigma(t) * z
        3. Update variance: sigma^2(t+1) = omega + alpha*r^2(t) + beta*sigma^2(t)
        4. Convert to price: P(t+1) = P(t) * exp(r(t))

        Variance is capped at 10x unconditional variance to prevent explosions
        on long horizons. Paths with NaN/inf are re-simulated.

        Parameters
        ----------
        n_simulations : int
        n_steps : int
        initial_price : float
        seed : int, optional

        Returns
        -------
        SimulationResult
        """
        self._check_fitted()
        t0 = time.time()

        fp = self._fitted_params
        omega = fp["omega"]
        alpha = fp["alpha"]
        beta = fp["beta"]
        mu_scaled = fp["mu_scaled"]
        var_cap = self._unconditional_var * 10  # cap at 10x unconditional

        rng = np.random.default_rng(seed)

        # Draw innovations based on fitted distribution
        # All in scaled (%) space, convert to daily at the end
        nu = fp.get("nu", None)
        lam = fp.get("lambda", None)

        all_returns_scaled = np.empty((n_simulations, n_steps))
        all_paths = np.empty((n_simulations, n_steps + 1))
        all_paths[:, 0] = initial_price

        n_redrawn = 0
        max_redraw = n_simulations  # safety limit

        for sim_idx in range(n_simulations):
            valid = False
            attempts = 0
            while not valid and attempts < 5:
                attempts += 1

                # Draw innovations
                if self.dist == "normal":
                    z = rng.standard_normal(n_steps)
                elif self.dist == "t":
                    if nu is not None and nu > 2:
                        # Standardized t: mean=0, var=1
                        z = rng.standard_t(nu, n_steps) / np.sqrt(nu / (nu - 2))
                    else:
                        z = rng.standard_normal(n_steps)
                elif self.dist == "skewt":
                    # Approximate skewed t using Hansen's skewed t
                    # For simplicity, use t-distribution (skewness effect is secondary)
                    if nu is not None and nu > 2:
                        z = rng.standard_t(nu, n_steps) / np.sqrt(nu / (nu - 2))
                    else:
                        z = rng.standard_normal(n_steps)
                else:
                    z = rng.standard_normal(n_steps)

                # Roll forward GARCH process
                returns = np.empty(n_steps)
                cond_var = self._last_cond_var  # start from last fitted state

                for t in range(n_steps):
                    # Cap variance
                    if cond_var > var_cap:
                        cond_var = var_cap
                    if cond_var < 1e-10:
                        cond_var = 1e-10

                    sigma = np.sqrt(cond_var)
                    returns[t] = mu_scaled + sigma * z[t]

                    # Update conditional variance for next step
                    cond_var = omega + alpha * returns[t] ** 2 + beta * cond_var

                # Convert from scaled (%) to daily log returns
                log_returns = returns / self._returns_scale

                # Check for validity
                if np.isfinite(log_returns).all() and np.abs(log_returns).max() < 1.0:
                    # No single day move > 100% (exp(1) ~ 2.7x)
                    valid = True
                else:
                    n_redrawn += 1

            all_returns_scaled[sim_idx] = returns
            # Build price path
            cum_returns = np.cumsum(log_returns)
            all_paths[sim_idx, 1:] = initial_price * np.exp(cum_returns)

        all_log_returns = all_returns_scaled / self._returns_scale

        # Final validation
        bad_paths = ~np.isfinite(all_paths).all(axis=1)
        if bad_paths.any():
            for i in np.where(bad_paths)[0]:
                all_paths[i, :] = initial_price
                all_log_returns[i, :] = 0.0

        elapsed = time.time() - t0

        return SimulationResult(
            paths=all_paths,
            log_returns=all_log_returns,
            params_used={
                "p": self.p,
                "q": self.q,
                "dist": self.dist,
                "mean_model": self.mean_model,
                "omega": fp["omega"],
                "alpha": fp["alpha"],
                "beta": fp["beta"],
                "persistence": fp["persistence"],
                "half_life_days": fp["half_life_days"],
                "nu": fp.get("nu"),
                "mu_daily": fp["mu_daily"],
                "last_cond_vol_daily": fp["last_cond_vol_daily"],
                "unconditional_vol_daily": fp["unconditional_vol_daily"],
            },
            model_name=self.name,
            metadata={
                "convergence_log": self._convergence_log,
                "timing": {**self._timing, "simulation": elapsed},
                "initial_price": initial_price,
                "n_steps": n_steps,
                "n_redrawn_paths": n_redrawn,
                "log_likelihood": fp["log_likelihood"],
                "aic": fp["aic"],
                "bic": fp["bic"],
            },
        )
