"""
Walk-forward out-of-sample backtesting framework.

Uses expanding-window validation with recency weighting:
1. Train on data up to time T
2. Simulate paths from T to T+horizon
3. Score against actual realized path
4. Slide window forward by step_size
5. Compute recency-weighted average across all windows
"""
import numpy as np
from typing import Any
from rich.console import Console

from models.base import BaseModel
from validation.scorer import Scorer, ScoreCard


console = Console()


class WalkForwardBacktest:
    """
    Expanding-window walk-forward validation.

    Parameters
    ----------
    prices : np.ndarray
        Full price series (1D).
    returns : np.ndarray
        Full log return series (1D).
    min_training_days : int
        Minimum days in training window before first evaluation.
    step_size_days : int
        Number of days to slide the window forward each step.
    recency_weighting : str
        "exponential", "linear", or "equal".
    recency_half_life_days : int
        For exponential weighting: weight halves every N days.
    dates : np.ndarray, optional
        Date strings aligned with prices, for labeling windows.
    """

    def __init__(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        min_training_days: int = 730,
        step_size_days: int = 90,
        recency_weighting: str = "exponential",
        recency_half_life_days: int = 730,
        dates: np.ndarray | None = None,
    ):
        self.prices = prices
        self.returns = returns
        self.min_training_days = min_training_days
        self.step_size_days = step_size_days
        self.recency_weighting = recency_weighting
        self.recency_half_life_days = recency_half_life_days
        self.dates = dates

    def _idx_to_date(self, idx: int) -> str | None:
        """Convert index to date string if dates available."""
        if self.dates is not None and 0 <= idx < len(self.dates):
            return str(self.dates[idx])
        return None

    def _compute_weights(self, n_windows: int, window_positions: list[int]) -> np.ndarray:
        """
        Compute recency weights for each walk-forward window.

        Parameters
        ----------
        n_windows : int
            Number of evaluation windows.
        window_positions : list[int]
            Day index where each window starts.

        Returns
        -------
        np.ndarray
            Normalized weights summing to 1.
        """
        if self.recency_weighting == "equal" or n_windows <= 1:
            return np.ones(n_windows) / n_windows

        positions = np.array(window_positions, dtype=float)
        latest = positions[-1]
        time_diffs = latest - positions  # days from most recent window

        if self.recency_weighting == "exponential":
            lam = np.log(2) / self.recency_half_life_days
            weights = np.exp(-lam * time_diffs)
        elif self.recency_weighting == "linear":
            max_diff = time_diffs[-1] if time_diffs[-1] > 0 else 1.0
            weights = 1.0 - 0.5 * (time_diffs / max_diff)
        else:
            weights = np.ones(n_windows)

        return weights / weights.sum()

    def run(
        self,
        model: BaseModel,
        horizon_days: int,
        n_simulations: int = 1000,
        scorer: Scorer | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Execute walk-forward backtest for a single model+horizon.

        Parameters
        ----------
        model : BaseModel
            Model to evaluate (will be re-fitted each window).
        horizon_days : int
            Simulation horizon in days.
        n_simulations : int
            Number of simulated paths per window.
        scorer : Scorer, optional
            Scoring engine. Defaults to standard weights.
        seed : int
            Base random seed.

        Returns
        -------
        dict
            weighted_composite: float
            window_scores: list[ScoreCard] (with _train/_test metadata attached)
            weights: np.ndarray
            window_positions: list[int]
            n_windows: int
        """
        if scorer is None:
            scorer = Scorer()

        total_days = len(self.prices)
        window_cards: list[ScoreCard] = []
        window_positions: list[int] = []

        # Generate evaluation windows
        t = self.min_training_days
        window_num = 0

        while t + horizon_days < total_days:
            # Training data: everything up to t
            train_returns = self.returns[:t]
            train_prices = self.prices[:t + 1]  # +1 for initial price

            # Out-of-sample: t to t+horizon
            oos_end = min(t + horizon_days + 1, total_days)
            oos_prices = self.prices[t:oos_end]
            oos_returns = self.returns[t:min(t + horizon_days, total_days - 1)]

            if len(oos_returns) < 10:
                break

            # Fit and simulate
            initial_price = train_prices[-1]
            model.fit(train_returns)
            sim_result = model.simulate(
                n_simulations=n_simulations,
                n_steps=len(oos_returns),
                initial_price=initial_price,
                seed=seed + window_num,
            )

            horizon_label = f"{horizon_days}d"
            card = scorer.score(
                sim_result=sim_result,
                hist_prices=oos_prices,
                hist_returns=oos_returns,
                horizon=horizon_label,
                train_prices=train_prices,
            )

            # Attach window metadata to the ScoreCard for granular reporting
            card._train_start_idx = 0
            card._train_end_idx = t
            card._test_start_idx = t
            card._test_end_idx = oos_end - 1
            card._train_start_date = self._idx_to_date(0)
            card._train_end_date = self._idx_to_date(t)
            card._test_start_date = self._idx_to_date(t)
            card._test_end_date = self._idx_to_date(oos_end - 1)
            card._train_days = t
            card._test_days = len(oos_returns)

            # Store the params the model actually used for this window
            card._fitted_params = sim_result.params_used
            card._model_metadata = sim_result.metadata

            # Store percentile bands + realized prices for visualization
            card._sim_percentiles = {
                "p5": np.percentile(sim_result.paths, 5, axis=0).tolist(),
                "p25": np.percentile(sim_result.paths, 25, axis=0).tolist(),
                "p50": np.median(sim_result.paths, axis=0).tolist(),
                "p75": np.percentile(sim_result.paths, 75, axis=0).tolist(),
                "p95": np.percentile(sim_result.paths, 95, axis=0).tolist(),
            }
            card._realized_prices = oos_prices.tolist()

            window_cards.append(card)
            window_positions.append(t)

            t += self.step_size_days
            window_num += 1

        if not window_cards:
            return {
                "weighted_composite": 0.0,
                "window_scores": [],
                "weights": np.array([]),
                "window_positions": [],
                "n_windows": 0,
            }

        # Compute recency-weighted composite
        weights = self._compute_weights(len(window_cards), window_positions)
        composites = np.array([c.composite_score for c in window_cards])
        weighted_composite = float(np.dot(weights, composites))

        return {
            "weighted_composite": weighted_composite,
            "window_scores": window_cards,
            "weights": weights,
            "window_positions": window_positions,
            "n_windows": len(window_cards),
        }
