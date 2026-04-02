"""
Composite scoring engine — combines 9 scoring metrics into a single score.

Only scoring metrics feed into the composite. Diagnostic metrics are
computed and reported separately.
"""
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from models.base import SimulationResult
from .metrics import ScoringMetrics
from .diagnostics import DiagnosticMetrics
from .distributions import compute_distributions


DEFAULT_WEIGHTS = {
    # Path Dynamics — 35% total (highest priority)
    "drawdown_duration_dist": 0.13,
    "recovery_time_dist": 0.12,
    "time_in_drawdown_ratio": 0.10,
    # Tail / Extreme Risk — 25% total
    "tail_index_accuracy": 0.13,
    "percentile_band_coverage": 0.12,
    # Distributional — 15% total
    "ks_statistic": 0.07,
    "qq_divergence": 0.08,
    # Temporal Dependence — 10%
    "vol_clustering_acf": 0.10,
    # Forecast Accuracy — 15%
    "mape": 0.15,
}


@dataclass
class ScoreCard:
    """Complete scoring result for a model+horizon+params combination."""

    model_name: str
    horizon: str  # "30d", "90d", "180d", "365d", "1460d"
    params: dict
    scoring_metrics: dict[str, float]  # metric_name -> score [0, 1]
    diagnostic_metrics: dict[str, Any]  # metric_name -> value (not scored)
    composite_score: float  # Weighted average of scoring_metrics only
    raw_metrics: dict[str, dict] = field(default_factory=dict)  # metric_name -> raw intermediates
    distributions: dict = field(default_factory=dict)  # distributional data for visualization
    rank: int | None = None


class Scorer:
    """
    Composite scoring engine.

    Computes all 9 scoring metrics, applies weights, and produces a
    single composite score. Also computes diagnostic metrics for reporting.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        """
        Parameters
        ----------
        weights : dict, optional
            Metric name -> weight for scoring metrics.
            Must sum to ~1.0. Defaults emphasize path dynamics and tails.
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()

        # Validate weights
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(
                f"Scoring weights must sum to ~1.0, got {weight_sum:.4f}"
            )

    def score(
        self,
        sim_result: SimulationResult,
        hist_prices: np.ndarray,
        hist_returns: np.ndarray,
        horizon: str,
        train_prices: np.ndarray | None = None,
        include_distributions: bool = True,
    ) -> ScoreCard:
        """
        Score a simulation result against historical data.

        Parameters
        ----------
        sim_result : SimulationResult
            Output from a model's simulate() method.
        hist_prices : np.ndarray
            Historical price series (out-of-sample window), 1D.
        hist_returns : np.ndarray
            Historical log returns (out-of-sample window), 1D.
        horizon : str
            Horizon label (e.g. "30d").
        train_prices : np.ndarray, optional
            Training prices for MASE diagnostic.

        Returns
        -------
        ScoreCard
            Complete scoring result.
        """
        # Compute all 9 scoring metrics with raw intermediates
        detailed = ScoringMetrics.compute_all_detailed(
            sim_paths=sim_result.paths,
            sim_returns=sim_result.log_returns,
            hist_prices=hist_prices,
            hist_returns=hist_returns,
        )

        # Split into normalized scores and raw intermediates
        scoring_metrics = {name: d["score"] for name, d in detailed.items()}
        raw_metrics = {name: d["raw"] for name, d in detailed.items()}

        # Compute composite score (weighted average of scoring metrics)
        composite = 0.0
        for metric_name, weight in self.weights.items():
            score_val = scoring_metrics.get(metric_name, 0.0)
            # Clamp to [0, 1] for safety
            score_val = max(0.0, min(1.0, score_val))
            composite += weight * score_val

        # Compute diagnostic metrics (not scored)
        diagnostic_metrics = DiagnosticMetrics.compute_all(
            sim_paths=sim_result.paths,
            sim_returns=sim_result.log_returns,
            hist_prices=hist_prices,
            hist_returns=hist_returns,
            train_prices=train_prices,
        )

        # Compute distributional data for visualization
        distributions = {}
        if include_distributions:
            distributions = compute_distributions(
                sim_paths=sim_result.paths,
                sim_returns=sim_result.log_returns,
                hist_prices=hist_prices,
                hist_returns=hist_returns,
            )

        return ScoreCard(
            model_name=sim_result.model_name,
            horizon=horizon,
            params=sim_result.params_used,
            scoring_metrics=scoring_metrics,
            diagnostic_metrics=diagnostic_metrics,
            composite_score=float(composite),
            raw_metrics=raw_metrics,
            distributions=distributions,
        )

    def score_quick(
        self,
        sim_result: SimulationResult,
        hist_prices: np.ndarray,
        hist_returns: np.ndarray,
    ) -> float:
        """
        Quick scoring — composite score only, no diagnostics.

        Used during optimization where we only need the number to maximize.

        Parameters
        ----------
        sim_result : SimulationResult
            Output from a model's simulate() method.
        hist_prices : np.ndarray
            Historical price series, 1D.
        hist_returns : np.ndarray
            Historical log returns, 1D.

        Returns
        -------
        float
            Composite score in [0, 1].
        """
        scoring_metrics = ScoringMetrics.compute_all(
            sim_paths=sim_result.paths,
            sim_returns=sim_result.log_returns,
            hist_prices=hist_prices,
            hist_returns=hist_returns,
        )

        composite = 0.0
        for metric_name, weight in self.weights.items():
            score_val = max(0.0, min(1.0, scoring_metrics.get(metric_name, 0.0)))
            composite += weight * score_val

        return float(composite)
