"""Tests for validation metrics and scorer."""
import numpy as np
import pytest

from validation.metrics import ScoringMetrics
from validation.diagnostics import DiagnosticMetrics
from validation.scorer import Scorer, ScoreCard, DEFAULT_WEIGHTS
from models.base import SimulationResult


@pytest.fixture
def synthetic_data():
    """Create paired simulated/historical data for testing."""
    rng = np.random.default_rng(42)
    n_days = 300

    # Historical
    hist_returns = rng.normal(0.0005, 0.04, n_days)
    hist_prices = 30000 * np.exp(np.cumsum(np.concatenate([[0], hist_returns])))

    # Simulated (similar distribution)
    n_sims = 100
    sim_returns = rng.normal(0.0005, 0.04, (n_sims, n_days))
    sim_paths = np.empty((n_sims, n_days + 1))
    sim_paths[:, 0] = hist_prices[0]
    sim_paths[:, 1:] = hist_prices[0] * np.exp(np.cumsum(sim_returns, axis=1))

    return {
        "hist_returns": hist_returns,
        "hist_prices": hist_prices,
        "sim_returns": sim_returns,
        "sim_paths": sim_paths,
    }


class TestScoringMetrics:
    def test_ks_statistic_identical(self):
        """Identical distributions should score ~1.0."""
        data = np.random.randn(10000)
        score = ScoringMetrics.ks_statistic(data, data)
        assert score > 0.99

    def test_ks_statistic_different(self):
        """Very different distributions should score low."""
        d1 = np.random.randn(5000)
        d2 = np.random.randn(5000) + 10  # Shifted mean
        score = ScoringMetrics.ks_statistic(d1, d2)
        assert score < 0.1

    def test_ks_statistic_range(self, synthetic_data):
        score = ScoringMetrics.ks_statistic(
            synthetic_data["sim_returns"], synthetic_data["hist_returns"]
        )
        assert 0.0 <= score <= 1.0

    def test_qq_divergence_identical(self):
        data = np.random.randn(5000)
        score = ScoringMetrics.qq_divergence(data, data)
        assert score > 0.99

    def test_qq_divergence_range(self, synthetic_data):
        score = ScoringMetrics.qq_divergence(
            synthetic_data["sim_returns"], synthetic_data["hist_returns"]
        )
        assert 0.0 <= score <= 1.0

    def test_tail_index_accuracy_range(self, synthetic_data):
        score = ScoringMetrics.tail_index_accuracy(
            synthetic_data["sim_returns"], synthetic_data["hist_returns"]
        )
        assert 0.0 <= score <= 1.0

    def test_var_backtest_kupiec_range(self, synthetic_data):
        score = ScoringMetrics.var_backtest_kupiec(
            synthetic_data["sim_returns"], synthetic_data["hist_returns"]
        )
        assert 0.0 <= score <= 1.0

    def test_drawdown_duration_dist_range(self, synthetic_data):
        score = ScoringMetrics.drawdown_duration_dist(
            synthetic_data["sim_paths"], synthetic_data["hist_prices"]
        )
        assert 0.0 <= score <= 1.0

    def test_recovery_time_dist_range(self, synthetic_data):
        score = ScoringMetrics.recovery_time_dist(
            synthetic_data["sim_paths"], synthetic_data["hist_prices"]
        )
        assert 0.0 <= score <= 1.0

    def test_time_in_drawdown_ratio_range(self, synthetic_data):
        score = ScoringMetrics.time_in_drawdown_ratio(
            synthetic_data["sim_paths"], synthetic_data["hist_prices"]
        )
        assert 0.0 <= score <= 1.0

    def test_vol_clustering_acf_range(self, synthetic_data):
        score = ScoringMetrics.vol_clustering_acf(
            synthetic_data["sim_returns"], synthetic_data["hist_returns"]
        )
        assert 0.0 <= score <= 1.0

    def test_mape_range(self, synthetic_data):
        score = ScoringMetrics.mape(
            synthetic_data["sim_paths"], synthetic_data["hist_prices"]
        )
        assert 0.0 <= score <= 1.0

    def test_compute_all_returns_9_metrics(self, synthetic_data):
        metrics = ScoringMetrics.compute_all(
            sim_paths=synthetic_data["sim_paths"],
            sim_returns=synthetic_data["sim_returns"],
            hist_prices=synthetic_data["hist_prices"],
            hist_returns=synthetic_data["hist_returns"],
        )
        assert len(metrics) == 9
        assert all(0.0 <= v <= 1.0 for v in metrics.values())

    def test_compute_all_keys_match_weights(self, synthetic_data):
        metrics = ScoringMetrics.compute_all(
            sim_paths=synthetic_data["sim_paths"],
            sim_returns=synthetic_data["sim_returns"],
            hist_prices=synthetic_data["hist_prices"],
            hist_returns=synthetic_data["hist_returns"],
        )
        assert set(metrics.keys()) == set(DEFAULT_WEIGHTS.keys())


class TestDrawdowns:
    def test_no_drawdown_in_monotonic_increase(self):
        prices = np.linspace(100, 200, 100)
        dds = ScoringMetrics._compute_drawdowns(prices)
        assert len(dds) == 0

    def test_single_drawdown(self):
        # Up, down, recover
        prices = np.array([100, 110, 105, 95, 100, 110, 115])
        dds = ScoringMetrics._compute_drawdowns(prices)
        assert len(dds) >= 1
        assert all(d["depth"] > 0 for d in dds)

    def test_drawdown_at_end_not_recovered(self):
        prices = np.array([100, 110, 105, 90])
        dds = ScoringMetrics._compute_drawdowns(prices)
        unrecovered = [d for d in dds if not d["recovered"]]
        assert len(unrecovered) >= 1


class TestDiagnosticMetrics:
    def test_mae(self, synthetic_data):
        mae = DiagnosticMetrics.mae(
            synthetic_data["sim_paths"], synthetic_data["hist_prices"]
        )
        assert mae >= 0

    def test_mase(self, synthetic_data):
        mase = DiagnosticMetrics.mase(
            synthetic_data["sim_paths"],
            synthetic_data["hist_prices"],
            synthetic_data["hist_prices"],  # use as train for simplicity
        )
        assert mase > 0

    def test_moment_matching(self, synthetic_data):
        result = DiagnosticMetrics.moment_matching(
            synthetic_data["sim_returns"], synthetic_data["hist_returns"]
        )
        assert "mean_hist" in result
        assert "kurtosis_rel_error" in result

    def test_es_match(self, synthetic_data):
        es = DiagnosticMetrics.es_match(
            synthetic_data["sim_returns"], synthetic_data["hist_returns"]
        )
        assert es >= 0


class TestScorer:
    def test_weights_sum_to_one(self):
        scorer = Scorer()
        assert abs(sum(scorer.weights.values()) - 1.0) < 0.01

    def test_invalid_weights_raises(self):
        with pytest.raises(ValueError, match="sum to"):
            Scorer(weights={"ks_statistic": 0.5})  # Doesn't sum to 1

    def test_score_returns_scorecard(self, synthetic_data):
        scorer = Scorer()
        sim_result = SimulationResult(
            paths=synthetic_data["sim_paths"],
            log_returns=synthetic_data["sim_returns"],
            params_used={"mu": 0.1, "sigma": 0.4},
            model_name="test_model",
        )
        card = scorer.score(
            sim_result=sim_result,
            hist_prices=synthetic_data["hist_prices"],
            hist_returns=synthetic_data["hist_returns"],
            horizon="180d",
        )
        assert isinstance(card, ScoreCard)
        assert card.model_name == "test_model"
        assert card.horizon == "180d"
        assert 0.0 <= card.composite_score <= 1.0
        assert len(card.scoring_metrics) == 9
        assert len(card.diagnostic_metrics) > 0

    def test_score_quick_matches_score(self, synthetic_data):
        scorer = Scorer()
        sim_result = SimulationResult(
            paths=synthetic_data["sim_paths"],
            log_returns=synthetic_data["sim_returns"],
            params_used={},
            model_name="test",
        )
        quick = scorer.score_quick(
            sim_result, synthetic_data["hist_prices"], synthetic_data["hist_returns"]
        )
        full = scorer.score(
            sim_result, synthetic_data["hist_prices"],
            synthetic_data["hist_returns"], "90d"
        )
        assert abs(quick - full.composite_score) < 1e-10
