"""Tests for parameter search and experiment runner."""
import numpy as np
import pytest

from models.registry import get_model
from optimization.search import ParameterSearch
from validation.scorer import Scorer


@pytest.fixture
def historical_returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.04, 500)


@pytest.fixture
def historical_prices(historical_returns):
    return 30000 * np.exp(np.cumsum(np.concatenate([[0], historical_returns])))


class TestParameterSearch:
    def test_basic_search(self, historical_returns, historical_prices):
        """Search should find parameters and return results."""
        model = get_model("gbm")
        scorer = Scorer()

        def objective(m):
            m.fit(historical_returns)
            result = m.simulate(50, 30, historical_prices[0], seed=42)
            return scorer.score_quick(
                result,
                historical_prices[:31],
                historical_returns[:30],
            )

        search = ParameterSearch(
            model=model,
            objective_fn=objective,
            n_trials=5,
            search_phase="sobol_only",
            seed=42,
        )
        result = search.run()

        assert "best_params" in result
        assert "best_score" in result
        assert result["n_trials_completed"] == 5
        assert result["best_score"] > 0

    def test_sobol_then_tpe(self, historical_returns, historical_prices):
        """Two-phase search should also work."""
        model = get_model("gbm")

        def objective(m):
            m.fit(historical_returns)
            result = m.simulate(50, 30, historical_prices[0], seed=42)
            return 0.5  # Dummy score

        search = ParameterSearch(
            model=model,
            objective_fn=objective,
            n_trials=10,
            search_phase="sobol_then_tpe",
            sobol_ratio=0.4,
            seed=42,
        )
        result = search.run()
        assert result["n_trials_completed"] == 10

    def test_all_trials_recorded(self, historical_returns, historical_prices):
        model = get_model("gbm")

        def objective(m):
            m.fit(historical_returns)
            return 0.5

        search = ParameterSearch(model=model, objective_fn=objective,
                                 n_trials=3, seed=42)
        result = search.run()
        assert len(result["all_trials"]) == 3

    def test_objective_error_handled(self, historical_returns):
        """If objective raises, trial should score 0 not crash."""
        model = get_model("gbm")

        def bad_objective(m):
            raise ValueError("intentional error")

        search = ParameterSearch(model=model, objective_fn=bad_objective,
                                 n_trials=3, seed=42)
        result = search.run()
        assert result["n_trials_completed"] == 3
        assert result["best_score"] == 0.0
