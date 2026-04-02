"""Tests for model interface compliance and basic behavior."""
import numpy as np
import pytest

from models.base import BaseModel, SimulationResult
from models.registry import get_model, list_models


@pytest.fixture
def historical_returns():
    """Synthetic BTC-like daily log returns."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.04, 1000)


class TestRegistry:
    def test_gbm_registered(self):
        assert "gbm" in list_models()

    def test_get_unknown_model_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("nonexistent_model")

    def test_list_models_returns_sorted(self):
        models = list_models()
        assert models == sorted(models)


class TestGBM:
    def test_implements_interface(self):
        model = get_model("gbm")
        assert isinstance(model, BaseModel)
        assert hasattr(model, "fit")
        assert hasattr(model, "simulate")
        assert hasattr(model, "get_default_params")
        assert hasattr(model, "get_param_space")
        assert hasattr(model, "name")

    def test_name(self):
        model = get_model("gbm")
        assert model.name == "gbm"

    def test_default_params(self):
        model = get_model("gbm")
        params = model.get_default_params()
        assert "mu" in params
        assert "sigma" in params

    def test_param_space(self):
        model = get_model("gbm")
        space = model.get_param_space()
        # GBM has no hyperparameters — mu/sigma estimated from data via MLE
        assert isinstance(space, dict)
        assert len(space) == 0

    def test_fit(self, historical_returns):
        model = get_model("gbm")
        model.fit(historical_returns)
        assert model._is_fitted

    def test_simulate_before_fit_raises(self):
        model = get_model("gbm")
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.simulate(10, 30, 50000.0)

    def test_simulate_shapes(self, historical_returns):
        model = get_model("gbm")
        model.fit(historical_returns)
        result = model.simulate(n_simulations=50, n_steps=30, initial_price=50000.0, seed=42)

        assert isinstance(result, SimulationResult)
        assert result.paths.shape == (50, 31)  # n_sims x (n_steps + 1)
        assert result.log_returns.shape == (50, 30)  # n_sims x n_steps
        assert result.model_name == "gbm"

    def test_simulate_initial_price(self, historical_returns):
        model = get_model("gbm")
        model.fit(historical_returns)
        result = model.simulate(n_simulations=10, n_steps=30, initial_price=50000.0, seed=42)
        np.testing.assert_array_equal(result.paths[:, 0], 50000.0)

    def test_simulate_reproducibility(self, historical_returns):
        model = get_model("gbm")
        model.fit(historical_returns)
        r1 = model.simulate(10, 30, 50000.0, seed=42)
        r2 = model.simulate(10, 30, 50000.0, seed=42)
        np.testing.assert_array_equal(r1.paths, r2.paths)

    def test_simulate_different_seeds(self, historical_returns):
        model = get_model("gbm")
        model.fit(historical_returns)
        r1 = model.simulate(10, 30, 50000.0, seed=42)
        r2 = model.simulate(10, 30, 50000.0, seed=99)
        assert not np.array_equal(r1.paths, r2.paths)

    def test_simulate_no_nan_inf(self, historical_returns):
        model = get_model("gbm")
        model.fit(historical_returns)
        result = model.simulate(100, 365, 50000.0, seed=42)
        assert not np.isnan(result.paths).any()
        assert not np.isinf(result.paths).any()
        assert not np.isnan(result.log_returns).any()
        assert not np.isinf(result.log_returns).any()

    def test_simulate_positive_prices(self, historical_returns):
        model = get_model("gbm")
        model.fit(historical_returns)
        result = model.simulate(100, 365, 50000.0, seed=42)
        assert (result.paths > 0).all()

    def test_mle_estimation(self, historical_returns):
        """GBM always estimates mu/sigma from training data."""
        model = get_model("gbm")
        model.fit(historical_returns)
        # mu and sigma should be estimated from the data, not None
        assert model.mu is not None
        assert model.sigma is not None
        # Annualized mu should be in a plausible range for BTC
        assert -2.0 < model.mu < 5.0
        # Annualized sigma should be positive and reasonable
        assert 0.1 < model.sigma < 3.0

    def test_refit_updates_params(self, historical_returns):
        """Each fit() call re-estimates from the provided data."""
        model = get_model("gbm")
        model.fit(historical_returns[:200])
        mu_1 = model.mu
        model.fit(historical_returns[:800])
        mu_2 = model.mu
        # Different training data should produce different estimates
        assert mu_1 != mu_2

    def test_returns_consistent_with_paths(self, historical_returns):
        model = get_model("gbm")
        model.fit(historical_returns)
        result = model.simulate(10, 30, 50000.0, seed=42)

        # log_returns should equal log(P_{t+1}/P_t)
        expected_returns = np.log(result.paths[:, 1:] / result.paths[:, :-1])
        np.testing.assert_allclose(result.log_returns, expected_returns, rtol=1e-10)


class TestSimulationResult:
    def test_valid_construction(self):
        paths = np.random.rand(10, 31) * 50000
        log_returns = np.random.randn(10, 30) * 0.04
        result = SimulationResult(
            paths=paths,
            log_returns=log_returns,
            params_used={"mu": 0.1},
            model_name="test",
        )
        assert result.model_name == "test"

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="inconsistent"):
            SimulationResult(
                paths=np.random.rand(10, 31),
                log_returns=np.random.randn(10, 29),  # Wrong!
                params_used={},
                model_name="test",
            )

    def test_nan_raises(self):
        paths = np.random.rand(10, 31)
        paths[5, 10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            SimulationResult(
                paths=paths,
                log_returns=np.random.randn(10, 30),
                params_used={},
                model_name="test",
            )

    def test_inf_raises(self):
        paths = np.random.rand(10, 31)
        paths[5, 10] = np.inf
        with pytest.raises(ValueError, match="inf"):
            SimulationResult(
                paths=paths,
                log_returns=np.random.randn(10, 30),
                params_used={},
                model_name="test",
            )
