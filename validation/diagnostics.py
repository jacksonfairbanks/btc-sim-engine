"""
Diagnostic metrics — reported alongside results for context, NOT scored.

These do not feed into the composite score or leaderboard ranking.
"""
import numpy as np
from scipy import stats


class DiagnosticMetrics:
    """Compute diagnostic metrics for simulated vs historical data."""

    @staticmethod
    def mae(sim_paths: np.ndarray, hist_prices: np.ndarray) -> float:
        """
        Mean Absolute Error of median simulated path vs realized (in dollars).

        Parameters
        ----------
        sim_paths : np.ndarray
            Simulated price paths, shape (n_sims, n_steps).
        hist_prices : np.ndarray
            Realized prices, 1D.

        Returns
        -------
        float
            MAE in price units.
        """
        n = min(sim_paths.shape[1], len(hist_prices))
        median_path = np.median(sim_paths[:, :n], axis=0)
        return float(np.mean(np.abs(median_path - hist_prices[:n])))

    @staticmethod
    def mase(sim_paths: np.ndarray, hist_prices: np.ndarray,
             train_prices: np.ndarray) -> float:
        """
        Mean Absolute Scaled Error.

        Normalizes MAE by the in-sample naive forecast error.
        MASE > 1 means the model is worse than predicting no change.

        Parameters
        ----------
        sim_paths : np.ndarray
            Simulated price paths, shape (n_sims, n_steps).
        hist_prices : np.ndarray
            Out-of-sample realized prices, 1D.
        train_prices : np.ndarray
            In-sample prices used for naive baseline.

        Returns
        -------
        float
            MASE value. Lower is better; >1 is worse than naive.
        """
        n = min(sim_paths.shape[1], len(hist_prices))
        median_path = np.median(sim_paths[:, :n], axis=0)
        model_mae = np.mean(np.abs(median_path - hist_prices[:n]))

        # Naive forecast error: |P(t) - P(t-1)| on training data
        naive_errors = np.abs(np.diff(train_prices))
        naive_mae = np.mean(naive_errors)

        if naive_mae == 0:
            return float('inf')

        return float(model_mae / naive_mae)

    @staticmethod
    def moment_matching(sim_returns: np.ndarray, hist_returns: np.ndarray) -> dict[str, float]:
        """
        Compare first 4 moments of log returns.

        Parameters
        ----------
        sim_returns : np.ndarray
            Simulated log returns (flattened internally).
        hist_returns : np.ndarray
            Historical log returns, 1D.

        Returns
        -------
        dict
            Moment name -> relative error (0 = perfect match).
        """
        sim_flat = sim_returns.flatten()

        hist_moments = {
            "mean": np.mean(hist_returns),
            "variance": np.var(hist_returns),
            "skewness": float(stats.skew(hist_returns)),
            "kurtosis": float(stats.kurtosis(hist_returns)),
        }
        sim_moments = {
            "mean": np.mean(sim_flat),
            "variance": np.var(sim_flat),
            "skewness": float(stats.skew(sim_flat)),
            "kurtosis": float(stats.kurtosis(sim_flat)),
        }

        result = {}
        for key in hist_moments:
            h = hist_moments[key]
            s = sim_moments[key]
            denom = max(abs(h), 1e-8)
            result[f"{key}_hist"] = h
            result[f"{key}_sim"] = s
            result[f"{key}_rel_error"] = abs(s - h) / denom

        return result

    @staticmethod
    def es_match(sim_returns: np.ndarray, hist_returns: np.ndarray,
                 alpha: float = 0.05) -> float:
        """
        Expected Shortfall (CVaR) comparison.

        Compares average loss given a tail event between simulated and historical.

        Parameters
        ----------
        sim_returns : np.ndarray
            Simulated log returns (flattened internally).
        hist_returns : np.ndarray
            Historical log returns, 1D.
        alpha : float
            Tail threshold (default 5%).

        Returns
        -------
        float
            Relative error between simulated and historical ES. 0 = perfect.
        """
        sim_flat = sim_returns.flatten()

        hist_var = np.percentile(hist_returns, alpha * 100)
        sim_var = np.percentile(sim_flat, alpha * 100)

        hist_tail = hist_returns[hist_returns <= hist_var]
        sim_tail = sim_flat[sim_flat <= sim_var]

        hist_es = np.mean(hist_tail) if len(hist_tail) > 0 else 0.0
        sim_es = np.mean(sim_tail) if len(sim_tail) > 0 else 0.0

        denom = max(abs(hist_es), 1e-8)
        return float(abs(sim_es - hist_es) / denom)

    @staticmethod
    def max_drawdown_depth_dist(sim_paths: np.ndarray, hist_prices: np.ndarray,
                                 window_size: int = 90) -> dict[str, float]:
        """
        Compare distribution of max drawdown magnitudes across rolling windows.

        Parameters
        ----------
        sim_paths : np.ndarray
            Simulated price paths, shape (n_sims, n_steps).
        hist_prices : np.ndarray
            Historical price series, 1D.
        window_size : int
            Rolling window size in days.

        Returns
        -------
        dict
            Statistics about max drawdown distributions.
        """
        def _rolling_max_dd(prices: np.ndarray, window: int) -> np.ndarray:
            """Max drawdown in each rolling window."""
            max_dds = []
            for start in range(0, len(prices) - window + 1, window // 2):
                chunk = prices[start:start + window]
                running_max = np.maximum.accumulate(chunk)
                dd = (chunk - running_max) / running_max
                max_dds.append(float(np.min(dd)))
            return np.array(max_dds)

        hist_dds = _rolling_max_dd(hist_prices, window_size)

        # Pool across simulated paths
        sim_dds = []
        for i in range(min(sim_paths.shape[0], 200)):
            sim_dds.extend(_rolling_max_dd(sim_paths[i], window_size).tolist())
        sim_dds = np.array(sim_dds)

        result = {
            "hist_median_max_dd": float(np.median(hist_dds)) if len(hist_dds) > 0 else 0.0,
            "sim_median_max_dd": float(np.median(sim_dds)) if len(sim_dds) > 0 else 0.0,
            "hist_p95_max_dd": float(np.percentile(hist_dds, 5)) if len(hist_dds) > 0 else 0.0,
            "sim_p95_max_dd": float(np.percentile(sim_dds, 5)) if len(sim_dds) > 0 else 0.0,
        }

        if len(hist_dds) >= 2 and len(sim_dds) >= 2:
            ks_stat, _ = stats.ks_2samp(sim_dds, hist_dds)
            result["ks_statistic"] = float(ks_stat)
        else:
            result["ks_statistic"] = float('nan')

        return result

    @staticmethod
    def compute_all(
        sim_paths: np.ndarray,
        sim_returns: np.ndarray,
        hist_prices: np.ndarray,
        hist_returns: np.ndarray,
        train_prices: np.ndarray | None = None,
    ) -> dict[str, object]:
        """
        Compute all diagnostic metrics.

        Parameters
        ----------
        sim_paths : np.ndarray
            Simulated price paths, shape (n_sims, n_steps).
        sim_returns : np.ndarray
            Simulated log returns, shape (n_sims, n_steps-1).
        hist_prices : np.ndarray
            Historical price series, 1D.
        hist_returns : np.ndarray
            Historical log returns, 1D.
        train_prices : np.ndarray, optional
            Training prices for MASE calculation.

        Returns
        -------
        dict
            Diagnostic name -> value or sub-dict.
        """
        from .metrics import ScoringMetrics

        d = DiagnosticMetrics
        result = {
            "mae": d.mae(sim_paths, hist_prices),
            "moment_matching": d.moment_matching(sim_returns, hist_returns),
            "es_match": d.es_match(sim_returns, hist_returns),
            "max_drawdown_depth_dist": d.max_drawdown_depth_dist(sim_paths, hist_prices),
            "var_backtest_kupiec": ScoringMetrics.var_backtest_kupiec_detailed(
                sim_returns, hist_returns
            ),
        }

        if train_prices is not None:
            result["mase"] = d.mase(sim_paths, hist_prices, train_prices)

        return result
