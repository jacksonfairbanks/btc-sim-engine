"""
Scoring metrics — 9 metrics that feed into the composite score.

Each metric: (simulated_paths, historical_data, ...) -> float in [0, 1]
where 1 = perfect match to historical behavior.

All metrics operate on price paths, not returns in isolation.

compute_all() returns normalized scores only (backward-compatible).
compute_all_detailed() returns both normalized scores and raw intermediates.
"""
import numpy as np
from scipy import stats
from typing import Tuple


class ScoringMetrics:
    """Compute individual scoring metrics for simulated vs historical paths."""

    # ── Distributional (2) ──────────────────────────────────────────────

    @staticmethod
    def ks_statistic(sim_returns: np.ndarray, hist_returns: np.ndarray) -> float:
        """
        Kolmogorov-Smirnov test comparing return distributions.

        Parameters
        ----------
        sim_returns : np.ndarray
            Simulated log returns, shape (n_sims, n_steps) or 1D pooled.
        hist_returns : np.ndarray
            Historical log returns, 1D.

        Returns
        -------
        float
            Score in [0, 1] where 1 = distributions are identical.
        """
        result = ScoringMetrics.ks_statistic_detailed(sim_returns, hist_returns)
        return result["score"]

    @staticmethod
    def ks_statistic_detailed(sim_returns: np.ndarray, hist_returns: np.ndarray) -> dict:
        """KS test with raw intermediates."""
        sim_flat = sim_returns.flatten()
        ks_stat, p_value = stats.ks_2samp(sim_flat, hist_returns)
        score = 1.0 - ks_stat
        return {
            "score": float(score),
            "raw": {
                "ks_stat": float(ks_stat),
                "p_value": float(p_value),
            },
        }

    @staticmethod
    def qq_divergence(sim_returns: np.ndarray, hist_returns: np.ndarray,
                      n_quantiles: int = 100) -> float:
        """
        Mean squared error between quantiles on a QQ plot.

        Parameters
        ----------
        sim_returns : np.ndarray
            Simulated log returns (flattened internally).
        hist_returns : np.ndarray
            Historical log returns, 1D.
        n_quantiles : int
            Number of quantile points to compare.

        Returns
        -------
        float
            Score in [0, 1] where 1 = perfect quantile match.
        """
        result = ScoringMetrics.qq_divergence_detailed(sim_returns, hist_returns, n_quantiles)
        return result["score"]

    @staticmethod
    def qq_divergence_detailed(sim_returns: np.ndarray, hist_returns: np.ndarray,
                               n_quantiles: int = 100) -> dict:
        """QQ divergence with raw intermediates."""
        sim_flat = sim_returns.flatten()
        quantiles = np.linspace(0, 100, n_quantiles + 2)[1:-1]  # exclude 0/100

        sim_q = np.percentile(sim_flat, quantiles)
        hist_q = np.percentile(hist_returns, quantiles)

        mse = float(np.mean((sim_q - hist_q) ** 2))
        hist_var = float(np.var(hist_q))
        if hist_var == 0:
            score = 1.0 if mse == 0 else 0.0
            normalized_mse = 0.0
        else:
            normalized_mse = mse / hist_var
            score = float(np.exp(-normalized_mse))

        return {
            "score": score,
            "raw": {
                "mse": mse,
                "normalized_mse": normalized_mse,
                "hist_quantile_variance": hist_var,
            },
        }

    # ── Tail / Extreme Risk (2) ─────────────────────────────────────────

    @staticmethod
    def _hill_estimator(returns: np.ndarray, tail_fraction: float = 0.05) -> float:
        """
        Hill estimator for the tail index of the left (crash) tail.

        Parameters
        ----------
        returns : np.ndarray
            1D array of returns.
        tail_fraction : float
            Fraction of data to use for tail estimation.

        Returns
        -------
        float
            Hill tail index estimate (higher = thinner tail).
        """
        losses = -returns  # flip sign so we're looking at losses
        losses_sorted = np.sort(losses)[::-1]  # descending
        k = max(int(len(losses_sorted) * tail_fraction), 10)
        k = min(k, len(losses_sorted) - 1)

        top_k = losses_sorted[:k]
        threshold = losses_sorted[k]

        if threshold <= 0:
            return float('inf')

        log_ratios = np.log(top_k / threshold)
        hill_index = 1.0 / np.mean(log_ratios) if np.mean(log_ratios) > 0 else float('inf')
        return hill_index

    @staticmethod
    def tail_index_accuracy(sim_returns: np.ndarray, hist_returns: np.ndarray) -> float:
        """
        Compare Hill estimator tail indices on the left (crash) tail.

        Parameters
        ----------
        sim_returns : np.ndarray
            Simulated log returns.
        hist_returns : np.ndarray
            Historical log returns, 1D.

        Returns
        -------
        float
            Score in [0, 1] where 1 = tail indices match perfectly.
        """
        return ScoringMetrics.tail_index_accuracy_detailed(sim_returns, hist_returns)["score"]

    @staticmethod
    def tail_index_accuracy_detailed(sim_returns: np.ndarray, hist_returns: np.ndarray) -> dict:
        """Tail index comparison with raw intermediates."""
        sim_flat = sim_returns.flatten()
        hist_hill = ScoringMetrics._hill_estimator(hist_returns)
        sim_hill = ScoringMetrics._hill_estimator(sim_flat)

        if np.isinf(hist_hill) or np.isinf(sim_hill):
            return {
                "score": 0.0,
                "raw": {
                    "hist_hill_index": float(hist_hill) if not np.isinf(hist_hill) else None,
                    "sim_hill_index": float(sim_hill) if not np.isinf(sim_hill) else None,
                    "relative_error": None,
                },
            }

        rel_error = abs(sim_hill - hist_hill) / max(abs(hist_hill), 1e-8)
        score = float(np.exp(-rel_error))
        return {
            "score": score,
            "raw": {
                "hist_hill_index": float(hist_hill),
                "sim_hill_index": float(sim_hill),
                "relative_error": float(rel_error),
            },
        }

    @staticmethod
    def var_backtest_kupiec(sim_returns: np.ndarray, hist_returns: np.ndarray) -> float:
        """
        Kupiec VaR backtest at 1% and 5% levels.

        Tests whether simulated VaR thresholds are breached at the expected
        frequency in historical data.

        Parameters
        ----------
        sim_returns : np.ndarray
            Simulated log returns.
        hist_returns : np.ndarray
            Historical log returns, 1D.

        Returns
        -------
        float
            Score in [0, 1] where 1 = VaR is perfectly calibrated.
        """
        return ScoringMetrics.var_backtest_kupiec_detailed(sim_returns, hist_returns)["score"]

    @staticmethod
    def var_backtest_kupiec_detailed(sim_returns: np.ndarray, hist_returns: np.ndarray) -> dict:
        """Kupiec VaR backtest with raw intermediates."""
        sim_flat = sim_returns.flatten()
        scores = []
        raw = {}
        n_total = len(hist_returns)

        for alpha in [0.01, 0.05]:
            label = f"{int(alpha*100)}pct"
            var_threshold = float(np.percentile(sim_flat, alpha * 100))
            n_breaches = int(np.sum(hist_returns < var_threshold))
            observed_rate = n_breaches / n_total

            raw[f"var_{label}_threshold"] = var_threshold
            raw[f"observed_rate_{label}"] = float(observed_rate)
            raw[f"expected_rate_{label}"] = alpha
            raw[f"n_breaches_{label}"] = n_breaches
            raw[f"n_total"] = n_total

            if observed_rate == 0 or observed_rate == 1:
                ratio_error = abs(observed_rate - alpha) / alpha
                p_val = float(np.exp(-ratio_error))
                raw[f"kupiec_lr_stat_{label}"] = None
                raw[f"kupiec_p_value_{label}"] = p_val
                scores.append(p_val)
                continue

            lr_stat = -2 * (
                n_breaches * np.log(alpha / observed_rate) +
                (n_total - n_breaches) * np.log((1 - alpha) / (1 - observed_rate))
            )
            p_value = 1.0 - stats.chi2.cdf(abs(lr_stat), df=1)
            raw[f"kupiec_lr_stat_{label}"] = float(lr_stat)
            raw[f"kupiec_p_value_{label}"] = float(p_value)
            scores.append(p_value)

        return {
            "score": float(np.mean(scores)),
            "raw": raw,
        }

    # ── Path Dynamics (3) — CRITICAL ────────────────────────────────────

    @staticmethod
    def _compute_drawdowns(prices: np.ndarray) -> list[dict]:
        """
        Compute drawdown events from a 1D price series.

        Returns list of dicts with keys:
        - start_idx: index where drawdown begins (peak)
        - trough_idx: index of lowest point
        - end_idx: index where price recovers to peak (or end of series)
        - duration: total steps from peak to recovery
        - depth: max percentage loss from peak
        - recovery_time: steps from trough to recovery
        - recovered: bool, whether the drawdown fully recovered
        """
        running_max = np.maximum.accumulate(prices)
        drawdown_pct = (prices - running_max) / running_max

        drawdowns = []
        in_drawdown = False
        start_idx = 0
        trough_idx = 0
        trough_val = 0.0

        for i in range(len(prices)):
            if drawdown_pct[i] < -1e-8:  # In a drawdown
                if not in_drawdown:
                    in_drawdown = True
                    start_idx = i - 1 if i > 0 else 0
                    trough_idx = i
                    trough_val = drawdown_pct[i]
                elif drawdown_pct[i] < trough_val:
                    trough_idx = i
                    trough_val = drawdown_pct[i]
            else:
                if in_drawdown:
                    drawdowns.append({
                        "start_idx": start_idx,
                        "trough_idx": trough_idx,
                        "end_idx": i,
                        "duration": i - start_idx,
                        "depth": abs(trough_val),
                        "recovery_time": i - trough_idx,
                        "recovered": True,
                    })
                    in_drawdown = False

        # Handle drawdown at end of series
        if in_drawdown:
            drawdowns.append({
                "start_idx": start_idx,
                "trough_idx": trough_idx,
                "end_idx": len(prices) - 1,
                "duration": len(prices) - 1 - start_idx,
                "depth": abs(trough_val),
                "recovery_time": len(prices) - 1 - trough_idx,
                "recovered": False,
            })

        return drawdowns

    @staticmethod
    def _ks_score(sample1: np.ndarray, sample2: np.ndarray) -> float:
        """KS-based similarity score between two samples."""
        if len(sample1) < 2 or len(sample2) < 2:
            return 0.0
        ks_stat, _ = stats.ks_2samp(sample1, sample2)
        return 1.0 - ks_stat

    @staticmethod
    def drawdown_duration_dist(sim_paths: np.ndarray, hist_prices: np.ndarray) -> float:
        """
        Compare distribution of drawdown durations.

        Parameters
        ----------
        sim_paths : np.ndarray
            Simulated price paths, shape (n_sims, n_steps).
        hist_prices : np.ndarray
            Historical price series, 1D.

        Returns
        -------
        float
            Score in [0, 1] where 1 = duration distributions match.
        """
        return ScoringMetrics.drawdown_duration_dist_detailed(sim_paths, hist_prices)["score"]

    @staticmethod
    def drawdown_duration_dist_detailed(sim_paths: np.ndarray, hist_prices: np.ndarray) -> dict:
        """Drawdown duration comparison with raw intermediates."""
        hist_dds = ScoringMetrics._compute_drawdowns(hist_prices)
        hist_durations = np.array([d["duration"] for d in hist_dds])

        if len(hist_durations) < 2:
            return {
                "score": 0.5,
                "raw": {
                    "hist_n_drawdowns": len(hist_durations),
                    "sim_n_drawdowns": 0,
                    "hist_median_duration_days": float(np.median(hist_durations)) if len(hist_durations) > 0 else None,
                    "hist_mean_duration_days": float(np.mean(hist_durations)) if len(hist_durations) > 0 else None,
                    "hist_max_duration_days": float(np.max(hist_durations)) if len(hist_durations) > 0 else None,
                    "sim_median_duration_days": None,
                    "sim_mean_duration_days": None,
                    "sim_max_duration_days": None,
                    "ks_stat": None,
                },
            }

        sim_durations = []
        for i in range(sim_paths.shape[0]):
            dds = ScoringMetrics._compute_drawdowns(sim_paths[i])
            sim_durations.extend([d["duration"] for d in dds])
        sim_durations = np.array(sim_durations)

        if len(sim_durations) < 2:
            return {
                "score": 0.0,
                "raw": {
                    "hist_n_drawdowns": len(hist_durations),
                    "sim_n_drawdowns": len(sim_durations),
                    "hist_median_duration_days": float(np.median(hist_durations)),
                    "hist_mean_duration_days": float(np.mean(hist_durations)),
                    "hist_max_duration_days": float(np.max(hist_durations)),
                    "sim_median_duration_days": float(np.median(sim_durations)) if len(sim_durations) > 0 else None,
                    "sim_mean_duration_days": float(np.mean(sim_durations)) if len(sim_durations) > 0 else None,
                    "sim_max_duration_days": float(np.max(sim_durations)) if len(sim_durations) > 0 else None,
                    "ks_stat": None,
                },
            }

        ks_stat, _ = stats.ks_2samp(sim_durations, hist_durations)
        score = 1.0 - ks_stat
        return {
            "score": float(score),
            "raw": {
                "hist_n_drawdowns": len(hist_durations),
                "sim_n_drawdowns": len(sim_durations),
                "hist_median_duration_days": float(np.median(hist_durations)),
                "hist_mean_duration_days": float(np.mean(hist_durations)),
                "hist_max_duration_days": float(np.max(hist_durations)),
                "sim_median_duration_days": float(np.median(sim_durations)),
                "sim_mean_duration_days": float(np.mean(sim_durations)),
                "sim_max_duration_days": float(np.max(sim_durations)),
                "ks_stat": float(ks_stat),
            },
        }

    @staticmethod
    def recovery_time_dist(sim_paths: np.ndarray, hist_prices: np.ndarray) -> float:
        """
        Compare distribution of recovery times (trough to prior peak).

        Parameters
        ----------
        sim_paths : np.ndarray
            Simulated price paths, shape (n_sims, n_steps).
        hist_prices : np.ndarray
            Historical price series, 1D.

        Returns
        -------
        float
            Score in [0, 1] where 1 = recovery time distributions match.
        """
        return ScoringMetrics.recovery_time_dist_detailed(sim_paths, hist_prices)["score"]

    @staticmethod
    def recovery_time_dist_detailed(sim_paths: np.ndarray, hist_prices: np.ndarray) -> dict:
        """Recovery time comparison with raw intermediates."""
        hist_dds = ScoringMetrics._compute_drawdowns(hist_prices)
        hist_recovery = np.array([d["recovery_time"] for d in hist_dds if d["recovered"]])

        if len(hist_recovery) < 2:
            return {
                "score": 0.5,
                "raw": {
                    "hist_n_recoveries": len(hist_recovery),
                    "sim_n_recoveries": 0,
                    "hist_median_recovery_days": float(np.median(hist_recovery)) if len(hist_recovery) > 0 else None,
                    "hist_mean_recovery_days": float(np.mean(hist_recovery)) if len(hist_recovery) > 0 else None,
                    "sim_median_recovery_days": None,
                    "sim_mean_recovery_days": None,
                    "ks_stat": None,
                },
            }

        sim_recovery = []
        for i in range(sim_paths.shape[0]):
            dds = ScoringMetrics._compute_drawdowns(sim_paths[i])
            sim_recovery.extend([d["recovery_time"] for d in dds if d["recovered"]])
        sim_recovery = np.array(sim_recovery)

        if len(sim_recovery) < 2:
            return {
                "score": 0.0,
                "raw": {
                    "hist_n_recoveries": len(hist_recovery),
                    "sim_n_recoveries": len(sim_recovery),
                    "hist_median_recovery_days": float(np.median(hist_recovery)),
                    "hist_mean_recovery_days": float(np.mean(hist_recovery)),
                    "sim_median_recovery_days": float(np.median(sim_recovery)) if len(sim_recovery) > 0 else None,
                    "sim_mean_recovery_days": float(np.mean(sim_recovery)) if len(sim_recovery) > 0 else None,
                    "ks_stat": None,
                },
            }

        ks_stat, _ = stats.ks_2samp(sim_recovery, hist_recovery)
        score = 1.0 - ks_stat
        return {
            "score": float(score),
            "raw": {
                "hist_n_recoveries": len(hist_recovery),
                "sim_n_recoveries": len(sim_recovery),
                "hist_median_recovery_days": float(np.median(hist_recovery)),
                "hist_mean_recovery_days": float(np.mean(hist_recovery)),
                "sim_median_recovery_days": float(np.median(sim_recovery)),
                "sim_mean_recovery_days": float(np.mean(sim_recovery)),
                "ks_stat": float(ks_stat),
            },
        }

    @staticmethod
    def time_in_drawdown_ratio(sim_paths: np.ndarray, hist_prices: np.ndarray) -> float:
        """
        Compare percentage of time spent below running maximum.

        Parameters
        ----------
        sim_paths : np.ndarray
            Simulated price paths, shape (n_sims, n_steps).
        hist_prices : np.ndarray
            Historical price series, 1D.

        Returns
        -------
        float
            Score in [0, 1] where 1 = ratios match perfectly.
        """
        return ScoringMetrics.time_in_drawdown_ratio_detailed(sim_paths, hist_prices)["score"]

    @staticmethod
    def time_in_drawdown_ratio_detailed(sim_paths: np.ndarray, hist_prices: np.ndarray) -> dict:
        """Time-in-drawdown ratio with raw intermediates."""
        hist_max = np.maximum.accumulate(hist_prices)
        hist_ratio = float(np.mean(hist_prices < hist_max * (1 - 1e-8)))

        sim_ratios = []
        for i in range(sim_paths.shape[0]):
            path = sim_paths[i]
            path_max = np.maximum.accumulate(path)
            ratio = np.mean(path < path_max * (1 - 1e-8))
            sim_ratios.append(ratio)

        sim_median_ratio = float(np.median(sim_ratios))

        if hist_ratio == 0:
            score = 1.0 if sim_median_ratio == 0 else 0.0
            rel_error = 0.0
        else:
            rel_error = abs(sim_median_ratio - hist_ratio) / hist_ratio
            score = float(np.exp(-2.0 * rel_error))

        return {
            "score": score,
            "raw": {
                "hist_ratio": hist_ratio,
                "sim_median_ratio": sim_median_ratio,
                "sim_mean_ratio": float(np.mean(sim_ratios)),
                "sim_std_ratio": float(np.std(sim_ratios)),
                "relative_error": float(rel_error),
            },
        }

    # ── Temporal Dependence (1) ─────────────────────────────────────────

    @staticmethod
    def _acf_squared(returns: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute ACF of squared returns at lags 1..max_lag."""
        r2 = returns ** 2
        r2_centered = r2 - np.mean(r2)
        var = np.var(r2)
        if var == 0:
            return np.zeros(max_lag)
        acf_vals = np.array([
            np.mean(r2_centered[lag:] * r2_centered[:-lag]) / var
            if lag > 0 else 1.0
            for lag in range(1, max_lag + 1)
        ])
        return acf_vals

    @staticmethod
    def vol_clustering_acf(sim_returns: np.ndarray, hist_returns: np.ndarray,
                           max_lag: int = 20) -> float:
        """
        Compare ACF of squared returns (volatility clustering proxy).

        Parameters
        ----------
        sim_returns : np.ndarray
            Simulated log returns, shape (n_sims, n_steps).
        hist_returns : np.ndarray
            Historical log returns, 1D.
        max_lag : int
            Maximum lag for ACF comparison.

        Returns
        -------
        float
            Score in [0, 1] where 1 = ACF profiles match.
        """
        return ScoringMetrics.vol_clustering_acf_detailed(sim_returns, hist_returns, max_lag)["score"]

    @staticmethod
    def vol_clustering_acf_detailed(sim_returns: np.ndarray, hist_returns: np.ndarray,
                                    max_lag: int = 20) -> dict:
        """Volatility clustering ACF comparison with raw intermediates."""
        hist_acf = ScoringMetrics._acf_squared(hist_returns, max_lag)

        sim_acfs = []
        n_paths = min(sim_returns.shape[0], 200)
        for i in range(n_paths):
            if sim_returns.shape[1] > max_lag + 1:
                sim_acfs.append(ScoringMetrics._acf_squared(sim_returns[i], max_lag))

        if not sim_acfs:
            return {
                "score": 0.0,
                "raw": {
                    "acf_rmse": None,
                    "hist_acf_lag1": float(hist_acf[0]) if len(hist_acf) > 0 else None,
                    "sim_acf_lag1": None,
                    "hist_acf_lag5": float(hist_acf[4]) if len(hist_acf) > 4 else None,
                    "sim_acf_lag5": None,
                },
            }

        sim_acf_mean = np.mean(sim_acfs, axis=0)
        rmse = float(np.sqrt(np.mean((sim_acf_mean - hist_acf) ** 2)))
        score = float(np.exp(-5.0 * rmse))

        return {
            "score": score,
            "raw": {
                "acf_rmse": rmse,
                "hist_acf_lag1": float(hist_acf[0]),
                "sim_acf_lag1": float(sim_acf_mean[0]),
                "hist_acf_lag5": float(hist_acf[4]) if len(hist_acf) > 4 else None,
                "sim_acf_lag5": float(sim_acf_mean[4]) if len(sim_acf_mean) > 4 else None,
            },
        }

    # ── Forecast Accuracy (1) ───────────────────────────────────────────

    @staticmethod
    def mape(sim_paths: np.ndarray, hist_prices: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error of median simulated path vs realized.

        Parameters
        ----------
        sim_paths : np.ndarray
            Simulated price paths, shape (n_sims, n_steps).
        hist_prices : np.ndarray
            Realized historical prices over the same horizon, 1D.

        Returns
        -------
        float
            Score in [0, 1] where 1 = perfect forecast.
        """
        return ScoringMetrics.mape_detailed(sim_paths, hist_prices)["score"]

    @staticmethod
    def mape_detailed(sim_paths: np.ndarray, hist_prices: np.ndarray) -> dict:
        """MAPE with raw intermediates."""
        n = min(sim_paths.shape[1], len(hist_prices))
        median_path = np.median(sim_paths[:, :n], axis=0)
        actual = hist_prices[:n]

        mask = actual > 0
        if not mask.any():
            return {
                "score": 0.0,
                "raw": {"mape_pct": None, "median_final_price": None, "actual_final_price": None},
            }

        ape = np.abs((median_path[mask] - actual[mask]) / actual[mask])
        mape_val = float(np.mean(ape))
        score = float(np.exp(-mape_val))

        return {
            "score": score,
            "raw": {
                "mape_pct": mape_val * 100,  # as percentage
                "median_final_price": float(median_path[-1]),
                "actual_final_price": float(actual[-1]),
                "median_mean_price": float(np.mean(median_path[mask])),
                "actual_mean_price": float(np.mean(actual[mask])),
            },
        }

    # ── Envelope Coverage (1) ──────────────────────────────────────

    @staticmethod
    def percentile_band_coverage(sim_paths: np.ndarray, hist_prices: np.ndarray) -> float:
        """
        Percentage of realized daily prices within the simulated 5th-95th
        percentile envelope. Directly measures whether the simulation
        brackets reality.

        Parameters
        ----------
        sim_paths : np.ndarray
            Simulated price paths, shape (n_sims, n_steps+1).
        hist_prices : np.ndarray
            Realized prices, 1D.

        Returns
        -------
        float
            Score in [0, 1] where 1 = 90%+ coverage.
        """
        return ScoringMetrics.percentile_band_coverage_detailed(
            sim_paths, hist_prices
        )["score"]

    @staticmethod
    def percentile_band_coverage_detailed(
        sim_paths: np.ndarray, hist_prices: np.ndarray
    ) -> dict:
        """Percentile band coverage with raw intermediates."""
        n = min(sim_paths.shape[1], len(hist_prices))
        p5 = np.percentile(sim_paths[:, :n], 5, axis=0)
        p95 = np.percentile(sim_paths[:, :n], 95, axis=0)
        actual = hist_prices[:n]

        inside = (actual >= p5) & (actual <= p95)
        coverage = float(np.mean(inside))

        # Score: 90%+ coverage = 1.0, linear decay below
        # At 0% coverage -> 0.0, at 45% -> 0.5, at 90%+ -> 1.0
        target = 0.90
        if coverage >= target:
            score = 1.0
        else:
            score = coverage / target

        return {
            "score": float(score),
            "raw": {
                "coverage_pct": coverage * 100,
                "target_pct": target * 100,
                "n_inside": int(np.sum(inside)),
                "n_total": int(n),
                "n_below_p5": int(np.sum(actual < p5)),
                "n_above_p95": int(np.sum(actual > p95)),
            },
        }

    # ── Convenience: compute all scoring metrics ────────────────────────

    @staticmethod
    def compute_all(
        sim_paths: np.ndarray,
        sim_returns: np.ndarray,
        hist_prices: np.ndarray,
        hist_returns: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute all 9 scoring metrics (normalized scores only).

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

        Returns
        -------
        dict
            Metric name -> score in [0, 1].
        """
        detailed = ScoringMetrics.compute_all_detailed(
            sim_paths, sim_returns, hist_prices, hist_returns
        )
        return {name: d["score"] for name, d in detailed.items()}

    @staticmethod
    def compute_all_detailed(
        sim_paths: np.ndarray,
        sim_returns: np.ndarray,
        hist_prices: np.ndarray,
        hist_returns: np.ndarray,
    ) -> dict[str, dict]:
        """
        Compute all 9 scoring metrics with raw intermediates.

        Returns
        -------
        dict
            Metric name -> {"score": float, "raw": dict}.
        """
        m = ScoringMetrics
        return {
            "ks_statistic": m.ks_statistic_detailed(sim_returns, hist_returns),
            "qq_divergence": m.qq_divergence_detailed(sim_returns, hist_returns),
            "tail_index_accuracy": m.tail_index_accuracy_detailed(sim_returns, hist_returns),
            "percentile_band_coverage": m.percentile_band_coverage_detailed(sim_paths, hist_prices),
            "drawdown_duration_dist": m.drawdown_duration_dist_detailed(sim_paths, hist_prices),
            "recovery_time_dist": m.recovery_time_dist_detailed(sim_paths, hist_prices),
            "time_in_drawdown_ratio": m.time_in_drawdown_ratio_detailed(sim_paths, hist_prices),
            "vol_clustering_acf": m.vol_clustering_acf_detailed(sim_returns, hist_returns),
            "mape": m.mape_detailed(sim_paths, hist_prices),
        }
