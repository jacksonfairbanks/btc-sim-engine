"""
Regime-Conditioned Block Bootstrap — flagship model.

Combines Markov regime switching with stationary block bootstrap to produce
price paths that preserve both regime-dependent dynamics and empirical
distributional properties.

Academic foundations:
- Hamilton (1989) — Markov switching model
- Politis & Romano (1994) — Stationary block bootstrap
- Kreiss & Paparoditis (2011) — Conditional bootstrap

Optimization uses univariate sensitivity analysis with joint confirmation:
Phase 1 — Block length sweep (regime params at defaults)
Phase 2 — Regime switching sweep (block length locked)
Phase 3 — Joint confirmation (top3 x top3 = 9 combos)
"""
import hashlib
import json
import pickle
import time
import numpy as np
import warnings
from pathlib import Path
from typing import Any

from .base import BaseModel, SimulationResult
from .registry import register_model


@register_model
class RegimeBlockBootstrapModel(BaseModel):
    """
    Regime-Conditioned Block Bootstrap price path generator.

    fit() classifies historical returns into bull/bear regimes via a
    2-state Markov Switching Model, then builds overlapping block pools
    per regime.

    simulate() chains blocks by probabilistically switching regimes
    via the fitted transition matrix.
    """

    def __init__(self):
        super().__init__()
        # Hyperparameters (set via set_params or Optuna)
        self.block_length_sampling: str = "geometric"
        self.mean_block_length: int = 20
        self.min_block_length: int = 5
        self.block_stride: int = 5
        self.min_pool_size: int = 8
        self.transition_matrix_method: str = "fitted"
        self.msm_variance_switching: bool = True
        self.msm_frequency: str = "weekly"  # "daily", "weekly", "monthly"
        self.regime_enabled: bool = True  # False = single pool, no regime switching
        self.regime_method: str = "msm"  # "msm" or "hmm_baum_welch"
        self.n_regimes: int = 2  # 2 or 3 (3 only with hmm_baum_welch)
        self.em_iter: int = 500  # Reduce to 200 for quick mode

        # Fitted state
        self._transition_matrix: np.ndarray | None = None
        self._stationary_dist: np.ndarray | None = None
        self._block_pools: dict[int, list[np.ndarray]] | None = None
        self._regime_labels: np.ndarray | None = None
        self._regime_means: np.ndarray | None = None
        self._regime_variances: np.ndarray | None = None
        self._msm_converged: bool = False
        self._fallback_used: bool = False
        self._convergence_log: list[str] = []
        self._timing: dict[str, float] = {}

        # MSM cache — avoids re-fitting when only block params change
        self._msm_cache_dir = Path("results/cache")
        self._last_msm_hash: str | None = None

    @property
    def name(self) -> str:
        return "regime_block_bootstrap"

    def get_default_params(self) -> dict[str, Any]:
        return {
            "block_length_sampling": "geometric",
            "mean_block_length": 20,
            "min_block_length": 5,
            "block_stride": 5,
            "min_pool_size": 8,
            "regime_enabled": True,
            "regime_method": "msm",
            "n_regimes": 2,
            "transition_matrix_method": "fitted",
            "msm_variance_switching": True,
            "msm_frequency": "weekly",
        }

    def get_param_space(self) -> dict[str, dict[str, Any]]:
        return {
            "block_length_sampling": {
                "type": "categorical", "choices": ["geometric", "fixed"],
            },
            "mean_block_length": {
                "type": "int", "low": 10, "high": 100,
            },
            "min_block_length": {
                "type": "int", "low": 3, "high": 20,
            },
            "block_stride": {
                "type": "int", "low": 1, "high": 10,
            },
            "min_pool_size": {
                "type": "int", "low": 5, "high": 15,
            },
            "transition_matrix_method": {
                "type": "categorical", "choices": ["fitted", "empirical"],
            },
            "msm_variance_switching": {
                "type": "categorical", "choices": [True, False],
            },
        }

    # ── Regime classification ───────────────────────────────────────

    def _regime_cache_key(self, returns: np.ndarray) -> str:
        """Hash of training data + config to key regime classification cache."""
        data_hash = hashlib.md5(returns.tobytes()).hexdigest()[:12]
        method = self.regime_method
        if method == "hmm_baum_welch":
            tied = "tied" if not self.msm_variance_switching else "full"
            return f"hmm_{data_hash}_k{self.n_regimes}_{tied}_em{self.em_iter}_{self.msm_frequency}"
        else:
            return f"msm_{data_hash}_vs{self.msm_variance_switching}_em{self.em_iter}_{self.msm_frequency}"

    def _load_msm_cache(self, cache_key: str) -> dict | None:
        """Load cached MSM result if available."""
        cache_file = self._msm_cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_msm_cache(self, cache_key: str, result: dict) -> None:
        """Save MSM result to cache."""
        self._msm_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._msm_cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass

    def _resample_returns(self, daily_returns: np.ndarray) -> tuple[np.ndarray, int]:
        """Resample daily returns to the configured frequency."""
        if self.msm_frequency == "weekly":
            period = 7
        elif self.msm_frequency == "monthly":
            period = 30
        else:
            return daily_returns, 1

        n_periods = len(daily_returns) // period
        resampled = np.array([
            daily_returns[i * period:(i + 1) * period].sum()
            for i in range(n_periods)
        ])
        return resampled, period

    def _fit_msm_single(self, returns: np.ndarray, k_regimes: int) -> Any:
        """Fit MSM with k_regimes, retry up to 3 times. Returns result or None."""
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        for attempt in range(3):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod = MarkovRegression(
                        returns, k_regimes=k_regimes,
                        switching_variance=self.msm_variance_switching,
                    )
                    if attempt == 0:
                        return mod.fit(disp=False, em_iter=self.em_iter)
                    else:
                        rng = np.random.default_rng(42 + attempt)
                        start = mod.start_params
                        start = start * (1 + 0.1 * rng.standard_normal(len(start)))
                        return mod.fit(disp=False, em_iter=self.em_iter,
                                       start_params=start)
            except Exception as e:
                self._convergence_log.append(
                    f"MSM {k_regimes}-state attempt {attempt + 1} failed: {str(e)[:80]}"
                )
        return None

    def _fit_msm(self, daily_returns: np.ndarray) -> bool:
        """
        Fit Markov Switching Model with frequency resampling and model selection.

        Fits both 2-state and 3-state MSMs on resampled returns, selects
        by BIC, then maps regime labels back to daily resolution.
        Uses disk cache to avoid re-fitting.
        """
        self._convergence_log = []

        # Check cache
        cache_key = self._regime_cache_key(daily_returns)
        cached = self._load_msm_cache(cache_key)
        if cached is not None:
            self._transition_matrix = cached["transition_matrix"]
            self._regime_labels = cached["regime_labels"]
            self._regime_means = cached["regime_means"]
            self._regime_variances = cached["regime_variances"]
            self._msm_converged = cached["msm_converged"]
            self._n_regimes = cached.get("n_regimes", 2)
            self._convergence_log.append(f"MSM loaded from cache ({cache_key})")
            return True

        t0 = time.time()

        # Resample returns
        resampled, period = self._resample_returns(daily_returns)
        freq_label = self.msm_frequency if self.msm_frequency != "daily" else "daily"
        self._convergence_log.append(
            f"MSM fitted on {freq_label} returns ({len(resampled)} observations)"
        )

        # Fit 2-state and 3-state, select by BIC
        res2 = self._fit_msm_single(resampled, 2)
        res3 = self._fit_msm_single(resampled, 3)

        if res2 is None and res3 is None:
            self._timing["msm_fit"] = time.time() - t0
            return False

        # Model selection by BIC (penalizes overfitting more than AIC)
        selected_res = None
        selected_k = 2
        if res2 is not None and res3 is not None:
            self._convergence_log.append(
                f"AIC 2-state: {res2.aic:.1f} | AIC 3-state: {res3.aic:.1f}"
            )
            self._convergence_log.append(
                f"BIC 2-state: {res2.bic:.1f} | BIC 3-state: {res3.bic:.1f}"
            )
            if res3.bic < res2.bic:
                selected_res = res3
                selected_k = 3
                self._convergence_log.append("Selected: 3-state (by BIC)")
            else:
                selected_res = res2
                selected_k = 2
                self._convergence_log.append("Selected: 2-state (by BIC)")
        elif res2 is not None:
            selected_res = res2
            selected_k = 2
        else:
            selected_res = res3
            selected_k = 3

        self._n_regimes = selected_k

        # Extract transition matrix (column-stochastic -> row-stochastic)
        P_col = selected_res.regime_transition[:, :, 0]
        P_row = P_col.T

        # Smoothed regime probabilities at resampled frequency
        sp = selected_res.smoothed_marginal_probabilities
        resampled_labels = np.argmax(sp, axis=1)

        # Sort regimes by mean return: 0=bull (highest), last=bear (lowest)
        regime_means_resampled = np.array([
            resampled[resampled_labels == r].mean()
            if (resampled_labels == r).sum() > 0 else 0.0
            for r in range(selected_k)
        ])
        order = np.argsort(regime_means_resampled)[::-1]  # highest first
        label_map = {int(order[i]): i for i in range(selected_k)}

        # Remap labels and transition matrix
        resampled_labels = np.array([label_map[l] for l in resampled_labels])
        P_remapped = np.zeros_like(P_row)
        for i in range(selected_k):
            for j in range(selected_k):
                P_remapped[label_map[i], label_map[j]] = P_row[i, j]
        P_row = P_remapped

        # Map resampled labels back to daily resolution
        n_daily = len(daily_returns)
        daily_labels = np.zeros(n_daily, dtype=int)
        for i in range(len(resampled_labels)):
            start = i * period
            end = min((i + 1) * period, n_daily)
            daily_labels[start:end] = resampled_labels[i]
        # Handle remainder days
        if len(resampled_labels) * period < n_daily:
            daily_labels[len(resampled_labels) * period:] = resampled_labels[-1]

        # Compute regime stats on DAILY returns using the mapped labels
        regime_means = np.array([
            daily_returns[daily_labels == r].mean()
            if (daily_labels == r).sum() > 0 else 0.0
            for r in range(selected_k)
        ])
        regime_vars = np.array([
            daily_returns[daily_labels == r].var()
            if (daily_labels == r).sum() > 0 else 0.0
            for r in range(selected_k)
        ])

        self._transition_matrix = P_row
        self._regime_labels = daily_labels
        self._regime_means = regime_means
        self._regime_variances = regime_vars
        self._msm_converged = True

        # Log regime run stats
        for r in range(selected_k):
            name = ["Bull", "Bear", "Chop"][r] if selected_k <= 3 else f"Regime {r}"
            days = int((daily_labels == r).sum())
            pct = days / len(daily_labels) * 100
            # Compute run lengths
            runs = []
            current_run = 0
            for lbl in daily_labels:
                if lbl == r:
                    current_run += 1
                elif current_run > 0:
                    runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)
            avg_run = float(np.mean(runs)) if runs else 0
            max_run = int(np.max(runs)) if runs else 0
            self._convergence_log.append(
                f"{name}: {days}d ({pct:.0f}%), avg run {avg_run:.0f}d, max {max_run}d"
            )
            if name == "Bear" and avg_run < 30:
                self._convergence_log.append(
                    f"WARNING: Average bear run {avg_run:.0f}d < 30d — "
                    f"may not capture sustained drawdowns"
                )

        elapsed = time.time() - t0
        self._timing["msm_fit"] = elapsed
        self._convergence_log.append(f"MSM fit completed in {elapsed:.1f}s")

        # Save to cache
        self._save_msm_cache(cache_key, {
            "transition_matrix": P_row,
            "regime_labels": daily_labels,
            "regime_means": regime_means,
            "regime_variances": regime_vars,
            "msm_converged": True,
            "n_regimes": selected_k,
        })
        return True

    def _empirical_regime_classification(self, returns: np.ndarray) -> None:
        """
        Fallback: classify regimes using 200-day rolling mean.
        Bull = returns above rolling mean, Bear = below.
        """
        window = min(200, len(returns) // 3)
        if window < 20:
            window = 20

        rolling_mean = np.convolve(
            returns, np.ones(window) / window, mode="valid"
        )
        # Pad the beginning with the first valid value
        pad = np.full(window - 1, rolling_mean[0])
        rolling_mean = np.concatenate([pad, rolling_mean])

        labels = (returns >= rolling_mean).astype(int)
        # 0 = bull (above mean), 1 = bear (below mean)
        # Invert if needed to match convention
        regime_means = np.array([
            returns[labels == r].mean() if (labels == r).sum() > 0 else 0.0
            for r in range(2)
        ])
        if regime_means[0] < regime_means[1]:
            labels = 1 - labels
            regime_means = regime_means[::-1]

        # Build empirical transition matrix by counting transitions
        transitions = np.zeros((2, 2))
        for i in range(len(labels) - 1):
            transitions[labels[i], labels[i + 1]] += 1
        # Normalize rows
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        P_row = transitions / row_sums

        regime_vars = np.array([
            returns[labels == r].var() if (labels == r).sum() > 0 else 0.0
            for r in range(2)
        ])

        self._transition_matrix = P_row
        self._regime_labels = labels
        self._regime_means = regime_means
        self._regime_variances = regime_vars
        self._msm_converged = False
        self._fallback_used = True
        self._convergence_log.append(
            "Fell back to empirical regime classification (200d rolling mean)"
        )

    def _compute_empirical_transition_matrix(
        self, labels: np.ndarray
    ) -> np.ndarray:
        """Count-based transition matrix from regime labels."""
        n_states = int(labels.max()) + 1
        transitions = np.zeros((n_states, n_states))
        for i in range(len(labels) - 1):
            transitions[labels[i], labels[i + 1]] += 1
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return transitions / row_sums

    # ── Baum-Welch HMM classification ────────────────────────────────

    def _fit_hmm(self, daily_returns: np.ndarray) -> bool:
        """
        Fit Gaussian HMM via Baum-Welch and classify regimes.

        Uses custom pure-numpy implementation (models/hmm.py) since
        hmmlearn requires C extensions unavailable on Python 3.14.

        States are labeled post-hoc by fitted means:
        - State 0 (highest mean) = bull
        - State 1 (middle) = base/sideways
        - State 2 (lowest mean, highest variance) = crisis

        For n_regimes=2, only 2 states are fit.
        """
        from .hmm import GaussianHMM

        self._convergence_log = getattr(self, '_convergence_log', [])
        k = self.n_regimes

        # Check cache
        cache_key = self._regime_cache_key(daily_returns)
        cached = self._load_msm_cache(cache_key)
        if cached is not None:
            self._transition_matrix = cached["transition_matrix"]
            self._regime_labels = cached["regime_labels"]
            self._regime_means = cached["regime_means"]
            self._regime_variances = cached["regime_variances"]
            self._msm_converged = cached["msm_converged"]
            self._n_regimes = cached.get("n_regimes", k)
            self._convergence_log.append(f"HMM loaded from cache ({cache_key})")
            return True

        t0 = time.time()

        # Resample to configured frequency
        resampled, period = self._resample_returns(daily_returns)
        freq_label = self.msm_frequency if self.msm_frequency != "daily" else "daily"
        self._convergence_log.append(
            f"HMM Baum-Welch fitting on {freq_label} returns "
            f"({len(resampled)} observations, {k} states)"
        )

        # Fit HMM
        tied = not self.msm_variance_switching
        hmm = GaussianHMM(
            n_states=k,
            max_iter=self.em_iter,
            tol=1e-6,
            n_restarts=5,
            tied_covariance=tied,
            seed=42,
        )

        try:
            hmm.fit(resampled)
        except Exception as e:
            self._convergence_log.append(f"HMM fit failed: {str(e)[:100]}")
            self._timing["hmm_fit"] = time.time() - t0
            return False

        self._convergence_log.append(
            f"HMM converged={hmm.converged_} in {hmm.n_iter_} iterations "
            f"(LL={hmm.log_likelihood_:.1f}, BIC={hmm.bic(resampled):.1f})"
        )

        # Get state labels at resampled frequency
        resampled_labels = hmm.decode(resampled)

        # Sort states by mean return: 0=bull (highest), last=crisis (lowest)
        order = np.argsort(hmm.means_)[::-1]  # highest mean first
        label_map = {int(order[i]): i for i in range(k)}

        # Remap labels
        resampled_labels = np.array([label_map[l] for l in resampled_labels])

        # Remap transition matrix
        P_fitted = hmm.transmat_.copy()
        P_remapped = np.zeros_like(P_fitted)
        for i in range(k):
            for j in range(k):
                P_remapped[label_map[i], label_map[j]] = P_fitted[i, j]

        # Remap means and variances
        remapped_means = hmm.means_[order]
        remapped_vars = hmm.variances_[order]

        # Map resampled labels back to daily resolution
        n_daily = len(daily_returns)
        daily_labels = np.zeros(n_daily, dtype=int)
        for i in range(len(resampled_labels)):
            start = i * period
            end = min((i + 1) * period, n_daily)
            daily_labels[start:end] = resampled_labels[i]
        if len(resampled_labels) * period < n_daily:
            daily_labels[len(resampled_labels) * period:] = resampled_labels[-1]

        # Compute regime stats on DAILY returns
        regime_means = np.array([
            daily_returns[daily_labels == r].mean()
            if (daily_labels == r).sum() > 0 else 0.0
            for r in range(k)
        ])
        regime_vars = np.array([
            daily_returns[daily_labels == r].var()
            if (daily_labels == r).sum() > 0 else 0.0
            for r in range(k)
        ])

        self._transition_matrix = P_remapped
        self._regime_labels = daily_labels
        self._regime_means = regime_means
        self._regime_variances = regime_vars
        self._msm_converged = hmm.converged_
        self._n_regimes = k

        # Log state info
        state_names = ["Bull", "Bear", "Crisis"] if k == 3 else ["Bull", "Bear"]
        for r in range(k):
            name = state_names[r] if r < len(state_names) else f"Regime {r}"
            days = int((daily_labels == r).sum())
            pct = days / len(daily_labels) * 100
            # Run lengths
            runs = []
            current_run = 0
            for lbl in daily_labels:
                if lbl == r:
                    current_run += 1
                elif current_run > 0:
                    runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)
            avg_run = float(np.mean(runs)) if runs else 0
            max_run = int(np.max(runs)) if runs else 0
            self._convergence_log.append(
                f"{name}: {days}d ({pct:.0f}%), avg run {avg_run:.0f}d, max {max_run}d, "
                f"daily_mean={regime_means[r]:.6f}, daily_std={np.sqrt(regime_vars[r]):.6f}"
            )

        # Log fitted (resampled) state params for sanity checking
        self._convergence_log.append(
            f"HMM fitted params ({freq_label}): "
            + " | ".join(
                f"{state_names[i] if i < len(state_names) else f'S{i}'}: "
                f"mu={remapped_means[i]:.6f}, sigma={np.sqrt(remapped_vars[i]):.6f}"
                for i in range(k)
            )
        )

        # Log transition matrix
        for i in range(k):
            row = " ".join(f"{P_remapped[i, j]:.4f}" for j in range(k))
            self._convergence_log.append(
                f"P[{state_names[i] if i < len(state_names) else f'S{i}'}] = [{row}]"
            )

        elapsed = time.time() - t0
        self._timing["hmm_fit"] = elapsed
        self._convergence_log.append(f"HMM fit completed in {elapsed:.1f}s")

        # Cache
        self._save_msm_cache(cache_key, {
            "transition_matrix": P_remapped,
            "regime_labels": daily_labels,
            "regime_means": regime_means,
            "regime_variances": regime_vars,
            "msm_converged": hmm.converged_,
            "n_regimes": k,
        })

        return True

    # ── Block pool construction ─────────────────────────────────────

    def _build_block_pools(self, returns: np.ndarray) -> dict[int, list[np.ndarray]]:
        """
        Build overlapping block pools per regime with configurable stride.

        Returns dict: {regime_id: [block_array, ...]}
        """
        labels = self._regime_labels
        block_len = self.mean_block_length
        stride = self.block_stride
        min_len = self.min_block_length

        n_reg = getattr(self, '_n_regimes', 2)
        pools = {r: [] for r in range(n_reg)}

        for regime in range(n_reg):
            # Find contiguous runs of this regime
            regime_mask = (labels == regime)
            runs = []
            start = None
            for i in range(len(regime_mask)):
                if regime_mask[i]:
                    if start is None:
                        start = i
                else:
                    if start is not None:
                        runs.append((start, i))
                        start = None
            if start is not None:
                runs.append((start, len(regime_mask)))

            # Extract overlapping blocks from each run
            for run_start, run_end in runs:
                run_returns = returns[run_start:run_end]
                run_len = len(run_returns)

                if run_len < min_len:
                    continue

                # For fixed blocks, extract at stride intervals
                for block_start in range(0, run_len - min_len + 1, stride):
                    block_end = min(block_start + block_len, run_len)
                    block = run_returns[block_start:block_end]
                    if len(block) >= min_len:
                        pools[regime].append(block)

        return pools

    # ── Stationary distribution ─────────────────────────────────────

    def _compute_stationary_distribution(
        self, P: np.ndarray
    ) -> np.ndarray:
        """Compute stationary distribution of row-stochastic transition matrix."""
        # Solve pi @ P = pi  =>  (P^T - I) @ pi = 0, sum(pi) = 1
        n = P.shape[0]
        A = P.T - np.eye(n)
        A[-1, :] = 1.0  # Replace last equation with normalization
        b = np.zeros(n)
        b[-1] = 1.0
        try:
            pi = np.linalg.solve(A, b)
            pi = np.clip(pi, 0, 1)
            pi /= pi.sum()
        except np.linalg.LinAlgError:
            pi = np.array([0.5, 0.5])
        return pi

    # ── fit() ───────────────────────────────────────────────────────

    def fit(self, historical_returns: np.ndarray) -> None:
        """
        Fit regime switching model and build block pools.

        Parameters
        ----------
        historical_returns : np.ndarray
            1D array of daily log returns.
        """
        if historical_returns.ndim != 1:
            raise ValueError("historical_returns must be 1D")

        self._fallback_used = False
        self._timing = {}
        self._convergence_log = []

        if not self.regime_enabled:
            # ── No regime switching: single pool of all returns ─────
            self._n_regimes = 1
            self._regime_labels = np.zeros(len(historical_returns), dtype=int)
            self._transition_matrix = np.array([[1.0]])
            self._stationary_dist = np.array([1.0])
            self._regime_means = np.array([float(np.mean(historical_returns))])
            self._regime_variances = np.array([float(np.var(historical_returns))])
            self._msm_converged = True
            self._convergence_log.append("Regime switching disabled — single pool mode")

            t0 = time.time()
            self._block_pools = self._build_block_pools(historical_returns)
            self._timing["block_pool"] = time.time() - t0

            self._convergence_log.append(
                f"Single pool: {len(self._block_pools[0])} blocks"
            )
        else:
            # ── Full regime switching pipeline ──────────────────────
            # Step 1: Regime classification
            if self.regime_method == "hmm_baum_welch":
                classify_ok = self._fit_hmm(historical_returns)
                if not classify_ok:
                    self._empirical_regime_classification(historical_returns)
            else:
                classify_ok = self._fit_msm(historical_returns)
                if not classify_ok:
                    self._empirical_regime_classification(historical_returns)

            # Step 2: Override transition matrix if empirical method requested
            if self.transition_matrix_method == "empirical":
                self._transition_matrix = self._compute_empirical_transition_matrix(
                    self._regime_labels
                )

            # Step 3: Compute stationary distribution
            self._stationary_dist = self._compute_stationary_distribution(
                self._transition_matrix
            )

            # Step 4: Build block pools
            t0 = time.time()
            self._block_pools = self._build_block_pools(historical_returns)
            self._timing["block_pool"] = time.time() - t0

        # Validate pool sizes
        for regime in range(getattr(self, '_n_regimes', 2)):
            pool = self._block_pools[regime]
            if len(pool) < self.min_pool_size:
                self._convergence_log.append(
                    f"WARNING: Regime {regime} has only {len(pool)} blocks "
                    f"(min_pool_size={self.min_pool_size}). "
                    f"Results may be unreliable."
                )

        self._fitted_params = {
            "regime_enabled": self.regime_enabled,
            "regime_method": self.regime_method,
            "block_length_sampling": self.block_length_sampling,
            "mean_block_length": self.mean_block_length,
            "min_block_length": self.min_block_length,
            "block_stride": self.block_stride,
            "transition_matrix_method": self.transition_matrix_method,
            "msm_variance_switching": self.msm_variance_switching,
            "msm_converged": self._msm_converged,
            "fallback_used": self._fallback_used,
            "n_regimes": self._n_regimes,
            "pool_sizes": {r: len(self._block_pools[r]) for r in range(self._n_regimes)},
            "bull_pool_size": len(self._block_pools[0]),
            "bear_pool_size": len(self._block_pools.get(1, [])) if self._n_regimes > 1 else 0,
            "crisis_pool_size": len(self._block_pools.get(2, [])) if self._n_regimes > 2 else 0,
            "bull_mean_return": float(self._regime_means[0]),
            "bear_mean_return": float(self._regime_means[1]) if self._n_regimes > 1 else None,
            "crisis_mean_return": float(self._regime_means[2]) if self._n_regimes > 2 else None,
            "stationary_dist_bull": float(self._stationary_dist[0]),
        }
        self._is_fitted = True

    # ── simulate() ──────────────────────────────────────────────────

    def simulate(
        self,
        n_simulations: int,
        n_steps: int,
        initial_price: float,
        seed: int | None = None,
        audit_mode: bool = False,
    ) -> SimulationResult:
        """
        Generate price paths by chaining regime-conditioned blocks.

        Parameters
        ----------
        n_simulations : int
            Number of paths to generate.
        n_steps : int
            Number of daily time steps per path.
        initial_price : float
            Starting price.
        seed : int, optional
            Random seed for reproducibility.
        audit_mode : bool
            If True, track block usage and regime transitions for audit.

        Returns
        -------
        SimulationResult
        """
        self._check_fitted()

        t0 = time.time()
        rng = np.random.default_rng(seed)
        P = self._transition_matrix
        pools = self._block_pools

        # Validate pools are not empty
        for regime in range(getattr(self, '_n_regimes', 2)):
            if len(pools[regime]) == 0:
                raise RuntimeError(
                    f"Regime {regime} block pool is empty. "
                    f"Cannot simulate. Try different block length parameters."
                )

        all_returns = np.zeros((n_simulations, n_steps))

        # Audit tracking
        n_reg = len(self._stationary_dist)
        if audit_mode:
            block_usage = {r: np.zeros(len(pools[r]), dtype=int) for r in range(n_reg)}
            regime_switches = {}
            for i in range(n_reg):
                for j in range(n_reg):
                    regime_switches[(i, j)] = 0
            sim_bull_runs = []
            sim_bear_runs = []

        for sim in range(n_simulations):
            current_regime = rng.choice(len(self._stationary_dist), p=self._stationary_dist)
            pos = 0
            current_run_len = 0

            while pos < n_steps:
                if self.block_length_sampling == "geometric":
                    p_geom = 1.0 / self.mean_block_length
                    bl = rng.geometric(p_geom)
                    bl = max(bl, self.min_block_length)
                else:
                    bl = self.mean_block_length

                pool = pools[current_regime]
                block_idx = rng.integers(0, len(pool))
                block = pool[block_idx]

                use_len = min(bl, len(block), n_steps - pos)
                all_returns[sim, pos:pos + use_len] = block[:use_len]
                pos += use_len
                current_run_len += use_len

                if audit_mode:
                    block_usage[current_regime][block_idx] += 1

                # Transition
                prev_regime = current_regime
                current_regime = rng.choice(len(P), p=P[current_regime])

                if audit_mode:
                    regime_switches[(prev_regime, current_regime)] += 1
                    if prev_regime != current_regime:
                        # Record completed run
                        if prev_regime == 0:
                            sim_bull_runs.append(current_run_len)
                        else:
                            sim_bear_runs.append(current_run_len)
                        current_run_len = 0

        # Build price paths from cumulative returns
        cum_returns = np.cumsum(all_returns, axis=1)
        paths = np.empty((n_simulations, n_steps + 1))
        paths[:, 0] = initial_price
        paths[:, 1:] = initial_price * np.exp(cum_returns)

        self._timing["simulation"] = time.time() - t0

        metadata = {
            "transition_matrix": self._transition_matrix.tolist(),
            "stationary_dist": self._stationary_dist.tolist(),
            "regime_means": self._regime_means.tolist(),
            "regime_variances": self._regime_variances.tolist(),
            "bull_pool_size": len(pools[0]),
            "bear_pool_size": len(pools.get(1, [])),
            "msm_converged": self._msm_converged,
            "fallback_used": self._fallback_used,
            "convergence_log": self._convergence_log,
            "regime_labels": self._regime_labels.tolist(),
            "initial_price": initial_price,
            "n_steps": n_steps,
            "timing": self._timing.copy(),
        }

        if audit_mode:
            total_switches = sum(regime_switches.values())
            metadata["audit"] = {
                "block_usage_bull": block_usage[0].tolist(),
                "block_usage_bear": block_usage.get(1, np.array([], dtype=int)).tolist(),
                "regime_switches": {f"{k[0]}->{k[1]}": v for k, v in regime_switches.items()},
                "total_transitions": total_switches,
                "sim_bull_run_lengths": sim_bull_runs,
                "sim_bear_run_lengths": sim_bear_runs,
                "bull_blocks_used": int(np.sum(block_usage[0] > 0)),
                "bear_blocks_used": int(np.sum(block_usage.get(1, np.array([], dtype=int)) > 0)),
                "bull_pool_total": len(pools[0]),
                "bear_pool_total": len(pools.get(1, [])),
            }

        return SimulationResult(
            paths=paths,
            log_returns=all_returns,
            params_used={
                "regime_enabled": self.regime_enabled,
                "regime_method": self.regime_method,
                "n_regimes": getattr(self, '_n_regimes', self.n_regimes),
                "block_length_sampling": self.block_length_sampling,
                "mean_block_length": self.mean_block_length,
                "min_block_length": self.min_block_length,
                "block_stride": self.block_stride,
                "min_pool_size": self.min_pool_size,
                "transition_matrix_method": self.transition_matrix_method,
                "msm_variance_switching": self.msm_variance_switching,
            },
            model_name=self.name,
            metadata=metadata,
        )
