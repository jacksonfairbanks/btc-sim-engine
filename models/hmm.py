"""
Baum-Welch Gaussian HMM — pure numpy implementation.

Fits a K-state univariate Gaussian Hidden Markov Model using the
Baum-Welch (EM) algorithm. No external dependencies beyond numpy/scipy.

Replaces hmmlearn.GaussianHMM which requires C extensions unavailable
on Python 3.14 / Windows without MSVC.

Academic reference: Rabiner (1989), "A tutorial on hidden Markov models
and selected applications in speech recognition."
"""
import hashlib
import numpy as np
from scipy.special import logsumexp
from typing import Any


class GaussianHMM:
    """
    Univariate Gaussian Hidden Markov Model with Baum-Welch fitting.

    Parameters
    ----------
    n_states : int
        Number of hidden states (e.g. 3 for bull/bear/crisis).
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on log-likelihood.
    n_restarts : int
        Number of random restarts (best log-likelihood wins).
    tied_covariance : bool
        If True, all states share one variance (analogous to
        msm_variance_switching=False). If False, each state has
        its own variance.
    seed : int or None
        Random seed for initialization.
    """

    def __init__(
        self,
        n_states: int = 3,
        max_iter: int = 500,
        tol: float = 1e-6,
        n_restarts: int = 5,
        tied_covariance: bool = False,
        seed: int | None = 42,
    ):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.tied_covariance = tied_covariance
        self.seed = seed

        # Fitted parameters
        self.means_: np.ndarray | None = None       # (n_states,)
        self.variances_: np.ndarray | None = None    # (n_states,)
        self.transmat_: np.ndarray | None = None     # (n_states, n_states) row-stochastic
        self.startprob_: np.ndarray | None = None    # (n_states,)
        self.log_likelihood_: float = -np.inf
        self.n_iter_: int = 0
        self.converged_: bool = False

    # ── Gaussian emission log-probabilities ────────────────────────

    @staticmethod
    def _log_gaussian(x: np.ndarray, mu: float, var: float) -> np.ndarray:
        """Log probability of x under N(mu, var). Returns shape (T,)."""
        return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)

    def _compute_log_emission(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute log emission probabilities.

        Parameters
        ----------
        obs : np.ndarray, shape (T,)

        Returns
        -------
        log_B : np.ndarray, shape (T, n_states)
        """
        T = len(obs)
        K = self.n_states
        log_B = np.empty((T, K))
        for k in range(K):
            log_B[:, k] = self._log_gaussian(obs, self.means_[k], self.variances_[k])
        return log_B

    # ── Forward-backward (log-space for numerical stability) ───────

    def _forward(self, log_B: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Forward pass in log-space.

        Returns
        -------
        log_alpha : np.ndarray, shape (T, K)
        log_likelihood : float
        """
        T, K = log_B.shape
        log_A = np.log(self.transmat_ + 1e-300)
        log_pi = np.log(self.startprob_ + 1e-300)

        log_alpha = np.empty((T, K))
        log_alpha[0] = log_pi + log_B[0]

        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = logsumexp(log_alpha[t - 1] + log_A[:, k]) + log_B[t, k]

        log_ll = logsumexp(log_alpha[-1])
        return log_alpha, log_ll

    def _backward(self, log_B: np.ndarray) -> np.ndarray:
        """
        Backward pass in log-space.

        Returns
        -------
        log_beta : np.ndarray, shape (T, K)
        """
        T, K = log_B.shape
        log_A = np.log(self.transmat_ + 1e-300)

        log_beta = np.zeros((T, K))  # log(1) = 0 for t = T-1

        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = logsumexp(
                    log_A[k, :] + log_B[t + 1, :] + log_beta[t + 1, :]
                )

        return log_beta

    # ── E-step: compute responsibilities ───────────────────────────

    def _e_step(self, obs: np.ndarray, log_B: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        E-step: compute gamma (state posteriors) and xi (transition posteriors).

        Returns
        -------
        gamma : np.ndarray, shape (T, K) — P(state=k | all obs)
        xi : np.ndarray, shape (T-1, K, K) — P(state_t=j, state_{t+1}=k | all obs)
        log_ll : float
        """
        T, K = log_B.shape
        log_A = np.log(self.transmat_ + 1e-300)

        log_alpha, log_ll = self._forward(log_B)
        log_beta = self._backward(log_B)

        # Gamma: P(z_t = k | obs)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Xi: P(z_t = j, z_{t+1} = k | obs)
        xi = np.empty((T - 1, K, K))
        for t in range(T - 1):
            log_xi_t = (
                log_alpha[t, :, None]       # (K, 1)
                + log_A                      # (K, K)
                + log_B[t + 1, None, :]      # (1, K)
                + log_beta[t + 1, None, :]   # (1, K)
            )
            log_xi_t -= logsumexp(log_xi_t)
            xi[t] = np.exp(log_xi_t)

        return gamma, xi, log_ll

    # ── M-step: update parameters ──────────────────────────────────

    def _m_step(self, obs: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        """M-step: re-estimate parameters from responsibilities."""
        T, K = gamma.shape

        # Start probabilities
        self.startprob_ = gamma[0] / gamma[0].sum()

        # Transition matrix
        xi_sum = xi.sum(axis=0)  # (K, K)
        row_sums = xi_sum.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.transmat_ = xi_sum / row_sums

        # Emission parameters
        gamma_sum = gamma.sum(axis=0)  # (K,)
        gamma_sum[gamma_sum == 0] = 1e-10

        for k in range(K):
            wk = gamma[:, k]
            self.means_[k] = np.dot(wk, obs) / gamma_sum[k]
            diff = obs - self.means_[k]
            self.variances_[k] = np.dot(wk, diff ** 2) / gamma_sum[k]
            # Floor variance to prevent degenerate states
            self.variances_[k] = max(self.variances_[k], 1e-10)

        if self.tied_covariance:
            # Shared variance across all states
            shared_var = np.dot(gamma_sum, self.variances_) / gamma_sum.sum()
            self.variances_[:] = max(shared_var, 1e-10)

    # ── Initialization ─────────────────────────────────────────────

    def _init_params(self, obs: np.ndarray, rng: np.random.Generator) -> None:
        """Initialize parameters using quantile-based seeding."""
        K = self.n_states
        T = len(obs)

        # Means: spread across quantiles of the data
        quantiles = np.linspace(0, 100, K + 2)[1:-1]
        base_means = np.percentile(obs, quantiles)
        # Add noise
        self.means_ = base_means + rng.normal(0, np.std(obs) * 0.1, K)

        # Sort means descending (state 0 = highest mean = bull)
        order = np.argsort(self.means_)[::-1]
        self.means_ = self.means_[order]

        # Variances: initialize from data, scaled
        global_var = np.var(obs)
        self.variances_ = np.full(K, global_var) * (1 + rng.uniform(-0.3, 0.3, K))
        self.variances_ = np.clip(self.variances_, 1e-10, None)

        if self.tied_covariance:
            self.variances_[:] = global_var

        # Transition matrix: slightly sticky diagonal
        self.transmat_ = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(self.transmat_, 0.9)
        # Normalize rows
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

        # Start probabilities: uniform
        self.startprob_ = np.ones(K) / K

    # ── fit() ──────────────────────────────────────────────────────

    def fit(self, obs: np.ndarray) -> "GaussianHMM":
        """
        Fit the HMM to observed sequence using Baum-Welch (EM).

        Parameters
        ----------
        obs : np.ndarray, shape (T,)
            Observed sequence (e.g. weekly log returns).

        Returns
        -------
        self
        """
        obs = np.asarray(obs, dtype=np.float64).ravel()
        rng = np.random.default_rng(self.seed)

        best_ll = -np.inf
        best_params = None

        for restart in range(self.n_restarts):
            self._init_params(obs, rng)
            prev_ll = -np.inf

            for iteration in range(self.max_iter):
                log_B = self._compute_log_emission(obs)
                gamma, xi, log_ll = self._e_step(obs, log_B)

                if np.isnan(log_ll) or np.isinf(log_ll):
                    break

                self._m_step(obs, gamma, xi)

                if abs(log_ll - prev_ll) < self.tol:
                    break
                prev_ll = log_ll

            if log_ll > best_ll and not np.isnan(log_ll):
                best_ll = log_ll
                best_params = {
                    "means": self.means_.copy(),
                    "variances": self.variances_.copy(),
                    "transmat": self.transmat_.copy(),
                    "startprob": self.startprob_.copy(),
                    "n_iter": iteration + 1,
                    "converged": abs(log_ll - prev_ll) < self.tol,
                }

        if best_params is None:
            raise RuntimeError("HMM fitting failed on all restarts")

        self.means_ = best_params["means"]
        self.variances_ = best_params["variances"]
        self.transmat_ = best_params["transmat"]
        self.startprob_ = best_params["startprob"]
        self.log_likelihood_ = best_ll
        self.n_iter_ = best_params["n_iter"]
        self.converged_ = best_params["converged"]

        return self

    # ── Decoding (Viterbi) ─────────────────────────────────────────

    def decode(self, obs: np.ndarray) -> np.ndarray:
        """
        Find most likely state sequence via Viterbi algorithm.

        Parameters
        ----------
        obs : np.ndarray, shape (T,)

        Returns
        -------
        states : np.ndarray, shape (T,), dtype int
        """
        obs = np.asarray(obs, dtype=np.float64).ravel()
        T = len(obs)
        K = self.n_states
        log_B = self._compute_log_emission(obs)
        log_A = np.log(self.transmat_ + 1e-300)
        log_pi = np.log(self.startprob_ + 1e-300)

        # Viterbi
        V = np.empty((T, K))
        backptr = np.empty((T, K), dtype=int)

        V[0] = log_pi + log_B[0]
        for t in range(1, T):
            for k in range(K):
                scores = V[t - 1] + log_A[:, k]
                backptr[t, k] = np.argmax(scores)
                V[t, k] = scores[backptr[t, k]] + log_B[t, k]

        # Backtrace
        states = np.empty(T, dtype=int)
        states[-1] = np.argmax(V[-1])
        for t in range(T - 2, -1, -1):
            states[t] = backptr[t + 1, states[t + 1]]

        return states

    # ── Smoothed posteriors ────────────────────────────────────────

    def predict_proba(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute smoothed state probabilities P(z_t=k | all obs).

        Parameters
        ----------
        obs : np.ndarray, shape (T,)

        Returns
        -------
        gamma : np.ndarray, shape (T, K)
        """
        obs = np.asarray(obs, dtype=np.float64).ravel()
        log_B = self._compute_log_emission(obs)
        log_alpha, _ = self._forward(log_B)
        log_beta = self._backward(log_B)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    # ── BIC for model selection ────────────────────────────────────

    def bic(self, obs: np.ndarray) -> float:
        """Bayesian Information Criterion. Lower is better."""
        T = len(obs)
        K = self.n_states
        # Number of free parameters
        n_params = K - 1  # start probs
        n_params += K * (K - 1)  # transition matrix
        n_params += K  # means
        n_params += 1 if self.tied_covariance else K  # variances
        return -2 * self.log_likelihood_ + n_params * np.log(T)
