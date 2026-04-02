"""
Parameter search — Sobol quasi-random sampling + Bayesian refinement via Optuna.

Phase 1 (Sobol): Fill parameter space evenly with low-discrepancy sequences.
Phase 2 (TPE): Drill into promising regions using Tree-structured Parzen Estimator.
"""
import optuna
from optuna.samplers import TPESampler, QMCSampler
import numpy as np
from typing import Callable, Any

from models.base import BaseModel


# Suppress Optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ParameterSearch:
    """
    Two-phase parameter optimization using Optuna.

    Parameters
    ----------
    model : BaseModel
        Model to optimize (must define get_param_space()).
    objective_fn : callable
        Function that takes a fitted model and returns a score to maximize.
        Signature: (model: BaseModel) -> float
    n_trials : int
        Total trial budget for this model.
    search_phase : str
        "sobol_only" or "sobol_then_tpe".
    sobol_ratio : float
        Fraction of trials allocated to Sobol phase.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        model: BaseModel,
        objective_fn: Callable[[BaseModel], float],
        n_trials: int = 30,
        search_phase: str = "sobol_only",
        sobol_ratio: float = 0.4,
        seed: int = 42,
    ):
        self.model = model
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.search_phase = search_phase
        self.sobol_ratio = sobol_ratio
        self.seed = seed
        self.param_space = model.get_param_space()
        self.study: optuna.Study | None = None

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest parameter values from the model's search space."""
        params = {}
        for name, spec in self.param_space.items():
            if spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        return params

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: set params, evaluate, return score."""
        params = self._suggest_params(trial)
        self.model.set_params(**params)

        try:
            score = self.objective_fn(self.model)
        except Exception as e:
            # Mark trial as failed but don't crash the study
            trial.set_user_attr("error", str(e))
            return 0.0

        return score

    def run(self) -> dict:
        """
        Execute the parameter search.

        Returns
        -------
        dict
            best_params: dict of optimal parameters
            best_score: float
            n_trials_completed: int
            all_trials: list of trial summaries
        """
        if self.search_phase == "sobol_then_tpe":
            n_startup = max(1, int(self.n_trials * self.sobol_ratio))
            sampler = TPESampler(
                n_startup_trials=n_startup,
                seed=self.seed,
            )
        else:
            # Sobol only — use QMC sampler for all trials
            sampler = QMCSampler(seed=self.seed)

        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        # Collect results
        all_trials = []
        for t in self.study.trials:
            all_trials.append({
                "number": t.number,
                "params": t.params,
                "score": t.value if t.value is not None else 0.0,
                "state": str(t.state),
            })

        return {
            "best_params": self.study.best_params,
            "best_score": self.study.best_value,
            "n_trials_completed": len(self.study.trials),
            "all_trials": all_trials,
        }
