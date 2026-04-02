"""
Univariate Sensitivity Analysis with Joint Confirmation.

Custom optimization strategy for models like Regime Block Bootstrap where
parameters group into logical categories (block construction vs regime switching).

Phase 1 — Sweep one parameter group (block length) with the other at defaults.
Phase 2 — Lock Phase 1 winner, sweep the other group (regime switching).
Phase 3 — Joint confirmation: top 3 from each phase combined (9 combos).

This avoids the curse of dimensionality from grid/random search across
all parameters simultaneously, while still validating interactions.
"""
import numpy as np
from typing import Any, Callable
from rich.console import Console
from rich.table import Table

from models.base import BaseModel


console = Console()


class SensitivitySearch:
    """
    Univariate sensitivity analysis with joint confirmation.

    Parameters
    ----------
    model : BaseModel
        Model to optimize.
    objective_fn : callable
        Function: (model: BaseModel) -> float. Score to maximize.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        model: BaseModel,
        objective_fn: Callable[[BaseModel], float],
        seed: int = 42,
    ):
        self.model = model
        self.objective_fn = objective_fn
        self.seed = seed
        self.all_trials: list[dict] = []
        self._trial_counter = 0

    def _eval(self, params: dict, phase: str) -> float:
        """Set params, evaluate, record trial with per-metric breakdown."""
        self.model.set_params(**params)
        try:
            result = self.objective_fn(self.model)
            if isinstance(result, dict):
                score = result["composite"]
                metrics = result.get("metrics", {})
            else:
                score = float(result)
                metrics = {}
        except Exception as e:
            console.print(f"    [red]Trial {self._trial_counter} failed: {str(e)[:80]}[/red]")
            score = 0.0
            metrics = {}

        self.all_trials.append({
            "number": self._trial_counter,
            "params": {k: v for k, v in params.items()},
            "score": score,
            "metrics": metrics,
            "phase": phase,
            "state": "COMPLETE",
        })
        self._trial_counter += 1
        return score

    def run(self) -> dict:
        """
        Execute the 3-phase sensitivity search.

        Returns
        -------
        dict
            Same format as ParameterSearch.run() for compatibility.
        """
        defaults = self.model.get_default_params()
        self.all_trials = []
        self._trial_counter = 0

        # ══════════════════════════════════════════════════════════════
        # Phase 1: Pure Block Length Sweep (NO regime switching)
        # Isolates block length effect from regime classification.
        # ══════════════════════════════════════════════════════════════
        console.print("  [bold]Phase 1: Block Length Sweep (no regimes)[/bold]")
        phase1_results = []

        block_lengths = list(range(10, 101, 10))  # 10, 20, ..., 100

        for sampling in ["geometric", "fixed"]:
            for bl in block_lengths:
                params = {**defaults}
                params["block_length_sampling"] = sampling
                params["mean_block_length"] = bl
                params["regime_enabled"] = False  # No regime switching

                score = self._eval(params, "phase1_block_length")
                phase1_results.append({"params": params.copy(), "score": score})
                console.print(
                    f"    {sampling:10s} bl={bl:3d}  score={score:.4f}"
                )

        phase1_results.sort(key=lambda x: x["score"], reverse=True)
        top3_block = phase1_results[:3]

        console.print("\n  [bold]Phase 1 Top 3 (pure block, no regimes):[/bold]")
        for i, r in enumerate(top3_block):
            console.print(
                f"    #{i+1}: {r['params']['block_length_sampling']} "
                f"bl={r['params']['mean_block_length']}  "
                f"score={r['score']:.4f}"
            )

        # ══════════════════════════════════════════════════════════════
        # Phase 2: Regime Switching Sweep (block length locked from P1)
        # Tests whether regime conditioning improves over pure bootstrap.
        # ══════════════════════════════════════════════════════════════
        console.print("\n  [bold]Phase 2: Regime Switching Sweep (regimes ON)[/bold]")
        best_block = top3_block[0]["params"]
        phase2_results = []

        # First: baseline score with no regimes (Phase 1 winner)
        no_regime_score = top3_block[0]["score"]
        console.print(
            f"    Baseline (no regimes): {no_regime_score:.4f}"
        )

        for tm_method in ["fitted", "empirical"]:
            for var_switch in [True, False]:
                params = {**defaults}
                params["block_length_sampling"] = best_block["block_length_sampling"]
                params["mean_block_length"] = best_block["mean_block_length"]
                params["regime_enabled"] = True
                params["transition_matrix_method"] = tm_method
                params["msm_variance_switching"] = var_switch

                score = self._eval(params, "phase2_regime")
                delta = score - no_regime_score
                phase2_results.append({"params": params.copy(), "score": score})
                console.print(
                    f"    tm={tm_method:10s} var_switch={var_switch!s:5s}  "
                    f"score={score:.4f}  delta={delta:+.4f}"
                )

        phase2_results.sort(key=lambda x: x["score"], reverse=True)
        top3_regime = phase2_results[:3]

        # Check if regime switching helps at all
        best_regime_score = top3_regime[0]["score"]
        regime_helps = best_regime_score > no_regime_score
        console.print(
            f"\n  Regime switching {'[green]HELPS[/green]' if regime_helps else '[red]HURTS[/red]'}: "
            f"best regime={best_regime_score:.4f} vs no-regime={no_regime_score:.4f} "
            f"(delta={best_regime_score - no_regime_score:+.4f})"
        )

        console.print("\n  [bold]Phase 2 Top 3:[/bold]")
        for i, r in enumerate(top3_regime):
            console.print(
                f"    #{i+1}: tm={r['params']['transition_matrix_method']} "
                f"var_switch={r['params']['msm_variance_switching']}  "
                f"score={r['score']:.4f}"
            )

        # ══════════════════════════════════════════════════════════════
        # Phase 3: Joint Confirmation (top3 block × top3 regime = 9)
        # Includes no-regime configs if they scored in top 3.
        # ══════════════════════════════════════════════════════════════
        console.print("\n  [bold]Phase 3: Joint Confirmation (3x3=9)[/bold]")
        phase3_results = []

        # Include no-regime top3 and regime top3
        # For regime configs, combine with top block lengths
        tested = set()
        for br in top3_block:
            for rr in top3_regime:
                params = {**defaults}
                params["block_length_sampling"] = br["params"]["block_length_sampling"]
                params["mean_block_length"] = br["params"]["mean_block_length"]
                params["regime_enabled"] = True
                params["transition_matrix_method"] = rr["params"]["transition_matrix_method"]
                params["msm_variance_switching"] = rr["params"]["msm_variance_switching"]

                key = (
                    params["block_length_sampling"],
                    params["mean_block_length"],
                    params["regime_enabled"],
                    params["transition_matrix_method"],
                    params["msm_variance_switching"],
                )
                if key in tested:
                    for t in self.all_trials:
                        tp = t["params"]
                        if (tp.get("block_length_sampling") == key[0] and
                            tp.get("mean_block_length") == key[1] and
                            tp.get("regime_enabled") == key[2] and
                            tp.get("transition_matrix_method") == key[3] and
                            tp.get("msm_variance_switching") == key[4]):
                            phase3_results.append({"params": params.copy(), "score": t["score"]})
                            break
                    continue
                tested.add(key)

                score = self._eval(params, "phase3_joint")
                phase3_results.append({"params": params.copy(), "score": score})
                regime_tag = "regime" if params["regime_enabled"] else "no-regime"
                console.print(
                    f"    {params['block_length_sampling']:10s} "
                    f"bl={params['mean_block_length']:3d}  "
                    f"{regime_tag:10s} "
                    f"tm={params.get('transition_matrix_method','n/a'):10s} "
                    f"score={score:.4f}"
                )

        # Also include the no-regime top 3 block configs in joint
        for br in top3_block:
            params = {**defaults}
            params["block_length_sampling"] = br["params"]["block_length_sampling"]
            params["mean_block_length"] = br["params"]["mean_block_length"]
            params["regime_enabled"] = False

            key = (
                params["block_length_sampling"],
                params["mean_block_length"],
                False, "n/a", "n/a",
            )
            if key not in tested:
                tested.add(key)
                # Reuse Phase 1 score
                phase3_results.append({"params": params.copy(), "score": br["score"]})

        phase3_results.sort(key=lambda x: x["score"], reverse=True)
        best = phase3_results[0]

        console.print(f"\n  [bold green]Winner: score={best['score']:.4f}[/bold green]")
        console.print(f"    Params: {best['params']}")
        if best["params"].get("regime_enabled"):
            console.print(f"    Regime switching: ON")
        else:
            console.print(f"    Regime switching: OFF (pure block bootstrap)")

        return {
            "best_params": best["params"],
            "best_score": best["score"],
            "n_trials_completed": len(self.all_trials),
            "all_trials": self.all_trials,
            "phase1_top3": [
                {"params": r["params"], "score": r["score"]} for r in top3_block
            ],
            "phase2_top3": [
                {"params": r["params"], "score": r["score"]} for r in top3_regime
            ],
            "phase3_results": [
                {"params": r["params"], "score": r["score"]} for r in phase3_results
            ],
            "regime_helps": regime_helps,
            "no_regime_baseline": no_regime_score,
        }
