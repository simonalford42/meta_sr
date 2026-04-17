#!/usr/bin/env python3
"""
Regression test for PySR negative caching policy.

Deterministic operator/runtime failures should be cached so identical reruns
can skip them. Transient infrastructure failures should remain uncached.
"""

import tempfile
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation_cache import get_pysr_cache, set_pysr_cache_path
from parallel_eval_pysr import (
    PySRTaskResult,
    PySRTaskSpec,
    _classify_pysr_error,
    _has_usable_pysr_cached_result,
    _store_pysr_result_in_cache,
)


def _make_task(run_index: int = 0) -> PySRTaskSpec:
    return PySRTaskSpec(
        config_id=7,
        dataset_name="feynman_II_11_27",
        pysr_kwargs={"niterations": 10, "population_size": 5},
        mutation_weights={"weight_add_node": 1.0},
        seed=42,
        data_seed=42,
        max_samples=100,
        run_index=run_index,
        fitness_metric="r2",
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "pysr_test_cache.db"
        set_pysr_cache_path(str(cache_path))

        task = _make_task(run_index=0)
        pysr_mutation_kwargs = {"weight_add_node": 1.0}
        model_kwargs = {**pysr_mutation_kwargs, **task.pysr_kwargs, "random_state": 42}

        deterministic_error = "Error: UndefVarError: `std` not defined"
        transient_error = "Error: Illegal instruction on worker node"

        assert _classify_pysr_error(deterministic_error) == "deterministic"
        assert _classify_pysr_error(transient_error) == "transient"
        assert _classify_pysr_error("Error: something unexpected") == "unknown"

        deterministic_result = PySRTaskResult(
            config_id=task.config_id,
            dataset_name=task.dataset_name,
            r2_score=-1.0,
            best_equation=None,
            best_loss=float("inf"),
            gt_match_score=None,
            error=deterministic_error,
            run_index=task.run_index,
            runtime_seconds=12.5,
        )
        _store_pysr_result_in_cache(task, pysr_mutation_kwargs, model_kwargs, deterministic_result)

        cache = get_pysr_cache()
        assert cache is not None
        cached = cache.lookup(
            mutation_weights=pysr_mutation_kwargs,
            pysr_kwargs=task.pysr_kwargs,
            dataset_name=task.dataset_name,
            seed=task.seed,
            data_seed=task.data_seed,
            max_samples=task.max_samples,
            run_index=task.run_index,
            custom_mutation_code=task.custom_mutation_code,
            allow_custom_mutations=task.allow_custom_mutations,
            pysr_model_kwargs=model_kwargs,
            target_noise=task.target_noise,
            custom_selection_code=task.custom_selection_code,
            custom_survival_code=task.custom_survival_code,
        )
        assert _has_usable_pysr_cached_result(cached)
        assert cached["error"] == deterministic_error
        assert cached["gt_match_score"] == 0.0

        transient_task = _make_task(run_index=1)
        transient_result = PySRTaskResult(
            config_id=transient_task.config_id,
            dataset_name=transient_task.dataset_name,
            r2_score=-1.0,
            best_equation=None,
            best_loss=float("inf"),
            gt_match_score=None,
            error=transient_error,
            run_index=transient_task.run_index,
            runtime_seconds=1.0,
        )
        transient_model_kwargs = {**model_kwargs, "random_state": 43}
        _store_pysr_result_in_cache(
            transient_task, pysr_mutation_kwargs, transient_model_kwargs, transient_result
        )

        uncached = cache.lookup(
            mutation_weights=pysr_mutation_kwargs,
            pysr_kwargs=transient_task.pysr_kwargs,
            dataset_name=transient_task.dataset_name,
            seed=transient_task.seed,
            data_seed=transient_task.data_seed,
            max_samples=transient_task.max_samples,
            run_index=transient_task.run_index,
            custom_mutation_code=transient_task.custom_mutation_code,
            allow_custom_mutations=transient_task.allow_custom_mutations,
            pysr_model_kwargs=transient_model_kwargs,
            target_noise=transient_task.target_noise,
            custom_selection_code=transient_task.custom_selection_code,
            custom_survival_code=transient_task.custom_survival_code,
        )
        assert uncached is None

        print("PySR negative caching policy OK")


if __name__ == "__main__":
    main()
