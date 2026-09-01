import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bundle_loader import load_bundle
from hpo_pysr import evaluate_baseline, evaluate_param_configs_batch
from operator_types import JuliaOperator, OperatorBundle


class CapturingEvaluator:
    def __init__(self):
        self.configs = None

    def evaluate_configs(self, configs, *args, **kwargs):
        self.configs = configs
        return [(0.5, [0.5], []) for _ in configs]


def _evolved_bundle():
    return OperatorBundle(
        operators={
            "mutation": JuliaOperator("custom_mut", "function custom_mut()\nend", weight=0.7),
            "selection": JuliaOperator("custom_sel", "function custom_sel()\nend"),
            "survival": JuliaOperator("custom_surv", "function custom_surv()\nend"),
            "loss": JuliaOperator("custom_loss", "function custom_loss()\nend"),
        },
        best_hparams={"population_size": 17, "weight_add_node": 1.25},
    )


def test_hpo_trials_keep_evolved_operators_and_override_tuned_params():
    evaluator = CapturingEvaluator()
    bundle = _evolved_bundle()

    evaluate_param_configs_batch(
        [{"population_size": 31, "weight_add_node": 2.5}],
        [4], evaluator, ["task"], {"maxsize": 40}, 42, 1, "gt",
        base_bundle=bundle,
    )

    config = evaluator.configs[0]
    assert config.pysr_kwargs["maxsize"] == 40
    assert config.pysr_kwargs["population_size"] == 31
    assert config.mutation_weights["weight_add_node"] == 2.5
    assert config.mutation_weights["weight_custom_mutation_1"] == 0.7
    assert config.custom_mutation_code == {"custom_mut": bundle.operators["mutation"].code}
    assert config.custom_selection_code == bundle.operators["selection"].code
    assert config.custom_survival_code == bundle.operators["survival"].code
    assert config.custom_loss_code == bundle.operators["loss"].code


def test_hpo_baseline_is_the_evolved_bundle():
    evaluator = CapturingEvaluator()
    bundle = _evolved_bundle()

    evaluate_baseline(
        evaluator, ["task"], {"maxsize": 40}, 42, 1, "gt",
        base_bundle=bundle,
    )

    config = evaluator.configs[0]
    assert config.pysr_kwargs["population_size"] == 17
    assert config.custom_selection_code == bundle.operators["selection"].code


def test_best_params_round_trip_embedded_base_bundle(tmp_path):
    bundle = _evolved_bundle()
    path = tmp_path / "best_params.json"
    path.write_text(json.dumps({
        "params": {"population_size": 31},
        "avg_score": 0.8,
        "avg_r2": 0.8,
        "base_bundle": bundle.to_dict(),
    }))

    loaded = load_bundle(str(path))

    assert loaded.display_name == bundle.display_name
    assert loaded.best_hparams == {"population_size": 31}
    assert loaded.score == 0.8
