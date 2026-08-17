import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bundle_loader import load_bundle, load_skeleton_bundle
from skeleton_operator_types import SkeletonBundle
from srbench_eval_source import (
    DEFAULT_SOFT_TIMEOUT,
    apply_soft_timeout,
    detect_evolve_backend,
    load_evaluation_source,
    saved_run_soft_timeout,
    scale_soft_timeout,
)


def _args(path, **overrides):
    args = SimpleNamespace(
        evolve_results=str(path),
        hpo_results=None,
        select_by="val",
        max_evals=12345,
        timeout=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _write_fullsr_run(tmp_path, config=None):
    bundle = SkeletonBundle.from_default_sr_config()
    bundle.score = 0.4
    run_data = {"best_bundle": bundle.to_dict(), "generations": []}
    if config is not None:
        run_data["config"] = config
    (tmp_path / "run_data.json").write_text(json.dumps(run_data))
    return tmp_path


def test_fullsr_run_is_detected_and_rejected_by_pysr_loader(tmp_path):
    bundle = SkeletonBundle.from_default_sr_config()
    bundle.score = 0.4
    run_data = {
        "generations": [{"population": [bundle.to_dict()], "offspring": []}],
        "best_bundle": bundle.to_dict(),
    }
    (tmp_path / "run_data.json").write_text(json.dumps(run_data))

    assert detect_evolve_backend(str(tmp_path)) == "fullsr"
    with pytest.raises(ValueError, match="evolve_fullsr"):
        load_bundle(str(tmp_path))


def test_fullsr_loader_selects_content_hash_keyed_validation_best(tmp_path):
    from bundle_loader import _skeleton_content_key

    low = SkeletonBundle.from_default_sr_config()
    low.score = 0.8
    high = SkeletonBundle.from_default_sr_config()
    high.functions["selection"].name = "validation_winner"
    high.functions["selection"].code = (
        high.functions["selection"].code + "\n# validation winner variant"
    )
    high.score = 0.3
    run_data = {
        "generations": [
            {"population": [low.to_dict(), high.to_dict()], "offspring": []}
        ],
        "best_bundle": low.to_dict(),
        "val_results": {
            _skeleton_content_key(low): {"val": {"avg_score": 0.2}},
            _skeleton_content_key(high): {"val": {"avg_score": 0.9}},
        },
    }
    (tmp_path / "run_data.json").write_text(json.dumps(run_data))

    selected = load_skeleton_bundle(str(tmp_path), select_by="val")

    assert selected.functions["selection"].name == "validation_winner"
    assert selected.score == 0.3
    assert selected.val_score == 0.9


def test_backend_aware_loader_builds_fullsr_config(tmp_path):
    bundle = SkeletonBundle.from_default_sr_config()
    bundle.score = 0.4
    (tmp_path / "run_data.json").write_text(
        json.dumps({"best_bundle": bundle.to_dict(), "generations": []})
    )

    loaded = load_evaluation_source(_args(tmp_path))

    assert loaded.backend == "fullsr"
    assert loaded.mode == "evolve_fullsr"
    assert loaded.config.policy_module_code
    assert loaded.config.engine_kwargs["max_evals"] == 12345
    assert len(loaded.method_meta["functions"]) == 8


def test_fullsr_baseline_uses_basic_sr_config():
    args = SimpleNamespace(
        evolve_results=None,
        hpo_results=None,
        fullsr_baseline=True,
        select_by="val",
        max_evals=12345,
    )

    loaded = load_evaluation_source(args)

    assert loaded.backend == "fullsr"
    assert loaded.mode == "fullsr_baseline"
    assert loaded.config.policy_name == "basic"
    assert loaded.config.policy_code is None
    assert loaded.config.policy_module_code is None
    assert loaded.config.engine_kwargs["max_evals"] == 12345


def test_soft_timeout_is_inherited_from_the_training_run(tmp_path):
    run = _write_fullsr_run(tmp_path, config={"timeout": 500, "max_evals": 1_000_000})

    loaded = load_evaluation_source(_args(run))

    assert loaded.soft_timeout == 500
    assert str(run) in loaded.soft_timeout_source
    assert loaded.config.engine_kwargs["timeout_in_seconds"] == 500


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"timeout": 500}, 500),                                    # evolve_fullsr
        ({"pysr_kwargs": {"timeout_in_seconds": 450}}, 450),        # evolve_pysr
        ({"base_pysr_kwargs": {"timeout_in_seconds": 300}}, 300),   # hpo_pysr
        ({"max_evals": 1_000_000}, None),                           # predates the key
    ],
)
def test_saved_run_soft_timeout_reads_each_writer_schema(tmp_path, config, expected):
    (tmp_path / "run_data.json").write_text(json.dumps({"config": config}))

    assert saved_run_soft_timeout(str(tmp_path)) == expected


def test_soft_timeout_falls_back_to_default_without_a_training_value(tmp_path):
    run = _write_fullsr_run(tmp_path)

    loaded = load_evaluation_source(_args(run))

    assert loaded.soft_timeout == DEFAULT_SOFT_TIMEOUT
    assert loaded.soft_timeout_source == "default"


def test_explicit_timeout_overrides_the_training_run(tmp_path):
    run = _write_fullsr_run(tmp_path, config={"timeout": 500})

    loaded = load_evaluation_source(_args(run, timeout=420))

    assert loaded.soft_timeout == 420
    assert loaded.config.engine_kwargs["timeout_in_seconds"] == 420


def test_timeout_zero_restores_an_unbounded_search(tmp_path):
    run = _write_fullsr_run(tmp_path, config={"timeout": 500})

    loaded = load_evaluation_source(_args(run, timeout=0))

    assert loaded.soft_timeout is None
    assert "timeout_in_seconds" not in loaded.config.engine_kwargs


def test_apply_soft_timeout_leaves_the_source_config_alone(tmp_path):
    run = _write_fullsr_run(tmp_path, config={"timeout": 500})
    loaded = load_evaluation_source(_args(run))

    black_box = apply_soft_timeout(loaded.config, loaded.backend, 1500)

    assert black_box.engine_kwargs["timeout_in_seconds"] == 1500
    assert loaded.config.engine_kwargs["timeout_in_seconds"] == 500


def test_scale_soft_timeout_preserves_the_soft_to_hard_ratio():
    # The ground-truth 500s/600s pair maps onto the black-box 1800s wall as the
    # 1500s evolve_fullsr.py uses for its own validation evaluations.
    assert scale_soft_timeout(500, 600, 1800) == 1500
    assert scale_soft_timeout(None, 600, 1800) is None
    assert scale_soft_timeout(500, 0, 1800) == 500


def test_skeleton_bundle_round_trips_full_file_module_body():
    bundle = SkeletonBundle.from_default_sr_config()
    bundle.raw_module_body = "# complete evolved module body"

    restored = SkeletonBundle.from_dict(bundle.to_dict())

    assert restored.raw_module_body == bundle.raw_module_body
