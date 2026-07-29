import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bundle_loader import load_bundle, load_skeleton_bundle
from skeleton_operator_types import SkeletonBundle
from srbench_eval_source import detect_evolve_backend, load_evaluation_source


def _args(path):
    return SimpleNamespace(
        evolve_results=str(path),
        hpo_results=None,
        select_by="val",
        max_evals=12345,
    )


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


def test_skeleton_bundle_round_trips_full_file_module_body():
    bundle = SkeletonBundle.from_default_sr_config()
    bundle.raw_module_body = "# complete evolved module body"

    restored = SkeletonBundle.from_dict(bundle.to_dict())

    assert restored.raw_module_body == bundle.raw_module_body
