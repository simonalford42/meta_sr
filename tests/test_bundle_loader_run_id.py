import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bundle_loader
from operator_types import OperatorBundle


def test_load_bundle_resolves_bare_run_id(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "120458"
    run_dir.mkdir(parents=True)

    bundle = OperatorBundle.create_default()
    (run_dir / "run_data.json").write_text(
        json.dumps({"best_bundle": bundle.to_dict(), "generations": []})
    )
    monkeypatch.setattr(bundle_loader, "RUNS_ROOT", runs_root)

    loaded = bundle_loader.load_bundle("120458")

    assert loaded.operators == bundle.operators


def test_resume_counts_serialized_bundle_seeds_without_top_level_names(tmp_path):
    from operator_types import JuliaOperator

    first = OperatorBundle.create_default()
    first.seeds_evaluated = 3
    first.score = 0.5
    second = OperatorBundle.create_default()
    second.operators['mutation'] = JuliaOperator(
        name='new_mutation', code='function new_mutation() end',
    )
    second.seeds_evaluated = 3
    second.score = 0.6
    initial = first.to_dict()
    first.seeds_evaluated = 5
    checkpoint = {'generations': [
        {'generation': 0, 'population': [initial]},
        {'generation': 1, 'population': [first.to_dict(), second.to_dict()],
         'offspring': [second.to_dict()]},
    ]}
    (tmp_path / 'run_data.json').write_text(json.dumps(checkpoint))
    state = bundle_loader.load_resume_state(str(tmp_path))
    assert state['eval_idx'] == 8  # latest count per distinct bundle, once
    assert state['start_gen'] == 2
    assert len(state['population']) == 2
