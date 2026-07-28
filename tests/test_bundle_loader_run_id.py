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
