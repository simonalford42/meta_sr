"""Assemble bundle.json (run_data format) from the .jl operators in this directory."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from operator_types import JuliaOperator, OperatorBundle  # noqa: E402

ops = {
    "loss": JuliaOperator(name="affine_scaled_mse_loss",
                          code=(HERE / "affine_scaled_mse_loss.jl").read_text()),
    "mutation": JuliaOperator(name="subtree_duplication",
                              code=(HERE / "subtree_duplication.jl").read_text(),
                              weight=0.5),
}
bundle = OperatorBundle(operators=ops)
(HERE / "bundle.json").write_text(json.dumps({"best_bundle": bundle.to_dict()}, indent=2) + "\n")
print(f"wrote {HERE / 'bundle.json'}")
