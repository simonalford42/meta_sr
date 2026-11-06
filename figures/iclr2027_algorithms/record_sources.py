"""Record exact source provenance for the pseudocode figures.

Refresh the bundle selection from the full (large) run log with:
  python figures/iclr2027_algorithms/record_sources.py --run-json runs/709715/run_data.json
The default mode checks the saved operator snapshots against their original files.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OPERATOR_PATHS = {
    "mutation": "runs/709715/operators/gen27_mutation9.jl",
    "survival": "runs/709715/operators/gen28_survival8.jl",
    "selection": "runs/709715/operators/gen45_selection9.jl",
    "loss": "runs/709715/operators/gen39_loss7.jl",
}
ENGINE_PATHS = [
    "evolve_pysr.py", "evolution_helpers.py", "bundle_loader.py", "operator_types.py",
    "skeleton_operator_types.py", "evolve_fullsr.py",
    *["SymbolicRegression.jl/src/" + name for name in (
        "RegularizedEvolution.jl", "Mutate.jl", "SingleIteration.jl", "Population.jl",
        "LossFunctions.jl", "SymbolicRegression.jl", "SkeletonSR.jl", "BasicSRConfig.jl", "SRConfig.jl",
        "CustomSelection.jl", "CustomSurvival.jl", "CustomMutations.jl", "CustomLoss.jl",
    )],
]


def code_loc(source):
    # Load the repository's LOC definition without importing its runtime dependencies.
    module = ast.parse((REPO / "evolution_helpers.py").read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "code_loc")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<repository code_loc>", "exec"), namespace)
    return namespace["code_loc"](source)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh(data):
    population = data["generations"][-1]["population"]
    def loc(bundle):
        return sum(code_loc(op["code"]) for op in bundle["operators"].values() if op)
    ranked = sorted(population, key=loc)
    selected = ranked[0]
    manifest = {
        "run": "709715",
        "selection_rule": "Minimum repository code_loc in the last logged generation's population",
        "generation": data["generations"][-1]["generation"],
        "total_code_loc": loc(selected),
        "logged_training_score": selected["score"],
        "seeds_evaluated_at_log": selected.get("seeds_evaluated"),
        "final_population_code_locs": [loc(b) for b in ranked],
        "operators": {},
        "run_configuration": data["config"],
    }
    manifest["run_configuration"].pop("repo_root", None)
    for slot, original in OPERATOR_PATHS.items():
        op = selected["operators"][slot]
        source = REPO / original
        assert source.read_text().strip() == op["code"].strip(), (slot, "source mismatch")
        snapshot = ROOT / "source" / f"{slot}.jl"
        snapshot.write_bytes(source.read_bytes())
        manifest["operators"][slot] = {
            "name": op["name"], "original": original,
            "snapshot": str(snapshot.relative_to(ROOT)),
            "code_loc": code_loc(op["code"]), "sha256": sha256(snapshot),
        }
    manifest["engine_sources"] = {p: sha256(REPO / p) for p in ENGINE_PATHS}
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for slot, record in manifest["operators"].items():
        assert sha256(ROOT / record["snapshot"]) == record["sha256"], slot
        assert sha256(REPO / record["original"]) == record["sha256"], slot
        assert code_loc((ROOT / record["snapshot"]).read_text()) == record["code_loc"], slot
    assert sum(op["code_loc"] for op in manifest["operators"].values()) == 245
    changed = [p for p, digest in manifest["engine_sources"].items() if sha256(REPO / p) != digest]
    print("Verified four operator snapshots; total: 245 code lines.")
    if changed:
        print("Engine sources changed since transcription; review before refreshing:", changed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path)
    args = parser.parse_args()
    if args.run_json:
        refresh(json.loads(args.run_json.read_bytes()))
    check()
