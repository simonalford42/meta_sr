"""Smoke-test the driver-side fallback in BOTH validators (fullsr + pysr):
_ensure_symbolicregression_loaded survives the shared-project corruption that
broke runs/689916. Restores Project.toml on exit."""
from pathlib import Path
from julia_env import configure_juliapkg_project

repo = Path(__file__).resolve().parent.parent
configure_juliapkg_project(repo)
from juliacall import Main as jl
import skeleton_operator_types as sot
import operator_types as ot

# Happy-path loads.
sot._ensure_symbolicregression_loaded(jl)
ot._ensure_symbolicregression_loaded(jl, "CustomLossModule")
print("[ok] initial loads (fullsr + pysr helpers)")

proj = Path(repo, ".juliapkg_env", "Project.toml")
original = proj.read_text()
try:
    proj.write_text(
        "\n".join(ln for ln in original.splitlines() if "SymbolicRegression" not in ln)
    )
    jl.seval(f'import Pkg; Pkg.activate("{proj.parent}"; io=devnull)')
    print("[corrupt] Project.toml stripped of SymbolicRegression")

    sot._ensure_symbolicregression_loaded(jl)
    print("[PASS] fullsr helper survived corruption")

    for mod in ("CustomMutationsModule", "CustomSurvivalModule",
                "CustomSelectionModule", "CustomLossModule"):
        ot._ensure_symbolicregression_loaded(jl, mod)
    print("[PASS] pysr helper survived corruption (all 4 submodules)")
finally:
    proj.write_text(original)
    print("[cleanup] restored Project.toml")
