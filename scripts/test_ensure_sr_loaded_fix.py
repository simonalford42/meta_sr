"""Smoke-test the driver-side fix: skeleton_operator_types._ensure_symbolicregression_loaded
survives the shared-project corruption that broke runs/689916."""
from pathlib import Path
from julia_env import configure_juliapkg_project

repo = Path(__file__).resolve().parent.parent
configure_juliapkg_project(repo)
from juliacall import Main as jl
import skeleton_operator_types as sot

# Initial load (happy path — absolute form).
sot._ensure_symbolicregression_loaded(jl)
print("[ok] initial load via _ensure_symbolicregression_loaded")

proj = Path(repo, ".juliapkg_env", "Project.toml")
original = proj.read_text()
try:
    proj.write_text(
        "\n".join(ln for ln in original.splitlines() if "SymbolicRegression" not in ln)
    )
    jl.seval(f'import Pkg; Pkg.activate("{proj.parent}"; io=devnull)')
    # Absolute form now fails; the helper must fall back to relative and succeed.
    try:
        jl.seval("using SymbolicRegression")
        print("[note] absolute using unexpectedly still worked (mtime cache)")
    except Exception as e:
        print(f"[expected] absolute using fails: {str(e).splitlines()[0][:70]}")
    sot._ensure_symbolicregression_loaded(jl)
    print("[PASS] _ensure_symbolicregression_loaded survived corrupted project")
finally:
    proj.write_text(original)
    print("[cleanup] restored Project.toml")
