#!/usr/bin/env python3
"""
Verify that Julia is loading the local SymbolicRegression.jl checkout.

This script sets a debug marker env var, imports SymbolicRegression in Julia,
and checks that the marker/path correspond to this repository.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    local_project = (repo_root / "SymbolicRegression.jl").resolve()
    expected_source = (local_project / "src" / "SymbolicRegression.jl").resolve()

    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_julia_env = Path(conda_prefix).resolve() / "julia_env" if conda_prefix else None
    conda_julia_exe = (
        conda_julia_env / "pyjuliapkg" / "install" / "bin" / "julia"
        if conda_julia_env
        else None
    )
    use_conda_julia = bool(conda_julia_env and conda_julia_exe and conda_julia_exe.exists())

    if use_conda_julia:
        julia_exe = str(conda_julia_exe)
        julia_project = str(conda_julia_env)
    else:
        julia_exe = "julia"
        julia_project = str(local_project)

    env = os.environ.copy()
    env.setdefault("JULIA_PROJECT", julia_project)
    env.setdefault("PYTHON_JULIAPKG_PROJECT", julia_project)
    env["SYMBOLICREGRESSION_DEBUG_IMPORT"] = "1"

    julia_code = (
        "using SymbolicRegression; "
        'println("SR_PATH=" * pathof(SymbolicRegression)); '
        'println("HAS_CUSTOM_SELECTION=" * string(isdefined(SymbolicRegression, :CustomSelectionModule))); '
        'println("HAS_CUSTOM_SURVIVAL=" * string(isdefined(SymbolicRegression, :CustomSurvivalModule)));'
    )

    proc = subprocess.run(
        [julia_exe, "-e", julia_code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )

    output = (proc.stdout or "") + (proc.stderr or "")
    print(output, end="")

    if proc.returncode != 0:
        print(f"\nFAIL: Julia command exited with code {proc.returncode}", file=sys.stderr)
        return proc.returncode

    if "SYMBOLICREGRESSION_LOCAL_MARKER" not in output:
        print("\nFAIL: Marker print not found; __init__ marker did not fire.", file=sys.stderr)
        return 1

    expected_line = f"SR_PATH={expected_source}"
    if expected_line not in output:
        print(
            "\nFAIL: SymbolicRegression path is not the local checkout.\n"
            f"Expected line: {expected_line}",
            file=sys.stderr,
        )
        return 1

    if "HAS_CUSTOM_SELECTION=true" not in output or "HAS_CUSTOM_SURVIVAL=true" not in output:
        print(
            "\nFAIL: Expected custom modules are missing in loaded SymbolicRegression.",
            file=sys.stderr,
        )
        return 1

    print("\nPASS: Local SymbolicRegression.jl was loaded (marker/path/modules confirmed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
