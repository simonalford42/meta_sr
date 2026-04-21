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
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that Julia loads the expected SymbolicRegression.jl checkout.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Repo root containing SymbolicRegression.jl (default: this repo root).",
    )
    parser.add_argument(
        "--expected-symbolicregression-root",
        type=str,
        default=None,
        help="Expected SymbolicRegression.jl root to be loaded (default: <repo-root>/SymbolicRegression.jl).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parent.parent
    juliapkg_project = (repo_root / ".juliapkg_env").resolve()
    expected_sr_root = (
        Path(args.expected_symbolicregression_root).resolve()
        if args.expected_symbolicregression_root
        else (repo_root / "SymbolicRegression.jl").resolve()
    )
    expected_source = (expected_sr_root / "src" / "SymbolicRegression.jl").resolve()

    # Prefer PYTHON_JULIAPKG_EXE (set when using juliaup), then the legacy
    # pyjuliapkg path, then fall back to the system julia.
    juliapkg_exe = os.environ.get("PYTHON_JULIAPKG_EXE")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    legacy_conda_julia_env = Path(conda_prefix).resolve() / "julia_env" if conda_prefix else None
    legacy_conda_exe = (
        legacy_conda_julia_env / "pyjuliapkg" / "install" / "bin" / "julia"
        if legacy_conda_julia_env
        else None
    )

    if juliapkg_exe and Path(juliapkg_exe).exists():
        julia_exe = juliapkg_exe
    elif legacy_conda_exe and legacy_conda_exe.exists():
        julia_exe = str(legacy_conda_exe)
    else:
        julia_exe = "julia"

    env = os.environ.copy()
    env["JULIA_PROJECT"] = str(juliapkg_project)
    env["PYTHON_JULIAPKG_PROJECT"] = str(juliapkg_project)
    env["SYMBOLICREGRESSION_DEBUG_IMPORT"] = "1"

    julia_code = (
        "using SymbolicRegression; "
        'println("SR_PATH=" * pathof(SymbolicRegression)); '
        'println("JULIA_PROJECT_ACTIVE=" * Base.active_project()); '
        'println("PYTHON_JULIAPKG_PROJECT_ENV=" * get(ENV, "PYTHON_JULIAPKG_PROJECT", "")); '
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
