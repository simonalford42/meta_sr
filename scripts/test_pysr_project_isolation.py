#!/usr/bin/env python3
"""
Verify PySR/SymbolicRegression project isolation across two repo roots.

This script is intended for the "baseline repo" vs "agent-loop sandbox repo"
setup. It checks:

1. Each repo root individually loads its own SymbolicRegression.jl checkout.
2. Both verifications can run concurrently without loading the wrong checkout.
3. Generated PySR SLURM worker scripts export distinct JULIA_PROJECT and
   PYTHON_JULIAPKG_PROJECT values for the two repos.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parallel_eval_pysr import PySRSlurmEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test isolation between two meta_sr/PySR/SymbolicRegression repo roots.",
    )
    parser.add_argument("--baseline-root", type=str, required=True)
    parser.add_argument("--sandbox-root", type=str, required=True)
    parser.add_argument(
        "--baseline-pyjuliapkg-project",
        type=str,
        default=None,
        help="Optional PYTHON_JULIAPKG_PROJECT for the baseline repo.",
    )
    parser.add_argument(
        "--sandbox-pyjuliapkg-project",
        type=str,
        default=None,
        help="Optional PYTHON_JULIAPKG_PROJECT for the sandbox repo.",
    )
    parser.add_argument(
        "--julia-depot-path",
        type=str,
        default=None,
        help="Optional shared JULIA_DEPOT_PATH to use for both verifications.",
    )
    return parser.parse_args()


def _default_pyjuliapkg_project(repo_root: Path) -> Path:
    return repo_root / ".julia_env"


def run_verify(
    repo_root: Path,
    pyjuliapkg_project: Path | None,
    julia_depot_path: str | None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "verify_local_symbolicregression.py"),
        "--repo-root",
        str(repo_root),
        "--julia-project",
        str(repo_root / "SymbolicRegression.jl"),
    ]
    if pyjuliapkg_project is not None:
        cmd.extend(["--python-juliapkg-project", str(pyjuliapkg_project)])
    if julia_depot_path:
        cmd.extend(["--julia-depot-path", julia_depot_path])

    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def assert_verify_passed(label: str, proc: subprocess.CompletedProcess[str], expected_repo_root: Path) -> None:
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise AssertionError(f"{label} verification failed with exit code {proc.returncode}:\n{output}")
    expected_sr_path = (expected_repo_root / "SymbolicRegression.jl" / "src" / "SymbolicRegression.jl").resolve()
    expected_project = (expected_repo_root / "SymbolicRegression.jl").resolve()
    if f"SR_PATH={expected_sr_path}" not in output:
        raise AssertionError(f"{label} loaded the wrong backend path:\n{output}")
    if f"JULIA_PROJECT_ACTIVE={expected_project}" not in output:
        raise AssertionError(f"{label} activated the wrong JULIA_PROJECT:\n{output}")
    if "PASS: Local SymbolicRegression.jl was loaded" not in output:
        raise AssertionError(f"{label} did not report PASS:\n{output}")


def assert_job_script_isolated(
    baseline_root: Path,
    sandbox_root: Path,
    baseline_pyjuliapkg_project: Path,
    sandbox_pyjuliapkg_project: Path,
    julia_depot_path: str | None,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_eval = PySRSlurmEvaluator(
            results_dir=str(Path(tmpdir) / "baseline"),
            partition="test",
            use_cache=False,
            repo_root=str(baseline_root),
            julia_project=str((baseline_root / "SymbolicRegression.jl").resolve()),
            python_juliapkg_project=str(baseline_pyjuliapkg_project.resolve()),
            julia_depot_path=julia_depot_path,
        )
        sandbox_eval = PySRSlurmEvaluator(
            results_dir=str(Path(tmpdir) / "sandbox"),
            partition="test",
            use_cache=False,
            repo_root=str(sandbox_root),
            julia_project=str((sandbox_root / "SymbolicRegression.jl").resolve()),
            python_juliapkg_project=str(sandbox_pyjuliapkg_project.resolve()),
            julia_depot_path=julia_depot_path,
        )

        baseline_batch = baseline_eval._new_batch_dir()
        sandbox_batch = sandbox_eval._new_batch_dir()
        (baseline_batch / "tasks.json").write_text("[]")
        (sandbox_batch / "tasks.json").write_text("[]")

        baseline_script = baseline_eval._create_job_script(baseline_batch, n_tasks=1).read_text()
        sandbox_script = sandbox_eval._create_job_script(sandbox_batch, n_tasks=1).read_text()

        assert str(baseline_root.resolve()) in baseline_script
        assert str(sandbox_root.resolve()) in sandbox_script
        assert f'export JULIA_PROJECT="{(baseline_root / "SymbolicRegression.jl").resolve()}"' in baseline_script
        assert f'export JULIA_PROJECT="{(sandbox_root / "SymbolicRegression.jl").resolve()}"' in sandbox_script
        assert f'export PYTHON_JULIAPKG_PROJECT="{baseline_pyjuliapkg_project.resolve()}"' in baseline_script
        assert f'export PYTHON_JULIAPKG_PROJECT="{sandbox_pyjuliapkg_project.resolve()}"' in sandbox_script
        if julia_depot_path:
            assert f'export JULIA_DEPOT_PATH="{julia_depot_path}"' in baseline_script
            assert f'export JULIA_DEPOT_PATH="{julia_depot_path}"' in sandbox_script


def main() -> int:
    args = parse_args()
    baseline_root = Path(args.baseline_root).resolve()
    sandbox_root = Path(args.sandbox_root).resolve()

    baseline_pyjuliapkg_project = Path(args.baseline_pyjuliapkg_project).resolve() if args.baseline_pyjuliapkg_project else _default_pyjuliapkg_project(baseline_root).resolve()
    sandbox_pyjuliapkg_project = Path(args.sandbox_pyjuliapkg_project).resolve() if args.sandbox_pyjuliapkg_project else _default_pyjuliapkg_project(sandbox_root).resolve()

    print("=" * 80)
    print("PySR project isolation test")
    print("=" * 80)
    print(f"Baseline root:             {baseline_root}")
    print(f"Sandbox root:              {sandbox_root}")
    print(f"Baseline PYTHON_JULIAPKG_PROJECT: {baseline_pyjuliapkg_project}")
    print(f"Sandbox PYTHON_JULIAPKG_PROJECT:  {sandbox_pyjuliapkg_project}")
    print(f"Shared JULIA_DEPOT_PATH:   {args.julia_depot_path or '(default/shared)'}")

    print("\n[1/3] Sequential verification")
    baseline_proc = run_verify(baseline_root, baseline_pyjuliapkg_project, args.julia_depot_path)
    assert_verify_passed("baseline", baseline_proc, baseline_root)
    print("  baseline OK")

    sandbox_proc = run_verify(sandbox_root, sandbox_pyjuliapkg_project, args.julia_depot_path)
    assert_verify_passed("sandbox", sandbox_proc, sandbox_root)
    print("  sandbox OK")

    print("\n[2/3] Concurrent verification")
    baseline_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "verify_local_symbolicregression.py"),
        "--repo-root", str(baseline_root),
        "--julia-project", str(baseline_root / "SymbolicRegression.jl"),
        "--python-juliapkg-project", str(baseline_pyjuliapkg_project),
    ]
    sandbox_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "verify_local_symbolicregression.py"),
        "--repo-root", str(sandbox_root),
        "--julia-project", str(sandbox_root / "SymbolicRegression.jl"),
        "--python-juliapkg-project", str(sandbox_pyjuliapkg_project),
    ]
    if args.julia_depot_path:
        baseline_cmd.extend(["--julia-depot-path", args.julia_depot_path])
        sandbox_cmd.extend(["--julia-depot-path", args.julia_depot_path])

    p1 = subprocess.Popen(baseline_cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    p2 = subprocess.Popen(sandbox_cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    baseline_output, _ = p1.communicate()
    sandbox_output, _ = p2.communicate()
    assert_verify_passed("baseline-concurrent", subprocess.CompletedProcess(baseline_cmd, p1.returncode, baseline_output, ""), baseline_root)
    assert_verify_passed("sandbox-concurrent", subprocess.CompletedProcess(sandbox_cmd, p2.returncode, sandbox_output, ""), sandbox_root)
    print("  concurrent import checks OK")

    print("\n[3/3] SLURM worker script isolation")
    assert_job_script_isolated(
        baseline_root,
        sandbox_root,
        baseline_pyjuliapkg_project,
        sandbox_pyjuliapkg_project,
        args.julia_depot_path,
    )
    print("  generated worker scripts export distinct Julia project settings")

    print("\nIsolation status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
