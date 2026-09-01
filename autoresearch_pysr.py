"""Commit-isolated PySR sandboxes for SymbolicRegression.jl autoresearch."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SUBMODULE = REPO_ROOT / "SymbolicRegression.jl"
DEFAULT_RESULTS_TSV = REPO_ROOT / "autoresearch_sr" / "results.tsv"
DEFAULT_SANDBOX_ROOT = REPO_ROOT / "outputs" / "autoresearch_pysr_sandboxes"


def read_results_tsv(path: Path) -> List[Dict[str, str]]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Autoresearch results file missing: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _score(row: Dict[str, str]) -> float:
    """Use confirmed train score when available, then the quick train score."""
    for key in ("train_confirm", "score2", "train_quick", "score"):
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise ValueError(f"Autoresearch row has no recognized score column: {row}")


def resolve_commit(
    target: Optional[str],
    submodule_path: Path = DEFAULT_SUBMODULE,
    results_tsv: Path = DEFAULT_RESULTS_TSV,
) -> str:
    """Resolve ``best``, ``latest``, ``expN``, a branch, or a commit SHA."""
    submodule_path = submodule_path.resolve()
    target = target or "best"
    if target == "best":
        all_rows = read_results_tsv(results_tsv)
        # A discarded candidate must never become the final winner. Legacy
        # tables without keep/discard labels fall back to every non-crash row.
        rows = [r for r in all_rows if r.get("status") == "keep"]
        if not rows:
            rows = [r for r in all_rows if r.get("status") != "crash"]
        if not rows:
            raise ValueError(f"No non-crash rows in {results_tsv}")
        row = max(rows, key=lambda r: (_score(r), int(r.get("exp", "0") or 0)))
        revspec = row["commit"]
        print(
            f"[autoresearch] best row: exp{row.get('exp', '?')} "
            f"score={_score(row):.6f} status={row.get('status', '?')}",
            flush=True,
        )
    elif target.lower().startswith("exp"):
        exp = target[3:]
        row = next(
            (r for r in read_results_tsv(results_tsv) if r.get("exp") == exp),
            None,
        )
        if row is None:
            raise ValueError(f"Experiment {target!r} not found in {results_tsv}")
        revspec = row["commit"]
    elif target.lower() in ("latest", "head"):
        revspec = "HEAD"
    else:
        revspec = target

    result = subprocess.run(
        ["git", "-C", str(submodule_path), "rev-parse", f"{revspec}^{{commit}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def build_sandbox(
    commit: str,
    submodule_path: Path = DEFAULT_SUBMODULE,
    sandbox_root: Path = DEFAULT_SANDBOX_ROOT,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Build or return a persistent, commit-isolated PySR repository overlay."""
    submodule_path = submodule_path.resolve()
    sandbox_root = sandbox_root.resolve()
    repo_root = repo_root.resolve()
    sandbox = sandbox_root / commit
    ready = sandbox / ".autoresearch_ready.json"
    expected = {
        "commit": commit,
        "submodule": str(submodule_path),
        "repo_root": str(repo_root),
    }
    if ready.exists():
        try:
            if json.loads(ready.read_text()) == expected:
                return sandbox
        except (OSError, json.JSONDecodeError):
            pass
        raise RuntimeError(f"Invalid existing autoresearch sandbox: {sandbox}")

    sandbox_root.mkdir(parents=True, exist_ok=True)
    sandbox.mkdir()
    sr_worktree = sandbox / "SymbolicRegression.jl"
    try:
        subprocess.run(
            ["git", "-C", str(submodule_path), "worktree", "add", "--detach",
             str(sr_worktree), commit],
            check=True,
        )

        real_pysr = repo_root / "PySR"
        sandbox_pysr = sandbox / "PySR"
        sandbox_pysr.mkdir()
        for item in real_pysr.iterdir():
            if item.name != "pysr":
                (sandbox_pysr / item.name).symlink_to(item)

        sandbox_package = sandbox_pysr / "pysr"
        sandbox_package.mkdir()
        for item in (real_pysr / "pysr").iterdir():
            if item.name != "juliapkg.json":
                (sandbox_package / item.name).symlink_to(item)

        deps = json.loads((real_pysr / "pysr" / "juliapkg.json").read_text())
        sr_dep = deps.setdefault("packages", {}).setdefault("SymbolicRegression", {})
        sr_dep["dev"] = True
        sr_dep["path"] = str(sr_worktree)
        (sandbox_package / "juliapkg.json").write_text(json.dumps(deps, indent=2) + "\n")

        (sandbox / ".juliapkg_env").mkdir()
        overrides = {
            ".git", ".juliapkg_env", ".autoresearch_ready.json", "PySR",
            "SymbolicRegression.jl",
        }
        for item in repo_root.iterdir():
            if item.name not in overrides:
                (sandbox / item.name).symlink_to(item)
        ready.write_text(json.dumps(expected, indent=2) + "\n")
    except Exception:
        subprocess.run(
            ["git", "-C", str(submodule_path), "worktree", "remove", "--force",
             str(sr_worktree)],
            check=False,
            capture_output=True,
        )
        shutil.rmtree(sandbox, ignore_errors=True)
        raise
    return sandbox


def resolve_and_build(
    target: Optional[str],
    submodule_path: Path = DEFAULT_SUBMODULE,
    results_tsv: Path = DEFAULT_RESULTS_TSV,
    sandbox_root: Path = DEFAULT_SANDBOX_ROOT,
) -> tuple[str, Path]:
    commit = resolve_commit(target, submodule_path, results_tsv)
    return commit, build_sandbox(commit, submodule_path, sandbox_root)
