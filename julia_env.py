"""Shared helper for configuring the juliacall/juliapkg environment.

Single policy: point PYTHON_JULIAPKG_PROJECT at <repo_root>/.juliapkg_env (or a
caller-supplied override), unset JULIA_PROJECT (juliapkg picks the project
itself and a pre-set JULIA_PROJECT blocks PythonCall resolution), and ensure
PYTHON_JULIACALL_HANDLE_SIGNALS=yes.

Uses setdefault so that SLURM worker scripts which already exported explicit
values win. If a caller wants strictness (override ambient env), it should
assign os.environ["PYTHON_JULIAPKG_PROJECT"] inline before calling this.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union


def configure_juliapkg_project(
    repo_root: Union[str, Path],
    python_juliapkg_project: Optional[Union[str, Path]] = None,
    julia_depot_path: Optional[Union[str, Path]] = None,
) -> str:
    """Pin juliapkg to <repo_root>/.juliapkg_env (or an explicit override).

    Returns the resolved PYTHON_JULIAPKG_PROJECT path (whatever ended up in
    the environment after setdefault).
    """
    if python_juliapkg_project is not None:
        target = str(Path(python_juliapkg_project).resolve())
    else:
        target = str((Path(repo_root) / ".juliapkg_env").resolve())
    Path(target).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", target)
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    os.environ.pop("JULIA_PROJECT", None)

    if julia_depot_path is not None:
        os.environ.setdefault("JULIA_DEPOT_PATH", str(Path(julia_depot_path).resolve()))

    return os.environ["PYTHON_JULIAPKG_PROJECT"]


@contextmanager
def _redirect_fds_to_file(log_path: Union[str, Path]) -> Iterator[None]:
    """Redirect OS-level fds 1 and 2 to log_path for the duration of the block.

    Captures both Python-level prints and direct fd writes from native code
    (juliapkg's `[juliapkg]` banner, Julia's Pkg.add/precompile output, etc.).
    Normal stdout/stderr handles are restored on exit.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
    finally:
        os.close(log_fd)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)


def warmup_julia(
    log_path: Union[str, Path],
    using_statements: Optional[list] = None,
) -> float:
    """Run the first-time juliacall import with all output captured to a log file.

    juliacall's first import resolves + installs the Julia environment and
    Julia's Pkg machinery prints a large block to stdout/stderr. This helper
    funnels all that into `log_path` so the main console log stays readable.
    Subsequent juliacall imports in the same process are no-ops — the setup is
    cached — so this is safe to call exactly once up-front.

    `using_statements` lets callers warm additional modules (e.g.
    SymbolicRegression + the CustomMutations submodules) so the first
    `validate_julia_code` call doesn't itself re-pay the `using` cost.
    """
    using_statements = using_statements or []
    start = time.time()
    with _redirect_fds_to_file(log_path):
        from juliacall import Main as jl  # noqa: F401
        for stmt in using_statements:
            try:
                jl.seval(stmt)
            except Exception as e:
                # Keep going — a failed `using` doesn't break the main loop;
                # subsequent validate_julia_code will surface it with context.
                print(f"[warmup_julia] {stmt!r} failed: {e}")
    return time.time() - start
