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

import errno
import fcntl
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


def clear_future_mtime_pidfiles(
    julia_depot: Optional[Union[str, Path]] = None,
    quiet: bool = False,
) -> int:
    """Delete Julia depot `.pid` files whose mtime is in the future.

    Julia's `pidfile.jl` uses mtime to decide if a lock is stale. If a prior
    process was killed in a way that left a .pid file with a future mtime
    (NFS clock skew, corrupted touch), Julia will wait on it forever via
    `watch_file`. Any PySR import touching that lock then hangs indefinitely.
    A real Pkg lock-holder updates the mtime every few seconds while alive,
    so a future mtime cannot be a live holder — it's always corruption.

    Returns the number of pidfiles cleared. Never raises.
    """
    if julia_depot is None:
        depot = os.environ.get("JULIA_DEPOT_PATH")
        julia_depot = depot.split(os.pathsep)[0] if depot else str(Path.home() / ".julia")
    depot_root = Path(julia_depot)
    if not depot_root.exists():
        return 0
    now = time.time()
    cleared = 0
    search_dirs = [depot_root / "logs", depot_root / "compiled"]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        try:
            candidates = list(search_dir.rglob("*.pid"))
        except OSError:
            continue
        for path in candidates:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime <= now:
                continue
            try:
                path.unlink()
            except OSError:
                continue
            cleared += 1
            if not quiet:
                print(
                    f"[julia-depot] cleared stale pidfile {path} (mtime was "
                    f"{mtime - now:.0f}s in the future)",
                    flush=True,
                )
    return cleared


def clear_stale_juliapkg_lock(
    project_path: Optional[Union[str, Path]] = None,
    quiet: bool = False,
) -> bool:
    """Unlink `lock.pid` in the juliapkg project if no process actively holds it.

    juliapkg uses a `filelock.FileLock` on `<project>/lock.pid`. The lock is an
    `fcntl.flock` on the file descriptor; the kernel releases it on process
    death, but the filename persists. If a worker is killed mid-resolve (SLURM
    preemption, scancel, OOM, the 30-min array cap), the file can be left
    behind in a state that confuses subsequent juliapkg runs on NFS.

    Strategy: open the file, try a non-blocking exclusive flock. If we get it,
    nobody is actively resolving — we can safely unlink and let juliapkg
    recreate it cleanly on the next run. If flock would block, leave it alone
    (a real holder is doing real work).

    Returns True if a stale lock was cleared, False otherwise. Never raises.
    """
    if project_path is None:
        project_path = os.environ.get("PYTHON_JULIAPKG_PROJECT")
    if not project_path:
        return False
    lock_file = Path(project_path) / "lock.pid"
    if not lock_file.exists():
        return False
    try:
        fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            return False
        try:
            lock_file.unlink()
        except FileNotFoundError:
            return False
        if not quiet:
            print(f"[juliapkg] cleared stale lock at {lock_file}", flush=True)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        return True
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


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
