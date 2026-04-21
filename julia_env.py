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
from pathlib import Path
from typing import Optional, Union


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
