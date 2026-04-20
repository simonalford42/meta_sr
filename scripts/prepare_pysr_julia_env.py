#!/usr/bin/env python3
"""Prepare this checkout's clone-local PySR Julia environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    juliapkg_project = repo_root / ".juliapkg_env"
    juliapkg_project.mkdir(parents=True, exist_ok=True)

    os.environ["PYTHON_JULIAPKG_PROJECT"] = str(juliapkg_project)
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    os.environ.pop("JULIA_PROJECT", None)

    from parallel_eval_pysr import _import_pysr_regressor

    _import_pysr_regressor()
    print(f"PySR OK")
    print(f"PYTHON_JULIAPKG_PROJECT={juliapkg_project}")
    print(f"JULIA_DEPOT_PATH={os.environ.get('JULIA_DEPOT_PATH', '(Julia default)')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
