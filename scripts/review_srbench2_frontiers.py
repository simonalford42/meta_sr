#!/usr/bin/env python3
"""Compatibility entry point for the API-only SRBench 2.0 reviewer."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manual_solve_check import main  # noqa: E402


if __name__ == "__main__":
    main()
