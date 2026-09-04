#!/usr/bin/env python3
"""Evaluate methods on the two SRBench 2.0 (2025) tracks.

This is the edition-specific entry point for the 12 black-box datasets and the
12 phenomenological/first-principles datasets in arXiv:2505.03977. It delegates
execution and aggregation to ``srbench_full_eval.py`` while fixing the edition
to SRBench 2.0 and defaulting to the benchmark's intrinsic-noise-only setting.

Examples::

    # Phenomenological/first-principles track (called ground truth internally)
    python srbench2_full_eval.py --ground-truth --noise-levels 0

    # Black-box track
    python srbench2_full_eval.py --black-box

    # Run both tracks
    python srbench2_full_eval.py --ground-truth --black-box --noise-levels 0

Ground-truth PySR workers retain the complete Pareto frontier so it can be
reviewed with ``scripts/review_srbench2_frontiers.py``.
"""

from __future__ import annotations

import sys

from srbench_full_eval import main as srbench_main


def _with_srbench2_defaults(argv: list[str]) -> list[str]:
    """Apply SRBench-2 defaults without overriding explicit user arguments."""
    out = list(argv)
    if "--noise-levels" not in out:
        out.extend(["--noise-levels", "0"])
    return out


def main() -> None:
    srbench_main(_with_srbench2_defaults(sys.argv[1:]), force_srbench_2025=True)


if __name__ == "__main__":
    main()
