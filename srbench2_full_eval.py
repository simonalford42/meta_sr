#!/usr/bin/env python3
"""Evaluate methods on the two SRBench 2.0 (2025) tracks.

This is the edition-specific entry point for the 12 black-box datasets and the
12 phenomenological/first-principles datasets in arXiv:2505.03977. Ground-truth
runs use this project's all-row exact-recovery protocol; black-box-only runs
retain the predictive SRBench protocol.

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


def _has_option(argv: list[str], *names: str) -> bool:
    return any(
        arg == name or arg.startswith(name + "=")
        for arg in argv
        for name in names
    )


def _with_srbench2_defaults(argv: list[str]) -> list[str]:
    """Apply SRBench-2 defaults without overriding explicit user arguments."""
    out = list(argv)
    if not _has_option(out, "--noise-levels"):
        out.extend(["--noise-levels", "0"])
    black_box_only = _has_option(out, "--black-box") and not _has_option(
        out, "--ground-truth"
    )
    if not black_box_only:
        if _has_option(out, "--black-box"):
            raise SystemExit(
                "Run SRBench 2.0 exact recovery and black-box evaluation in "
                "separate commands; they use different data/search protocols."
            )
        defaults = [
            (("--n-runs", "--n-trials-per-dataset"), "10"),
            (("--max-evals",), "1000000000"),
            (("--timeout",), "3600"),
            (("--pysr-wall-limit",), "3900"),
        ]
        for options, value in defaults:
            if not _has_option(out, *options):
                out.extend([options[0], value])
        out.append("--srbench2-exact-recovery")
    return out


def main() -> None:
    srbench_main(_with_srbench2_defaults(sys.argv[1:]), force_srbench_2025=True)


if __name__ == "__main__":
    main()
