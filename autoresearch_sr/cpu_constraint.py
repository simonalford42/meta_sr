#!/usr/bin/env python3
"""SLURM --constraint for PySR workers, chosen from this (driver) node's CPU.

Each candidate commit's SymbolicRegression.jl is compiled once by the driver,
for the driver's CPU family. A Julia package image loads on that family and on
newer families that have all its CPU features, but not on older ones (those
recompile it, serially, behind one lock). So workers are pinned to the driver's
family and newer. Verified 2026-11-10: a Cascade Lake build loads on Ice Lake
(14 s) and Sapphire Rapids (7 s) workers without recompiling.

Prints the constraint, or exits 1 for unsupported driver CPUs.
"""

import sys
from pathlib import Path

# (CPU flag identifying the family, worker constraint), newest family first.
FAMILIES = [
    ("amx_tile", "sr"),             # Sapphire Rapids
    ("gfni", "il|sr"),              # Ice Lake
    ("avx512_vnni", "cl|il|sr"),    # Cascade Lake
]


def worker_constraint() -> str:
    flags = set()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("flags"):
            flags = set(line.split(":", 1)[1].split())
            break
    for flag, constraint in FAMILIES:
        if flag in flags:
            return constraint
    raise SystemExit(
        "Unsupported driver CPU: run on a Cascade Lake, Ice Lake or Sapphire Rapids "
        "node (e.g. ellis-compute-01 or ellis-compute-02)."
    )


if __name__ == "__main__":
    print(worker_constraint())
