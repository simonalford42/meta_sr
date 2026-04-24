"""Agent-facing CLI for parallel MiniSR experiments.

Subcommands:
  claim   Claim a free slot, seed its MiniSR.jl with the current main-repo
          version, print the slot path. Agent then edits
          <slot>/SymbolicRegression.jl/src/MiniSR.jl and optionally commits
          in that slot's submodule for diff capture.

  launch  Register an experiment in runs/<tag>/inflight.json and fork
          `evaluate_minisr.py --repo-root <slot>` into the background.
          Prints the PID.

  release Mark a slot free (and remove any inflight entry).

The authoritative MiniSR.jl state lives in the main repo at
  <main>/SymbolicRegression.jl/src/MiniSR.jl
Slots are scratch sandboxes; after an experiment is kept, the agent copies
the slot's MiniSR.jl back to the main repo and commits there.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

AUTORESEARCH_DIR = Path(__file__).resolve().parent
META_SR_ROOT = AUTORESEARCH_DIR.parent
MAIN_MINISR = META_SR_ROOT / "SymbolicRegression.jl" / "src" / "MiniSR.jl"

sys.path.insert(0, str(AUTORESEARCH_DIR))
from workspace_pool import claim as pool_claim, release as pool_release, slot_path  # noqa: E402


def inflight_path(tag: str) -> Path:
    return AUTORESEARCH_DIR / "runs" / tag / "inflight.json"


def _inflight_lock_path(tag: str) -> Path:
    return AUTORESEARCH_DIR / "runs" / tag / ".inflight.lock"


@contextmanager
def inflight_lock(tag: str) -> Iterator[None]:
    lock = _inflight_lock_path(tag)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch(exist_ok=True)
    with open(lock, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _load_inflight(tag: str) -> list:
    p = inflight_path(tag)
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def _save_inflight(tag: str, entries: list) -> None:
    p = inflight_path(tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(entries, f, indent=2)
    tmp.replace(p)


def _short_sha(repo: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def cmd_claim(args: argparse.Namespace) -> None:
    if not MAIN_MINISR.exists():
        raise SystemExit(f"Main MiniSR.jl not found at {MAIN_MINISR}")
    slot_idx = pool_claim()
    slot = slot_path(slot_idx)
    slot_minisr = slot / "SymbolicRegression.jl" / "src" / "MiniSR.jl"
    slot_minisr.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MAIN_MINISR, slot_minisr)

    base_sr_sha = _short_sha(META_SR_ROOT / "SymbolicRegression.jl")
    print(json.dumps({
        "slot": slot_idx,
        "slot_path": str(slot),
        "minisr_path": str(slot_minisr),
        "base_sr_sha": base_sr_sha,
    }, indent=2))


def cmd_launch(args: argparse.Namespace) -> None:
    slot_idx = args.slot
    slot = slot_path(slot_idx)
    if not slot.exists():
        raise SystemExit(f"Slot {slot_idx} does not exist at {slot}")

    exp_dir = AUTORESEARCH_DIR / "runs" / args.tag / str(args.exp_num)
    exp_dir.mkdir(parents=True, exist_ok=True)
    eval_log = exp_dir / "evaluate.log"

    # Launch evaluate_minisr.py detached. Use double-fork so parent returns
    # immediately and the child survives the agent's next shell invocation.
    slot_sr = slot / "SymbolicRegression.jl"
    eval_script = AUTORESEARCH_DIR / "evaluate_minisr.py"
    log_f = open(eval_log, "w")
    proc = subprocess.Popen(
        [sys.executable, str(eval_script), "--repo-root", str(slot)],
        cwd=str(AUTORESEARCH_DIR),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )

    entry = {
        "exp_num": args.exp_num,
        "slot": slot_idx,
        "slot_path": str(slot),
        "pid": proc.pid,
        "started_at": time.time(),
        "evaluate_log": str(eval_log),
        "exp_dir": str(exp_dir),
        "base_sr_sha": _short_sha(META_SR_ROOT / "SymbolicRegression.jl"),
        "slot_sr_sha": _short_sha(slot_sr),
        "description": args.description or "",
    }
    with inflight_lock(args.tag):
        entries = _load_inflight(args.tag)
        entries = [e for e in entries if e.get("exp_num") != args.exp_num]
        entries.append(entry)
        _save_inflight(args.tag, entries)

    print(json.dumps({
        "exp_num": args.exp_num,
        "slot": slot_idx,
        "pid": proc.pid,
        "evaluate_log": str(eval_log),
    }, indent=2))


def cmd_release(args: argparse.Namespace) -> None:
    pool_release(args.slot)
    if args.tag and args.exp_num is not None:
        with inflight_lock(args.tag):
            entries = _load_inflight(args.tag)
            entries = [e for e in entries if e.get("exp_num") != args.exp_num]
            _save_inflight(args.tag, entries)
    print(f"released slot {args.slot}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("claim", help="Claim a free slot and seed its MiniSR.jl")

    p_launch = sub.add_parser("launch", help="Register and launch an eval for a claimed slot")
    p_launch.add_argument("--tag", required=True)
    p_launch.add_argument("--exp-num", type=int, required=True)
    p_launch.add_argument("--slot", type=int, required=True)
    p_launch.add_argument("--description", type=str, default="")

    p_release = sub.add_parser("release", help="Release a slot (and optionally clear its inflight entry)")
    p_release.add_argument("slot", type=int)
    p_release.add_argument("--tag", type=str, default=None)
    p_release.add_argument("--exp-num", type=int, default=None)

    args = parser.parse_args()
    {"claim": cmd_claim, "launch": cmd_launch, "release": cmd_release}[args.cmd](args)


if __name__ == "__main__":
    main()
