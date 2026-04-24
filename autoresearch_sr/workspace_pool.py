"""Workspace pool for parallel MiniSR autoresearch experiments.

Each slot is a directory whose top-level entries are symlinks to the main
repo, except for two slot-owned items:
  - SymbolicRegression.jl/  (own clone — the submodule under experiment)
  - .juliapkg_env/          (own Manifest.toml, dev-pointed at the slot's SR)

So K experiments can evaluate K different MiniSR.jl versions in parallel,
while all Python harness code (mini_pysr.py, parallel_eval_minisr.py, etc.)
is live-linked — uncommitted edits in main apply to every slot immediately.

Usage:
    python workspace_pool.py setup --k 8
    python workspace_pool.py list
    python workspace_pool.py claim     # prints slot index
    python workspace_pool.py release 3
"""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

AUTORESEARCH_DIR = Path(__file__).resolve().parent
META_SR_ROOT = AUTORESEARCH_DIR.parent
WORKSPACES_DIR = AUTORESEARCH_DIR / "workspaces"
STATE_FILE = WORKSPACES_DIR / "pool_state.json"
LOCK_FILE = WORKSPACES_DIR / "pool.lock"

# A slot *owns* three things:
#   - SymbolicRegression.jl   (the submodule under experiment)
#   - .juliapkg_env           (Manifest dev-pinned to the slot's SR)
#   - PySR                    (overlay whose pysr/juliapkg.json points at the
#                              slot's SR — juliapkg reads this at startup)
# Everything else is symlinked to the main repo, so uncommitted edits to
# Python harness files (mini_pysr.py, parallel_eval_minisr.py, etc.) apply
# to every slot automatically.
SLOT_OWNED = {"SymbolicRegression.jl", ".juliapkg_env", "PySR"}
# Never symlink these (git internals, our own nested pool dir).
SLOT_SKIP = {".git", "autoresearch_sr"}


def slot_path(slot_idx: int) -> Path:
    return WORKSPACES_DIR / f"slot_{slot_idx}"


@contextmanager
def pool_lock() -> Iterator[None]:
    WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.touch(exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"slots": {}}
    with open(STATE_FILE) as f:
        return json.load(f)


def _save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(STATE_FILE)


def _run(cmd: list, cwd: Path | None = None, check: bool = True) -> None:
    loc = f" (in {cwd})" if cwd else ""
    print(f"  $ {' '.join(str(c) for c in cmd)}{loc}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def _build_slot(slot: Path) -> None:
    """Populate slot with symlinks into main for everything *except*
    SymbolicRegression.jl (own clone) and .juliapkg_env (own manifest).

    Any uncommitted edit to main's harness files (mini_pysr.py, etc.)
    therefore applies to every slot automatically — no re-clone needed.
    """
    slot.mkdir(parents=True, exist_ok=True)
    for item in META_SR_ROOT.iterdir():
        if item.name in SLOT_OWNED or item.name in SLOT_SKIP:
            continue
        link = slot / item.name
        if link.is_symlink():
            if link.resolve() == item.resolve():
                continue
            link.unlink()
        elif link.exists():
            if link.is_dir():
                shutil.rmtree(link)
            else:
                link.unlink()
        link.symlink_to(item, target_is_directory=item.is_dir())

    # SymbolicRegression.jl: slot gets its own clone from main's checked-out
    # submodule (fast, re-uses main's git objects, diverges on agent edits).
    main_sr = META_SR_ROOT / "SymbolicRegression.jl"
    slot_sr = slot / "SymbolicRegression.jl"
    if not (slot_sr / ".git").exists():
        if slot_sr.exists():
            shutil.rmtree(slot_sr)
        _run(["git", "clone", "--local", "--no-hardlinks", str(main_sr), str(slot_sr)])


def _build_pysr_overlay(slot: Path) -> None:
    """Create <slot>/PySR as a symlink overlay of main/PySR, except
    <slot>/PySR/pysr/juliapkg.json is rewritten to dev the slot's SR.

    juliapkg reads pysr/juliapkg.json at startup and re-resolves the Manifest
    if it mismatches, so substituting just the Manifest is not enough — the
    juliapkg.json must also point at the slot's SR.
    """
    main_pysr = META_SR_ROOT / "PySR"
    slot_pysr = slot / "PySR"
    slot_pysr.mkdir(exist_ok=True)

    # Top level of slot/PySR mirrors main/PySR, except 'pysr' (subpackage) is
    # its own overlay dir.
    for item in main_pysr.iterdir():
        if item.name == "pysr":
            continue
        link = slot_pysr / item.name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(item, target_is_directory=item.is_dir())

    slot_pysr_pkg = slot_pysr / "pysr"
    slot_pysr_pkg.mkdir(exist_ok=True)
    for item in (main_pysr / "pysr").iterdir():
        if item.name == "juliapkg.json":
            continue
        link = slot_pysr_pkg / item.name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(item, target_is_directory=item.is_dir())

    # Rewrite juliapkg.json: dev SR at the slot's SR, absolute path.
    src_juliapkg = main_pysr / "pysr" / "juliapkg.json"
    data = json.loads(src_juliapkg.read_text())
    sr_entry = data.setdefault("packages", {}).setdefault("SymbolicRegression", {})
    sr_entry["path"] = str((slot / "SymbolicRegression.jl").resolve())
    (slot_pysr_pkg / "juliapkg.json").write_text(json.dumps(data, indent=4))


def _patch_manifest(slot: Path) -> None:
    """Copy main env's Project.toml + Manifest.toml into the slot, rewriting
    the SymbolicRegression dev path to point at the slot's own submodule."""
    src_env = META_SR_ROOT / ".juliapkg_env"
    dst_env = slot / ".juliapkg_env"
    dst_env.mkdir(exist_ok=True)

    shutil.copy(src_env / "Project.toml", dst_env / "Project.toml")

    src_manifest = (src_env / "Manifest.toml").read_text()
    main_sr = str((META_SR_ROOT / "SymbolicRegression.jl").resolve())
    slot_sr = str((slot / "SymbolicRegression.jl").resolve())
    dst_manifest = src_manifest.replace(
        f'path = "{main_sr}"',
        f'path = "{slot_sr}"',
    )
    if dst_manifest == src_manifest:
        raise RuntimeError(
            f"Manifest path substitution failed: expected 'path = \"{main_sr}\"' "
            f"in {src_env / 'Manifest.toml'}"
        )
    (dst_env / "Manifest.toml").write_text(dst_manifest)

    # pyjuliapkg/ exists in the main env but is empty; create it for parity.
    (dst_env / "pyjuliapkg").mkdir(exist_ok=True)


def _setup_slot(slot_idx: int, force: bool) -> None:
    slot = slot_path(slot_idx)
    if slot.exists() and force:
        print(f"slot_{slot_idx}: --force given, removing {slot}")
        shutil.rmtree(slot)

    print(f"slot_{slot_idx}: building at {slot}")
    _build_slot(slot)

    print(f"slot_{slot_idx}: building PySR overlay pointing at slot's SR")
    _build_pysr_overlay(slot)

    print(f"slot_{slot_idx}: patching .juliapkg_env to dev its own SR submodule")
    _patch_manifest(slot)


def setup_pool(k: int, force: bool = False) -> None:
    WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)
    with pool_lock():
        state = _load_state()
        for i in range(k):
            _setup_slot(i, force=force)
            state["slots"].setdefault(str(i), {"busy": False, "path": str(slot_path(i))})
            state["slots"][str(i)]["path"] = str(slot_path(i))
        _save_state(state)
    print(f"\nPool ready: {k} slots at {WORKSPACES_DIR}")


def claim() -> int:
    with pool_lock():
        state = _load_state()
        for slot_id_str, info in sorted(state["slots"].items(), key=lambda kv: int(kv[0])):
            if not info.get("busy"):
                info["busy"] = True
                state["slots"][slot_id_str] = info
                _save_state(state)
                return int(slot_id_str)
    raise SystemExit("No free slots available")


def release(slot_idx: int) -> None:
    with pool_lock():
        state = _load_state()
        key = str(slot_idx)
        if key not in state["slots"]:
            raise SystemExit(f"Unknown slot {slot_idx}")
        state["slots"][key]["busy"] = False
        _save_state(state)


def list_pool() -> None:
    with pool_lock():
        state = _load_state()
    if not state["slots"]:
        print("Pool is empty. Run: python workspace_pool.py setup --k 8")
        return
    print(f"Pool state ({len(state['slots'])} slots):")
    for slot_id_str, info in sorted(state["slots"].items(), key=lambda kv: int(kv[0])):
        status = "BUSY" if info.get("busy") else "FREE"
        print(f"  slot_{slot_id_str}  {status}  {info.get('path')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_setup = sub.add_parser("setup", help="Create/refresh the pool")
    p_setup.add_argument("--k", type=int, default=8, help="Number of slots (default: 8)")
    p_setup.add_argument("--force", action="store_true",
                         help="Delete and re-clone any existing slot directories")

    sub.add_parser("list", help="List pool state")

    sub.add_parser("claim", help="Claim a free slot; print its index to stdout")

    p_release = sub.add_parser("release", help="Mark a slot as free")
    p_release.add_argument("slot", type=int)

    args = parser.parse_args()

    if args.cmd == "setup":
        setup_pool(args.k, force=args.force)
    elif args.cmd == "list":
        list_pool()
    elif args.cmd == "claim":
        print(claim())
    elif args.cmd == "release":
        release(args.slot)


if __name__ == "__main__":
    main()
