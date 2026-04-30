"""One-shot: convert pysr_evaluation_cache.db from WAL to DELETE journaling.

Run with NO other process holding the DB open. The script:
  1. Backs up .db, .db-wal, .db-shm together as a matched set.
  2. Runs PRAGMA integrity_check (refuses to proceed if not 'ok').
  3. Runs PRAGMA wal_checkpoint(TRUNCATE) and refuses to proceed unless the
     result tuple is clean (busy=0 AND checkpointed covered every WAL frame).
  4. Switches journal_mode to DELETE.
  5. Re-runs integrity_check.
  6. Removes leftover .db-wal / .db-shm only after the steps above all pass.

If any step fails, the script aborts with the original DB and backups intact.

Usage:
    python scripts/migrate_pysr_cache_to_truncate.py [path/to/cache.db]
"""

import shutil
import sqlite3
import sys
from pathlib import Path


def _backup_set(db_path: Path) -> Path:
    """Snapshot db, wal, shm together before any mutation."""
    backup_dir = db_path.parent / f"{db_path.name}.pre_truncate_backup"
    if backup_dir.exists():
        sys.exit(
            f"Backup dir already exists: {backup_dir} (refusing to overwrite). "
            "Move or remove it and re-run."
        )
    backup_dir.mkdir()
    print(f"  Backing up matched set to: {backup_dir}/")
    for ext in ("", "-wal", "-shm"):
        src = Path(str(db_path) + ext)
        if src.exists():
            dst = backup_dir / src.name
            shutil.copy2(src, dst)
            print(f"    {src.name} ({src.stat().st_size:,} bytes)")
    return backup_dir


def _integrity_check(conn: sqlite3.Connection, label: str) -> None:
    print(f"  Integrity check ({label})...")
    rows = conn.execute("PRAGMA integrity_check").fetchall()
    if rows == [("ok",)]:
        print("    -> ok")
    else:
        sys.exit(f"    -> FAILED: {rows[:5]}")


def migrate(db_path: Path) -> None:
    if not db_path.exists():
        sys.exit(f"DB not found: {db_path}")

    print(f"Migrating: {db_path}")

    # Step 0: matched-set backup (covers .db + .db-wal + .db-shm).
    _backup_set(db_path)

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        cur = conn.cursor()

        current_mode = cur.execute("PRAGMA journal_mode").fetchone()[0]
        print(f"  Current journal_mode: {current_mode}")
        if current_mode.lower() != "wal":
            print("  Already non-WAL; nothing to do.")
            return

        # Step 1: integrity check before mutating anything.
        _integrity_check(conn, "pre-checkpoint")

        # Step 2: checkpoint and verify the result is clean.
        # PRAGMA wal_checkpoint(TRUNCATE) returns (busy, log_frames, checkpointed).
        # busy == 0  -> no other connection blocked us
        # log_frames == checkpointed -> every WAL frame was moved into the main DB
        # (For TRUNCATE specifically log/checkpointed report the work done; both
        # being 0 means there was nothing to checkpoint, which is also clean.)
        print("  Running PRAGMA wal_checkpoint(TRUNCATE)...")
        result = cur.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        print(f"    -> (busy={result[0]}, log_frames={result[1]}, checkpointed={result[2]})")
        if result[0] != 0:
            sys.exit(
                "    ABORT: checkpoint busy != 0 -- another process holds the DB. "
                "Stop all evolve_pysr.py / val eval / readers and re-run."
            )
        if result[1] != result[2]:
            sys.exit(
                "    ABORT: checkpoint did not cover every WAL frame "
                f"(log_frames={result[1]}, checkpointed={result[2]}). "
                "Refusing to delete -wal/-shm — backups are at "
                f"{db_path.name}.pre_truncate_backup/."
            )

        # Step 3: switch to DELETE journaling (persistent for WAL->non-WAL).
        print("  Switching journal_mode to DELETE...")
        new_mode = cur.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        print(f"    -> {new_mode}")
        if new_mode.lower() != "delete":
            sys.exit(f"    ABORT: journal_mode is {new_mode!r}, expected DELETE.")

        # Step 4: integrity check after the switch.
        _integrity_check(conn, "post-switch")
    finally:
        conn.close()

    # Step 5: only now remove the leftover WAL/SHM files.
    for ext in ("-wal", "-shm"):
        side = Path(str(db_path) + ext)
        if side.exists():
            print(f"  Removing leftover {side.name}")
            side.unlink()

    print(
        "Done. The DB is now in DELETE journal mode. evaluation_cache.py's "
        "synchronous=FULL / busy_timeout=300s pragmas will apply on next open. "
        f"Backups: {db_path.name}.pre_truncate_backup/"
    )


if __name__ == "__main__":
    default = Path("caches/pysr_evaluation_cache.db")
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    migrate(db.resolve())
