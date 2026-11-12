#!/usr/bin/env python3
"""Trim old PySR SLURM worker stderr logs to their first and last N bytes.

Before 2026-09-02 SLURM workers ran PySR with progress=True, so every task's
stderr log (``.../logs/*.err``) holds ~1 MB of redrawn progress bars and
hall-of-fame tables. The informative parts are at the start (compile messages,
warnings) and the end (final population, any traceback). This keeps the first
and last --keep bytes (cut at line boundaries) with a marker line in between.

Only touches files that: end in ``.err``, sit directly in a directory named
``logs``, are larger than --min-size, and were last modified more than
--min-age-days ago. ``.out`` files (parsed by analysis scripts) are never
touched. Each file is rewritten atomically (temp file + rename) with its
original mode and timestamps; files that change during processing are skipped.

    python scripts/trim_pysr_err_logs.py --dry-run ROOT [ROOT ...]
    python scripts/trim_pysr_err_logs.py ROOT [ROOT ...]
"""

import argparse
import os
import sys
import time
from datetime import date
from multiprocessing import Pool

MARKER = ("\n[... trimmed {day}: {n} bytes of PySR progress output removed from the middle "
          "of this log by scripts/trim_pysr_err_logs.py; head and tail kept ...]\n")


def candidates(roots, min_size, min_age_s, exclude):
    now = time.time()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if any(dirpath.startswith(e) for e in exclude):
                dirnames[:] = []
                continue
            if os.path.basename(dirpath) != "logs":
                continue
            for name in filenames:
                if not name.endswith(".err"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    st = os.lstat(path)
                except OSError:
                    continue
                if (os.path.isfile(path) and not os.path.islink(path)
                        and st.st_size > min_size and now - st.st_mtime > min_age_s):
                    yield path


def trim(args):
    path, keep, dry_run = args
    try:
        st = os.lstat(path)
        size = st.st_size
        with open(path, "rb") as fh:
            head = fh.read(keep)
            fh.seek(max(size - keep, keep))
            tail = fh.read()
        cut = head.rfind(b"\n")
        head = head[: cut + 1] if cut > 0 else head
        nl = tail.find(b"\n")
        tail = tail[nl + 1:] if 0 <= nl < len(tail) - 1 else tail
        removed = size - len(head) - len(tail)
        if removed <= 0:
            return path, 0, "skip-small"
        if dry_run:
            return path, removed, "dry"
        tmp = path + ".trimtmp"
        with open(tmp, "wb") as out:
            out.write(head)
            out.write(MARKER.format(day=date.today().isoformat(), n=removed).encode())
            out.write(tail)
        os.chmod(tmp, st.st_mode & 0o7777)
        st2 = os.lstat(path)
        if (st2.st_size, st2.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
            os.unlink(tmp)
            return path, 0, "skip-changed"
        os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
        os.replace(tmp, path)
        return path, removed, "trimmed"
    except Exception as exc:  # never abort the whole pass on one file
        try:
            if os.path.exists(path + ".trimtmp"):
                os.unlink(path + ".trimtmp")
        except OSError:
            pass
        return path, 0, f"error: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--keep", type=int, default=32768, help="bytes kept at each end")
    ap.add_argument("--min-size", type=int, default=262144)
    ap.add_argument("--min-age-days", type=float, default=3.0)
    ap.add_argument("--exclude", nargs="*", default=[], help="path prefixes to skip")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", default=None, help="write one line per processed file")
    args = ap.parse_args()
    roots = [os.path.realpath(r) for r in args.roots]
    exclude = [os.path.realpath(e) for e in args.exclude]
    jobs = ((p, args.keep, args.dry_run)
            for p in candidates(roots, args.min_size, args.min_age_days * 86400, exclude))
    counts, removed, t0 = {}, 0, time.time()
    log = open(args.log, "w") if args.log else None
    with Pool(args.workers) as pool:
        for i, (path, n, status) in enumerate(pool.imap_unordered(trim, jobs, chunksize=64), 1):
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            removed += n
            if log:
                log.write(f"{status}\t{n}\t{path}\n")
            if key == "error":
                print(f"{status}: {path}", file=sys.stderr)
            if i % 50000 == 0:
                print(f"{i} files, {removed / 1e12:.2f} TB removed, {time.time() - t0:.0f}s",
                      flush=True)
    print(f"done: {counts}; {'would remove' if args.dry_run else 'removed'} "
          f"{removed / 1e12:.3f} TB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
