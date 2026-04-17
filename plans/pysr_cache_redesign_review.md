# PySR Cache Redesign Review Notes

## Context

We hit SQLite corruption in `caches/pysr_evaluation_cache.db` while running multiple SLURM jobs on Unicorn. Unicorn docs indicate:

- `/home/<NETID>` is on an NFS server
- `/scratch` is local node storage

The repo lives under `/home/sca63/meta_sr`, so the global PySR cache DB was on NFS and was being accessed by many worker processes across nodes.

Observed failure mode:

- `parallel_eval_pysr.py` cache prefilter started logging `database disk image is malformed`
- baseline and other evaluations were rerun because cache lookup exceptions were caught and the code fell back to recomputation

## DB Recovery Done First

Before redesigning the cache path, I salvaged the corrupted DB:

- Original corrupted file preserved as:
  - `caches/pysr_evaluation_cache.db.corrupt`
  - `caches/pysr_evaluation_cache.db.bak`
- Sequential salvage initially copied `192000` readable rows into a fresh DB
- Healthy recovered DB now installed at:
  - `caches/pysr_evaluation_cache.db`
- The old corrupted DB reported `192599` rows, so about `599` rows were not recoverable by normal reads at salvage time
- The live recovered DB later drifted upward to `192775` rows because legacy jobs were still running and writing through the old worker-side SQLite path during review

Verification:

- recovered DB passes `PRAGMA quick_check`
- corrupted DB still fails `PRAGMA quick_check`

## Design Goal

Keep the existing SQLite cache for reads so currently-running old jobs are not broken, but stop worker-side SQLite writes.

New model:

1. Workers write only per-task JSON result files
2. Parent process queues finished task results and flushes them into SQLite at controlled boundaries
3. Cache lookups still read from the existing SQLite cache as before

This keeps compatibility with:

- `evolve_pysr.py`
- `evaluate_new_pysr.py`
- `run_openevolve_pysr.py`

because they all flow through `parallel_eval_pysr.PySRSlurmEvaluator.evaluate_configs()`

## Files Changed

- `parallel_eval_pysr.py`
- `evaluation_cache.py`

## What Changed

### 1. Added atomic JSON writes

In `parallel_eval_pysr.py`:

- added `_write_json_atomic(path, payload)`

It writes JSON to a temp file in the target directory, `fsync`s, then `os.replace`s into place.

This is now used for:

- cached result prewrites into `results/task_*.json`
- `tasks.json`
- `combined.json`
- worker result files
- worker fatal-error result files

### 2. Centralized cache identity construction

In `parallel_eval_pysr.py`:

- added `_build_cache_identity(spec)`

This builds:

- normalized `pysr_mutation_kwargs`
- `model_kwargs` including deterministic `random_state`

The goal is to ensure lookup and import use the same cache identity logic.

### 3. Added helper to turn a task/result pair into a cache row

In `parallel_eval_pysr.py`:

- added `_build_pysr_cache_entry(spec, result)`

This computes:

- `config_hash`
- `request_hash`
- cache row payload fields

using the same request-key logic as the existing SQLite cache.

### 4. Exposed deterministic request hash in the cache module

In `evaluation_cache.py`:

- added `PySRCacheDB.make_request_hash(...)`

This is just a public wrapper around the existing internal `_make_cache_key(...)`.

### 5. Added bulk insert/merge API

In `evaluation_cache.py`:

- added `PySRCacheDB.store_many(entries)`

This stores many cache rows in one transaction via SQLAlchemy `merge`.

### 6. Removed worker-side SQLite reads and writes

In `parallel_eval_pysr.py`:

- `_evaluate_pysr_task(...)` no longer touches SQLite at all
- it no longer performs worker-side cache lookup
- it no longer calls `cache.store(...)` on success
- it no longer calls `cache.store(...)` for deterministic failures

So workers now:

- write only JSON result files

### 7. Added parent-side queued compaction into SQLite

In `parallel_eval_pysr.py`:

- added `PySRSlurmEvaluator._queue_results_for_cache(tasks, results)`
- added `PySRSlurmEvaluator.flush_pending_cache()`

`evaluate_configs()` now queues finished task results and flushes them at the end of that completed evaluator batch.

Import policy:

- import successful results
- skip all failures for now
- skip malformed/worker-placeholder rows (`config_id < 0` or mismatched config)
- preserve `gt_match_score=None` instead of coercing it to `0.0`

### 8. Hooked flush ownership into the shared evaluator path

In `parallel_eval_pysr.py`, inside `PySRSlurmEvaluator.evaluate_configs(...)`:

- after final result collection / retries
- call `_queue_results_for_cache(tasks, results)`
- always flush at the end of that completed evaluator batch

This means all callers using the shared evaluator benefit automatically, and cache durability policy lives in one place.

### 9. Removed caller-managed flush policy

After review, the temporary caller-managed flush flags were removed:

- `evolve_pysr.py` no longer decides when cache flushing happens
- `hpo_evolve_pysr.py` no longer decides when cache flushing happens
- every `evaluate_configs()` batch flushes when it finishes

This gives:

- baseline flush after baseline batch
- initial population flush after initial-pop batch
- offspring flush after offspring batch
- racing flush after racing batch
- HPO flush after each trial batch

### 10. Made cache failures loud

In `parallel_eval_pysr.py`:

- cache prefilter failures now raise instead of silently falling back to uncached execution
- `flush_pending_cache()` raises on compaction failure instead of printing a warning and continuing

### 11. Added safer SQLite connection settings

In `evaluation_cache.py`:

- SQLite engines now use `connect_args={"timeout": 60}`
- `PRAGMA journal_mode=WAL`
- `PRAGMA synchronous=NORMAL`
- `PRAGMA busy_timeout=60000`

This does not make SQLite-on-NFS perfect, but it reduces contention risk for the remaining parent-side writes.

## Behavior After Patch

### Reads

Current state:

- prefilter still reads the existing SQLite cache
- only parent processes read SQLite
- workers no longer read SQLite

### Writes

Changed:

- workers no longer write SQLite
- parent writes SQLite at the end of each completed `evaluate_configs()` batch

### Compatibility

- existing old jobs that still expect SQLite reads should continue to work
- new code will continue to populate SQLite, but only from parent-side queued compaction / flush

## Validation Performed

### Compile check

Ran:

```bash
python -m py_compile parallel_eval_pysr.py evaluation_cache.py evolve_pysr.py evaluate_new_pysr.py run_openevolve_pysr.py
```

Then re-ran after the follow-up fixes with HPO included:

```bash
python -m py_compile parallel_eval_pysr.py evaluation_cache.py evolve_pysr.py hpo_evolve_pysr.py evaluate_new_pysr.py run_openevolve_pysr.py
```

### Sanity checks

Ran small Python checks to verify:

- `_write_json_atomic(...)` writes readable JSON
- `_build_pysr_cache_entry(...)` produces stable request/config hashes
- queued cache entries flush correctly via `flush_pending_cache()`
- queued compaction imports:
  - successful results
  - skips failures
- preserves nullable `gt_match_score`
- SQLite connection pragmas are applied on each connection

Observed result in targeted test:

- 2 entries imported
- success and nullable-`gt_match_score` result written
- failure result skipped
- `busy_timeout=60000`
- `journal_mode=wal`

## Important Constraints / Notes

### What this patch does **not** do

- It does **not** remove SQLite from the system
- It does **not** make SQLite on NFS “safe” in a general sense
- It does **not** stop multiple parent processes from compacting into the same SQLite DB

What it does do is drastically reduce writer concurrency:

- before: many workers across nodes could write SQLite
- after: only parent processes write SQLite

It also now:

- removes worker-side SQLite access entirely
- raises on cache prefilter failures
- raises on cache flush failures
- makes `parallel_eval_pysr.py` the sole owner of flush timing

That should be much safer, but it is still not as strong as a fully file-based cache or a true DB server.

### Old jobs

User explicitly requested:

- do not remove the old cache because there are existing jobs still running

So I preserved the existing cache path and kept reads intact.

## Follow-up Work I’d Recommend

### 1. Add a rebuild/import tool

Suggested new script:

- scan `runs/*/slurm_pysr/*/results/task_*.json`
- reconstruct request hashes from the associated `tasks.json`
- repopulate SQLite from historical results

### 2. Add batch/import manifests

To avoid reimporting the same batch repeatedly:

- write a marker file per batch after successful compaction
- or keep a tiny imported-batches table

### 3. Consider a future file-based cache

If we later want to leave SQLite entirely:

- use request-hash-addressed JSON files as the cache itself
- keep SQLite optional or derived-only

## Questions For Review

1. Is parent-only SQLite access now sufficient for this repo, or should we still move to a fully file-based cache later?
2. Should `store_many(...)` use lower-level SQLite upsert/insert-or-ignore instead of ORM `merge` for performance?
3. Is per-`evaluate_configs()` batch flush the right boundary, or is there a strong reason to batch more coarsely?
4. Should failure results remain uncached, or do you want lookup semantics expanded to make some failures reusable?
