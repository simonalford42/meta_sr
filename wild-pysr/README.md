# Wild PySR: first-pass repository census

This directory records a read-only first pass toward a `RealScience` symbolic-regression benchmark. The pass inventories repositories containing the exact identifier `PySRRegressor`, then classifies how the repository uses it. No external repository was cloned and no external code was executed.

## Snapshot result

The 2026-07-02 snapshot found **38 repositories in Sourcegraph's public index**. The inventory is the union of repositories returned by the repository-selection and matching-file queries, including indexed forks and archived repositories. It is **not proven to be every public GitHub repository**: GitHub requires authentication for code search in this environment, and Sourcegraph does not claim to index all of GitHub.

An earlier run observed slightly different index state between the two Sourcegraph queries: the repository query returned 37 repositories while the file query returned matches from 38, with `THUIR/MemoryBench` only in the latter. This motivated the documented union rule. In the final committed snapshot both queries returned all 38 repositories.

The manual first-pass labels are:

| Class | Count | Meaning |
|---|---:|---|
| `real_science_candidate` | 6 | PySR is applied within a real scientific or engineering investigation; worth inspecting for extractable benchmark problems. |
| `scientific_synthetic` | 4 | Genuine scientific research or pedagogy, but the PySR targets are synthetic/simulated systems with known construction. |
| `sr_method_or_benchmark` | 11 | The repository develops/evaluates SR methods or infrastructure rather than applying SR to a new domain problem. |
| `incidental_or_educational` | 16 | PySR is a tutorial, example, auxiliary analysis, future-work snippet, vendored file, or otherwise not central. |
| `false_positive` | 1 | The text match is not repository code using PySR. |
| **Total** | **38** | |

See [`notes/decision_log.md`](notes/decision_log.md) for the result for every repository and [`data/classification.csv`](data/classification.csv) for a machine-readable version.

## Reproduce the census

From the repository root:

```bash
python3 wild-pysr/scripts/discover.py
```

The script makes read-only requests to Sourcegraph's streaming search endpoint and overwrites the generated snapshot tables in `data/`. The exact queries, timestamps, completion status, and any Sourcegraph skip warnings are saved in `data/search_snapshot.json`.

Discovery query:

```text
context:global fork:yes archived:yes select:repo PySRRegressor count:10000 timeout:2m
```

Matching-file query:

```text
context:global fork:yes archived:yes select:file PySRRegressor count:10000 timeout:2m
```

Classification is deliberately manual and is not overwritten by the discovery script. Evidence was limited to public repository metadata, README files, and the matching source/notebook files. The next pass should validate candidates by cloning at pinned commits, mapping each PySR call to its `X` and `y`, and checking paper/config/result correspondence.

The final file query reported Sourcegraph's per-shard match limit because repositories such as Frontier-CS contain many repetitive solution files. Therefore, `matched_files.csv` contains every path returned by the query (234), but it is a lower bound on matching paths. The repository-selection query completed with no skipped results, and both queries represented the same 38 repositories.

## Files

- `data/repositories.csv`: one row per indexed repository and metadata reported by Sourcegraph.
- `data/matched_files.csv`: every matching path returned by the file query.
- `data/search_snapshot.json`: query provenance and raw repository-level search results.
- `data/classification.csv`: manual labels, priority, confidence, and concise rationale.
- `notes/decision_log.md`: readable per-repository review with evidence links.
- `scripts/discover.py`: reproducible read-only census script.
