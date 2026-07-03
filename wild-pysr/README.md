# Wild PySR: first-pass repository census

This directory records a read-only first pass toward a `RealScience` symbolic-regression benchmark. It inventories repositories containing the exact identifier `PySRRegressor`, then prioritizes plausible scientific and engineering applications. No external repository was cloned and no external code was executed.

## Current result

The authenticated GitHub search completed on 2026-07-03. A canonical union with the earlier Sourcegraph pass contains **640 repositories**:

| Source coverage | Repositories |
|---|---:|
| GitHub and Sourcegraph | 35 |
| GitHub only | 602 |
| Sourcegraph only | 3 |
| **Canonical union** | **640** |

`MilesCranmer/PySR` is canonicalized to its current GitHub name, `astroautomata/PySR`, before calculating the union.

The GitHub API returned 2,638 raw file records from non-overlapping size-partitioned queries. After removing 310 duplicate records, the snapshot contains **2,328 unique public matching files in 637 repositories**. No private or fork repository rows were written.

GitHub's reported `total_count` was not reliable: the unqualified query reported 2,736 files, and counts changed between pages of identical queries. Pagination therefore continued until an empty page for every size partition; counts were retained only as provenance. See [`notes/github_query_pass.md`](notes/github_query_pass.md) for details.

## Scientific review

All 637 GitHub repositories received a transparent heuristic score so the screening decision is reproducible:

| Heuristic priority | Repositories |
|---|---:|
| High | 60 |
| Medium | 141 |
| Deprioritized | 436 |

Heuristic priority is not a scientific classification. Public README, metadata, and matching-file evidence was manually reviewed for 64 plausible applications:

| Manual class | Repositories |
|---|---:|
| `real_science_candidate` | 45 |
| `scientific_synthetic` | 13 |
| `unclear` | 4 |
| `method_or_incidental` | 2 |
| **Reviewed** | **64** |

Of the 45 real-science candidates, **25 are high priority** for the next extraction/reproduction pass. Data origin is recorded separately (`observational`, `experimental`, `empirical`, `simulation`, `mixed`, or `unknown`) so simulation-based scientific work is not confused with measured data.

The earlier 38-repository Sourcegraph review remains in [`notes/decision_log.md`](notes/decision_log.md); the expanded GitHub review supersedes its candidate count.

## Reproduce

From the repository root:

```bash
# Authenticated GitHub search. Uses GITHUB_TOKEN/GH_TOKEN or `git credential fill`.
python3 wild-pysr/scripts/discover_github.py

# Score all GitHub repositories for review, then build the canonical union.
python3 wild-pysr/scripts/triage_github.py
python3 wild-pysr/scripts/combine_inventories.py

# Optional independent Sourcegraph pass.
python3 wild-pysr/scripts/discover.py
```

The GitHub script never stores the credential. It omits private results before writing repository names or paths.

The exact GitHub base query is:

```text
PySRRegressor
```

GitHub's REST code-search endpoint exposes at most 1,000 results per query. The script covers the documented searchable file-size range with non-overlapping `size:` ranges and recursively splits any range whose reported count is too large.

## Scope limitations

This is the most complete reproducible snapshot obtained from the two indexes, not proof of every public GitHub use. GitHub documents that legacy API code search only indexes default branches, excludes archived repositories, ignores files at least 384 KiB, applies repository-activity/size restrictions, and does not normally include forks. Sourcegraph has different coverage and returned three repositories absent from the GitHub results. GitHub also produced unstable counts and duplicate pagination records during this run.

Manual review is intentionally selective because 640 repositories greatly exceeds the target benchmark size. The unreviewed/deprioritized set remains preserved so screening can be expanded later without rerunning discovery.

## Files

- `data/combined_repositories.csv`: canonical 640-repository union with source coverage, triage, and manual-review columns.
- `data/github_repositories.csv`: 637 repositories returned by authenticated GitHub retrieval.
- `data/github_matched_files.csv`: 2,328 unique public GitHub file matches with blob-pinned URLs.
- `data/github_search_snapshot.json`: exact partitions, page observations, counts, and API limitations.
- `data/github_triage.csv`: reproducible heuristic score and reasons for every GitHub repository.
- `data/github_manual_review.csv`: 64 evidence-backed manual decisions, including 45 real-science candidates.
- `notes/github_query_pass.md`: GitHub methodology, findings, and the 25 high-priority candidates.
- `notes/benchmark_metrics.md`: metric and validity review for those 25, plus the `planet_eqs` calibration case.
- `data/benchmark_metrics_review.csv`: machine-readable per-repository benchmark recommendations.
- `scripts/discover_github.py`: authenticated, rate-limit-aware GitHub search.
- `scripts/triage_github.py`: deterministic shortlist heuristic.
- `scripts/combine_inventories.py`: canonical GitHub/Sourcegraph union.
- `data/repositories.csv`, `data/matched_files.csv`, `data/search_snapshot.json`, `data/classification.csv`: original Sourcegraph snapshot and classifications.
