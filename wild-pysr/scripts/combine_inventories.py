#!/usr/bin/env python3
"""Combine GitHub and Sourcegraph repository inventories into canonical rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# GitHub redirects the historical repository name to the current canonical name.
ALIASES = {"milescranmer/pysr": "astroautomata/pysr"}


def canonical(name: str) -> str:
    return ALIASES.get(name.casefold(), name).casefold()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    args = parser.parse_args()

    sourcegraph_rows = read_rows(args.data_dir / "repositories.csv")
    github_rows = read_rows(args.data_dir / "github_repositories.csv")
    triage_rows = read_rows(args.data_dir / "github_triage.csv")
    review_rows = read_rows(args.data_dir / "github_manual_review.csv")
    old_classifications = read_rows(args.data_dir / "classification.csv")

    sourcegraph = {canonical(row["repository"]): row for row in sourcegraph_rows}
    github = {canonical(row["repository"]): row for row in github_rows}
    triage = {canonical(row["repository"]): row for row in triage_rows}
    review = {canonical(row["repository"]): row for row in review_rows}
    old = {canonical(row["repository"]): row for row in old_classifications}

    names = sorted(set(sourcegraph) | set(github))
    output_rows: list[dict[str, str | int]] = []
    for key in names:
        gh = github.get(key, {})
        sg = sourcegraph.get(key, {})
        manual = review.get(key, {})
        heuristic = triage.get(key, {})
        old_row = old.get(key, {})
        display_name = gh.get("repository") or sg.get("repository") or key
        output_rows.append(
            {
                "repository": display_name,
                "url": gh.get("url") or sg.get("url") or f"https://github.com/{display_name}",
                "found_by_github": str(bool(gh)).lower(),
                "found_by_sourcegraph": str(bool(sg)).lower(),
                "github_matching_file_count": gh.get("matching_file_count", ""),
                "sourcegraph_matching_file_count": sg.get("matching_file_count", ""),
                "description": gh.get("description") or sg.get("description", ""),
                "heuristic_score": heuristic.get("science_likelihood_score", ""),
                "heuristic_priority": heuristic.get("triage_priority", ""),
                "manually_reviewed": str(bool(manual)).lower(),
                "manual_classification": manual.get("classification", ""),
                "phase2_priority": manual.get("phase2_priority", ""),
                "data_origin": manual.get("data_origin", ""),
                "confidence": manual.get("confidence", ""),
                "manual_evidence_summary": manual.get("evidence_summary", ""),
                "prior_sourcegraph_classification": old_row.get("classification", ""),
            }
        )

    output = args.data_dir / "combined_repositories.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)

    github_only = set(github) - set(sourcegraph)
    sourcegraph_only = set(sourcegraph) - set(github)
    overlap = set(github) & set(sourcegraph)
    print(
        f"Wrote {len(output_rows)} canonical repositories to {output}; "
        f"GitHub-only={len(github_only)}, Sourcegraph-only={len(sourcegraph_only)}, "
        f"both={len(overlap)}"
    )


if __name__ == "__main__":
    main()
