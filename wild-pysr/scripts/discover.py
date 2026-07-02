#!/usr/bin/env python3
"""Inventory GitHub repositories indexed by Sourcegraph that mention PySRRegressor.

This script performs read-only public HTTP requests. It does not clone repositories,
execute repository code, or submit jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCEGRAPH_STREAM_URL = "https://sourcegraph.com/.api/search/stream"
REPOSITORY_QUERY = (
    "context:global fork:yes archived:yes select:repo "
    "PySRRegressor count:10000 timeout:2m"
)
FILE_QUERY = (
    "context:global fork:yes archived:yes select:file "
    "PySRRegressor count:10000 timeout:2m"
)


def sourcegraph_search(query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = urllib.parse.urlencode({"q": query, "v": "V3"})
    request = urllib.request.Request(
        f"{SOURCEGRAPH_STREAM_URL}?{params}",
        headers={"User-Agent": "wild-pysr-inventory/1.0"},
    )

    matches: list[dict[str, Any]] = []
    final_progress: dict[str, Any] | None = None
    current_event: str | None = None

    with urllib.request.urlopen(request, timeout=150) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8")
            if line.startswith("event: "):
                current_event = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                payload = json.loads(line.removeprefix("data: "))
                if current_event == "matches":
                    matches.extend(payload)
                elif current_event == "progress" and payload.get("done"):
                    final_progress = payload

    if final_progress is None:
        raise RuntimeError(f"Sourcegraph stream did not finish cleanly for: {query}")
    return matches, final_progress


def write_snapshot(output_dir: Path) -> None:
    start_clock = time.monotonic()
    started_at = datetime.now(timezone.utc)
    repository_matches, repository_progress = sourcegraph_search(REPOSITORY_QUERY)
    file_matches, file_progress = sourcegraph_search(FILE_QUERY)

    repository_metadata = {
        match["repository"]: match
        for match in repository_matches
        if match.get("type") == "repo"
    }
    file_repository_names = {
        match["repository"]
        for match in file_matches
        if match.get("type") == "path"
    }
    repository_names = set(repository_metadata) | file_repository_names
    files_by_repository = Counter(
        match["repository"]
        for match in file_matches
        if match.get("type") == "path"
    )

    if repository_progress.get("skipped"):
        raise RuntimeError(
            "Repository query was not exhaustive: "
            + json.dumps(repository_progress["skipped"], sort_keys=True)
        )
    missing_file_repositories = sorted(set(repository_metadata) - set(files_by_repository))
    if missing_file_repositories:
        raise RuntimeError(
            "File query omitted repositories: " + ", ".join(missing_file_repositories)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    completed_at = datetime.now(timezone.utc)
    snapshot = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "source": SOURCEGRAPH_STREAM_URL,
        "repository_query": REPOSITORY_QUERY,
        "file_query": FILE_QUERY,
        "repository_progress": repository_progress,
        "file_progress": file_progress,
        "scope_note": (
            "This is exhaustive for the stated Sourcegraph query at this time, not for "
            "all public GitHub repositories. GitHub code search required authentication."
        ),
        "inventory_rule": (
            "Union repository names returned by the select:repo and select:file queries. "
            "The independent queries can observe slightly different index state."
        ),
        "repository_metadata_missing": sorted(
            file_repository_names - set(repository_metadata)
        ),
        "repositories": [
            repository_metadata.get(
                name,
                {
                    "type": "repo",
                    "repository": name,
                    "metadata_missing": True,
                },
            )
            for name in sorted(repository_names)
        ],
    }
    (output_dir / "search_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (output_dir / "repositories.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "repository",
            "url",
            "sourcegraph_stars",
            "sourcegraph_last_fetched",
            "description",
            "topics",
            "matching_file_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for name in sorted(repository_names, key=str.casefold):
            match = repository_metadata.get(name, {})
            github_name = name.removeprefix("github.com/")
            writer.writerow(
                {
                    "repository": github_name,
                    "url": f"https://github.com/{github_name}",
                    "sourcegraph_stars": match.get("repoStars", ""),
                    "sourcegraph_last_fetched": match.get("repoLastFetched", ""),
                    "description": match.get("description", ""),
                    "topics": ";".join(match.get("topics", [])),
                    "matching_file_count": files_by_repository[name],
                }
            )

    with (output_dir / "matched_files.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["repository", "path", "url"],
            lineterminator="\n",
        )
        writer.writeheader()
        rows = sorted(
            (
                match["repository"].removeprefix("github.com/"),
                match["path"],
            )
            for match in file_matches
            if match.get("type") == "path"
        )
        for repository, path in rows:
            quoted_path = urllib.parse.quote(path, safe="/")
            writer.writerow(
                {
                    "repository": repository,
                    "path": path,
                    "url": f"https://github.com/{repository}/blob/HEAD/{quoted_path}",
                }
            )

    elapsed = time.monotonic() - start_clock
    print(
        f"Wrote {len(repository_names)} repositories and {len(file_matches)} matching "
        f"file paths to {output_dir} (completed {completed_at.isoformat()}; "
        f"elapsed={elapsed:.0f}s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    args = parser.parse_args()
    write_snapshot(args.output_dir)


if __name__ == "__main__":
    main()
