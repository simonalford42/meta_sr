#!/usr/bin/env python3
"""Run an authenticated, read-only GitHub code search for PySRRegressor.

GitHub's legacy code-search REST API exposes at most 1,000 results for one
query. This script partitions the query by non-overlapping file-size ranges,
which are supported by the API, and deduplicates the returned files. It reads
authentication from GITHUB_TOKEN/GH_TOKEN or the configured Git credential
helper. The credential is never written to an output file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_ROOT = "https://api.github.com"
SEARCH_URL = f"{API_ROOT}/search/code"
BASE_QUERY = "PySRRegressor"
API_VERSION = "2022-11-28"
MAX_INDEXED_FILE_SIZE = 393_215  # GitHub documents files smaller than 384 KiB.
MAX_RESULTS_PER_QUERY = 1_000
TARGET_RESULTS_PER_PARTITION = 900

# Starting with logarithmic ranges keeps common small source files separated
# from larger notebooks. Any range above the target is split recursively.
INITIAL_SIZE_RANGES = [
    (0, 1_023),
    (1_024, 2_047),
    (2_048, 4_095),
    (4_096, 8_191),
    (8_192, 16_383),
    (16_384, 32_767),
    (32_768, 65_535),
    (65_536, 131_071),
    (131_072, 262_143),
    (262_144, MAX_INDEXED_FILE_SIZE),
]


def get_token() -> tuple[str, str]:
    for variable in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.environ.get(variable)
        if token:
            return token, variable

    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        fields = dict(
            line.split("=", 1)
            for line in result.stdout.splitlines()
            if "=" in line
        )
        if fields.get("password"):
            return fields["password"], "git-credential-helper"

    raise RuntimeError(
        "No GitHub credential found. Set GITHUB_TOKEN/GH_TOKEN or configure "
        "a credential helper for github.com."
    )


class GitHubClient:
    def __init__(self, token: str):
        self.token = token
        self.search_requests = 0

    def _curl_json(
        self,
        url: str,
        parameters: dict[str, str] | None = None,
        attempts: int = 4,
    ) -> dict[str, Any]:
        config_lines = [
            f'url = "{url}"',
            'header = "Accept: application/vnd.github+json"',
            f'header = "Authorization: Bearer {self.token}"',
            f'header = "X-GitHub-Api-Version: {API_VERSION}"',
            "silent",
            "show-error",
            "fail-with-body",
        ]
        if parameters:
            config_lines.append("get")
            for key, value in parameters.items():
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                config_lines.append(f'data-urlencode = "{key}={escaped}"')
        config = "\n".join(config_lines) + "\n"

        for attempt in range(1, attempts + 1):
            result = subprocess.run(
                ["curl", "--config", "-"],
                input=config,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                return json.loads(result.stdout)

            try:
                error_payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                error_payload = {}
            error_message = str(error_payload.get("message", "")).lower()
            if "rate limit" in error_message:
                raise GitHubRateLimitError(error_payload.get("message", "rate limit"))

            if attempt == attempts:
                message = result.stderr.strip() or result.stdout[:500]
                raise RuntimeError(f"GitHub request failed after retries: {message}")
            delay = min(30, 2**attempt)
            print(f"GitHub request failed; retrying in {delay}s", flush=True)
            time.sleep(delay)

        raise AssertionError("unreachable")

    def rate(self) -> dict[str, Any]:
        payload = self._curl_json(f"{API_ROOT}/rate_limit")
        return payload["resources"]["code_search"]

    def wait_for_search_slot(self) -> None:
        while True:
            rate = self.rate()
            if rate["remaining"] > 0:
                return
            remaining = max(1, int(rate["reset"]) - int(time.time()) + 2)
            wait = min(30, remaining)
            print(
                f"GitHub code-search rate limit exhausted; waiting {wait}s "
                f"({remaining}s to reset)",
                flush=True,
            )
            time.sleep(wait)

    def wait_for_reset(self) -> None:
        rate = self.rate()
        while True:
            if rate["remaining"] > 0:
                return
            remaining = max(1, int(rate["reset"]) - int(time.time()) + 2)
            if remaining <= 1:
                return
            wait = min(30, remaining)
            print(
                f"GitHub rejected a raced rate-limit slot; waiting {wait}s "
                f"({remaining}s to reset)",
                flush=True,
            )
            time.sleep(wait)
            rate = self.rate()

    def search(self, query: str, per_page: int, page: int = 1) -> dict[str, Any]:
        while True:
            self.wait_for_search_slot()
            self.search_requests += 1
            try:
                payload = self._curl_json(
                    SEARCH_URL,
                    {
                        "q": query,
                        "per_page": str(per_page),
                        "page": str(page),
                    },
                )
                break
            except GitHubRateLimitError:
                self.wait_for_reset()
        if "items" not in payload:
            raise RuntimeError(
                "Unexpected GitHub search response: "
                + json.dumps(payload, sort_keys=True)[:1_000]
            )
        return payload


class GitHubRateLimitError(RuntimeError):
    pass


def size_query(low: int, high: int) -> str:
    return f"{BASE_QUERY} size:{low}..{high}"


def plan_partitions(
    client: GitHubClient,
    low: int,
    high: int,
    observations: list[dict[str, Any]],
) -> list[tuple[int, int, int]]:
    query = size_query(low, high)
    payload = client.search(query, per_page=1)
    count = int(payload["total_count"])
    observations.append(
        {
            "phase": "plan",
            "query": query,
            "low": low,
            "high": high,
            "reported_total_count": count,
            "incomplete_results": bool(payload.get("incomplete_results")),
        }
    )
    print(f"Planned {low}..{high} bytes: reported {count} files", flush=True)

    if count <= TARGET_RESULTS_PER_PARTITION:
        return [(low, high, count)]
    if low == high:
        raise RuntimeError(
            f"More than {MAX_RESULTS_PER_QUERY} matches have the exact size {low}; "
            "size partitioning cannot retrieve all results."
        )

    middle = (low + high) // 2
    return plan_partitions(client, low, middle, observations) + plan_partitions(
        client, middle + 1, high, observations
    )


def fetch_partition(
    client: GitHubClient,
    low: int,
    high: int,
    planned_count: int,
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    query = size_query(low, high)
    items: list[dict[str, Any]] = []
    page = 1
    latest_count = planned_count

    while page <= 10:
        payload = client.search(query, per_page=100, page=page)
        latest_count = int(payload["total_count"])
        page_items = payload["items"]
        observations.append(
            {
                "phase": "fetch",
                "query": query,
                "low": low,
                "high": high,
                "page": page,
                "reported_total_count": latest_count,
                "returned_items": len(page_items),
                "incomplete_results": bool(payload.get("incomplete_results")),
            }
        )
        print(
            f"Fetched {low}..{high} page {page}: {len(page_items)} files "
            f"(reported total {latest_count})",
            flush=True,
        )
        if not page_items:
            break
        items.extend(page_items)
        page += 1

    if page > 10 and page_items:
        raise RuntimeError(
            f"Partition {low}..{high} reached GitHub's 1,000-result cap; "
            "lower TARGET_RESULTS_PER_PARTITION and rerun."
        )
    return items


def write_outputs(output_dir: Path) -> None:
    started_at = datetime.now(timezone.utc)
    start_clock = time.monotonic()
    token, auth_source = get_token()
    client = GitHubClient(token)

    unqualified = client.search(BASE_QUERY, per_page=1)
    unqualified_count = int(unqualified["total_count"])
    observations: list[dict[str, Any]] = [
        {
            "phase": "unqualified",
            "query": BASE_QUERY,
            "reported_total_count": unqualified_count,
            "incomplete_results": bool(unqualified.get("incomplete_results")),
        }
    ]
    print(f"Unqualified query reported {unqualified_count} files", flush=True)

    partitions: list[tuple[int, int, int]] = []
    for low, high in INITIAL_SIZE_RANGES:
        partitions.extend(plan_partitions(client, low, high, observations))

    raw_items: list[dict[str, Any]] = []
    partition_summaries: list[dict[str, Any]] = []
    for low, high, planned_count in partitions:
        items = fetch_partition(
            client, low, high, planned_count, observations
        )
        raw_items.extend(items)
        partition_summaries.append(
            {
                "query": size_query(low, high),
                "low": low,
                "high": high,
                "planned_count": planned_count,
                "retrieved_count": len(items),
            }
        )

    public_raw_items = [
        item for item in raw_items if not item["repository"].get("private", False)
    ]
    private_items_omitted = len(raw_items) - len(public_raw_items)
    unique_items: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in public_raw_items:
        key = (item["repository"]["full_name"], item["path"], item["sha"])
        unique_items[key] = item

    repositories: dict[str, dict[str, Any]] = {}
    file_counts = Counter()
    for item in unique_items.values():
        name = item["repository"]["full_name"]
        repositories[name] = item["repository"]
        file_counts[name] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    completed_at = datetime.now(timezone.utc)
    elapsed = time.monotonic() - start_clock

    with (output_dir / "github_matched_files.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ["repository", "path", "sha", "url"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in sorted(unique_items, key=lambda value: (value[0].casefold(), value[1])):
            item = unique_items[key]
            writer.writerow(
                {
                    "repository": item["repository"]["full_name"],
                    "path": item["path"],
                    "sha": item["sha"],
                    "url": item["html_url"],
                }
            )

    with (output_dir / "github_repositories.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "repository",
            "url",
            "description",
            "owner_type",
            "fork",
            "private",
            "matching_file_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name in sorted(repositories, key=str.casefold):
            repository = repositories[name]
            writer.writerow(
                {
                    "repository": name,
                    "url": repository["html_url"],
                    "description": repository.get("description") or "",
                    "owner_type": repository["owner"].get("type", ""),
                    "fork": repository.get("fork", ""),
                    "private": repository.get("private", ""),
                    "matching_file_count": file_counts[name],
                }
            )

    snapshot = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "api": SEARCH_URL,
        "api_version": API_VERSION,
        "auth_source": auth_source,
        "auth_identity": "not stored; validate separately with GET /user",
        "base_query": BASE_QUERY,
        "unqualified_reported_total_count": unqualified_count,
        "max_results_per_query": MAX_RESULTS_PER_QUERY,
        "target_results_per_partition": TARGET_RESULTS_PER_PARTITION,
        "indexed_size_range_bytes": [0, MAX_INDEXED_FILE_SIZE],
        "partition_strategy": (
            "Non-overlapping file-size ranges; recursively split when the reported "
            "count exceeds the target."
        ),
        "partitions": partition_summaries,
        "observations": observations,
        "raw_retrieved_items": len(raw_items),
        "private_items_omitted": private_items_omitted,
        "private_repository_names_stored": False,
        "unique_retrieved_files": len(unique_items),
        "duplicate_public_items": len(public_raw_items) - len(unique_items),
        "unique_repositories": len(repositories),
        "search_requests": client.search_requests,
        "documented_scope_limits": [
            "Only default branches are indexed.",
            "Archived repositories are not searchable.",
            "Files must be smaller than 384 KiB.",
            "Forks are excluded unless they satisfy GitHub indexing rules and fork:true is used; this query mirrors the user's unqualified query.",
            "GitHub reports approximate/inconsistent total_count values across related queries; retrieved files are deduplicated by repository, path, and blob SHA.",
            "Authenticated search may include private repositories; private results are counted only in aggregate and omitted before writing repository or file rows.",
        ],
    }
    (output_dir / "github_search_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"Wrote {len(unique_items)} unique matching files in "
        f"{len(repositories)} repositories after {client.search_requests} search "
        f"requests ({elapsed:.0f}s).",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    args = parser.parse_args()
    write_outputs(args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
