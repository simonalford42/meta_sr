#!/usr/bin/env python3
"""Score every GitHub-discovered repository for manual science review.

This is a transparent prioritization heuristic, not a scientific classification.
The output preserves one row per repository and explains every score component.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


DOMAIN_PATTERNS = {
    "astronomy_cosmology": re.compile(
        r"astro|cosmo|galax|black.?hole|supernova|snia|cmb|lensing|lya|"
        r"stellar|planet|solar|lunar|gravit|exoplanet|neutron.?star|reioniz",
        re.I,
    ),
    "physics": re.compile(
        r"physics|physical|fluid|turbulen|rayleigh|lorenz|dynamical|dynamics|"
        r"quantum|plasma|thermo|wave|pendulum|nuclear|magnetopause|detector",
        re.I,
    ),
    "biology_chemistry": re.compile(
        r"biolog|bio\b|chem|molecul|protein|peptide|gene|genom|metabol|"
        r"bioprocess|ecolog|forest|kelp|medical|health|drug|cell.?type|crispr",
        re.I,
    ),
    "engineering": re.compile(
        r"engineering|battery|corrosion|material|circuit|reactor|control|"
        r"catenary|robot|energy|combust|manufactur|concrete|beam|tether|"
        r"aero|power.?grid|engine",
        re.I,
    ),
    "earth_climate": re.compile(
        r"climate|weather|ocean|earth|geolog|hydro|aquifer|atmospher|"
        r"environment|cloud.?cover|wildfire|alkalinity",
        re.I,
    ),
}

RESEARCH_SIGNAL = re.compile(
    r"paper|publication|reproduc|thesis|research|experimental|observational|"
    r"empirical|real.?world|data|analysis|study|scientific",
    re.I,
)
METHOD_SIGNAL = re.compile(
    r"benchmark|symbolic.?regression|pysr|equation.?discovery|framework|"
    r"library|toolkit|package|method|algorithm|baseline",
    re.I,
)
EDUCATION_SIGNAL = re.compile(
    r"tutorial|course|lecture|homework|school|workshop|demo|example",
    re.I,
)
AGENT_SIGNAL = re.compile(r"\bllm\b|language.?model|agent|prompt|skillbank", re.I)
NONPRIMARY_PATH = re.compile(
    r"(^|/)(tests?|docs?|examples?|tutorials?|benchmarks?|external|vendor|"
    r"skillbank|coverage|\.ipynb_checkpoints)(/|$)",
    re.I,
)
CODE_EXTENSIONS = {".py", ".ipynb", ".jl", ".r"}
ARTIFACT_EXTENSIONS = {".json", ".md", ".txt", ".html", ".csv", ".yml", ".yaml"}


def score_repository(
    repository: dict[str, str], paths: list[str]
) -> dict[str, str | int]:
    name = repository["repository"]
    description = repository["description"]
    searchable = " ".join([name, description, *paths])
    domains = [
        label for label, pattern in DOMAIN_PATTERNS.items() if pattern.search(searchable)
    ]
    code_paths = [path for path in paths if Path(path).suffix.lower() in CODE_EXTENSIONS]
    primary_code_paths = [path for path in code_paths if not NONPRIMARY_PATH.search(path)]
    artifact_only = bool(paths) and all(
        Path(path).suffix.lower() in ARTIFACT_EXTENSIONS for path in paths
    )

    score = 0
    reasons: list[str] = []
    if domains:
        contribution = min(8, 4 * len(domains))
        score += contribution
        reasons.append(f"domain_keywords:+{contribution}")
    if RESEARCH_SIGNAL.search(searchable):
        score += 3
        reasons.append("research_signal:+3")
    if primary_code_paths:
        score += 3
        reasons.append("primary_code_match:+3")
    elif code_paths:
        score += 1
        reasons.append("nonprimary_code_match:+1")
    if len(paths) >= 3:
        score += 1
        reasons.append("multiple_matches:+1")
    if METHOD_SIGNAL.search(" ".join([name, description])):
        score -= 3
        reasons.append("method_or_benchmark_signal:-3")
    if EDUCATION_SIGNAL.search(searchable):
        score -= 4
        reasons.append("education_signal:-4")
    if AGENT_SIGNAL.search(" ".join([name, description])):
        score -= 3
        reasons.append("agent_or_llm_signal:-3")
    if artifact_only:
        score -= 6
        reasons.append("artifact_only:-6")

    if score >= 9:
        priority = "high"
    elif score >= 6:
        priority = "medium"
    else:
        priority = "deprioritized"

    return {
        "repository": name,
        "url": repository["url"],
        "description": description,
        "matching_file_count": len(paths),
        "science_likelihood_score": score,
        "triage_priority": priority,
        "domain_signals": ";".join(domains),
        "score_reasons": ";".join(reasons),
        "artifact_only": str(artifact_only).lower(),
        "code_match_count": len(code_paths),
        "primary_code_match_count": len(primary_code_paths),
        "sample_matching_paths": ";".join(paths[:5]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    args = parser.parse_args()

    with (args.data_dir / "github_repositories.csv").open(encoding="utf-8") as handle:
        repositories = list(csv.DictReader(handle))
    paths_by_repository: dict[str, list[str]] = defaultdict(list)
    with (args.data_dir / "github_matched_files.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            paths_by_repository[row["repository"]].append(row["path"])

    rows = [
        score_repository(repository, sorted(paths_by_repository[repository["repository"]]))
        for repository in repositories
    ]
    rows.sort(
        key=lambda row: (
            -int(row["science_likelihood_score"]),
            str(row["repository"]).casefold(),
        )
    )

    output = args.data_dir / "github_triage.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["triage_priority"])] += 1
    print(f"Wrote {len(rows)} rows to {output}: {dict(counts)}")


if __name__ == "__main__":
    main()
