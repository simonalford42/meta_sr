#!/usr/bin/env python3
"""Parse OpenEvolve SLURM logs and plot GT score over time with ancestry."""

import re
import sys
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def parse_log(filepath):
    """Extract iteration data, parent relationships, and new-best markers."""
    entries = []  # (timestamp, iteration, program_id, parent_id, gt_score, is_new_best)
    new_bests = set()

    # Track "New best program" lines
    best_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - .*New best program (\S+) replaces"
    )
    # Track "Evaluated program" lines (includes initial program at iteration 0)
    eval_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - .*Evaluated program (\S+) in [\d.]+s: combined_score=([\d.]+).*avg_gt=([\d.]+)"
    )
    # Track "Iteration N: Program X (parent: Y)" lines
    iter_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - .*Iteration (\d+): Program (\S+) \(parent: (\S+)\) completed"
    )

    # First pass: find all new bests
    with open(filepath) as f:
        for line in f:
            m = best_pattern.search(line)
            if m:
                new_bests.add(m.group(2))

    # Second pass: build entries
    program_scores = {}
    program_timestamps = {}
    program_parents = {}
    iteration_map = {}  # program_id -> iteration

    with open(filepath) as f:
        for line in f:
            m = eval_pattern.search(line)
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                pid = m.group(2)
                gt = float(m.group(4))
                program_scores[pid] = gt
                program_timestamps[pid] = ts

            m = iter_pattern.search(line)
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                iteration = int(m.group(2))
                pid = m.group(3)
                parent = m.group(4)
                iteration_map[pid] = iteration
                program_parents[pid] = parent
                program_timestamps[pid] = ts  # use iteration completion time

    # Build sorted entries by completion time
    # Initial program (iteration 0)
    for pid in program_scores:
        if pid not in iteration_map:
            # This is the initial program
            entries.append({
                'timestamp': program_timestamps[pid],
                'iteration': 0,
                'program_id': pid,
                'parent_id': None,
                'gt_score': program_scores[pid],
                'is_new_best': True,  # initial is always best
            })

    # Evolved programs
    for pid, iteration in sorted(iteration_map.items(), key=lambda x: program_timestamps.get(x[0], datetime.min)):
        entries.append({
            'timestamp': program_timestamps[pid],
            'iteration': iteration,
            'program_id': pid,
            'parent_id': program_parents.get(pid),
            'gt_score': program_scores.get(pid, 0.0),
            'is_new_best': pid in new_bests,
        })

    # Sort by timestamp
    entries.sort(key=lambda e: e['timestamp'])

    return entries, program_parents


def build_ancestry(program_parents, target_id):
    """Trace back ancestry from target_id to root."""
    chain = [target_id]
    current = target_id
    visited = set()
    while current in program_parents and current not in visited:
        visited.add(current)
        current = program_parents[current]
        chain.append(current)
    return list(reversed(chain))


def plot_all(logs_info):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for ax, (title, filepath) in zip(axes, logs_info):
        entries, parents = parse_log(filepath)

        if not entries:
            ax.set_title(f"{title}\n(no data)")
            continue

        timestamps = [e['timestamp'] for e in entries]
        scores = [e['gt_score'] for e in entries]
        is_best = [e['is_new_best'] for e in entries]

        # Use elapsed hours from start
        t0 = timestamps[0]
        hours = [(t - t0).total_seconds() / 3600 for t in timestamps]

        # Plot all points
        regular_h = [h for h, b in zip(hours, is_best) if not b]
        regular_s = [s for s, b in zip(scores, is_best) if not b]
        best_h = [h for h, b in zip(hours, is_best) if b]
        best_s = [s for s, b in zip(scores, is_best) if b]

        ax.scatter(regular_h, regular_s, alpha=0.3, s=20, c='steelblue', label='Evaluated')
        ax.scatter(best_h, best_s, s=80, c='red', zorder=5, edgecolors='darkred',
                   linewidths=1, label='New Best', marker='*')

        # Running best line
        running_best = []
        current_best = -1
        for s in scores:
            current_best = max(current_best, s)
            running_best.append(current_best)
        ax.plot(hours, running_best, 'r-', alpha=0.5, linewidth=1.5, label='Best so far')

        # Stats
        n_iters = len(entries) - 1  # exclude initial
        best_score = max(scores)
        n_failures = sum(1 for s in scores if s == 0.0)

        ax.set_title(f"{title}\n{n_iters} iters, best GT={best_score:.4f}, {n_failures} failures")
        ax.set_xlabel("Hours elapsed")
        ax.set_ylim(-0.05, max(0.6, best_score + 0.05))
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("GT Score (avg_gt)")
    plt.suptitle("OpenEvolve: GT Score Over Time by Operator Type", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("/home/sca63/meta_sr/openevolve_gt_scores.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot to /home/sca63/meta_sr/openevolve_gt_scores.png")

    # Print ancestry for best program in each run
    print("\n" + "="*80)
    print("ANCESTRY OF BEST PROGRAMS")
    print("="*80)
    for title, filepath in logs_info:
        entries, parents = parse_log(filepath)
        if not entries:
            continue
        # Find best program
        best_entry = max(entries, key=lambda e: e['gt_score'])
        chain = build_ancestry(parents, best_entry['program_id'])

        print(f"\n--- {title} ---")
        print(f"Best program: {best_entry['program_id'][:12]}... (GT={best_entry['gt_score']:.4f})")
        print(f"Ancestry chain ({len(chain)} deep):")

        # Look up scores for each ancestor
        score_map = {e['program_id']: e['gt_score'] for e in entries}
        iter_map = {e['program_id']: e['iteration'] for e in entries}
        for i, pid in enumerate(chain):
            score = score_map.get(pid, '?')
            iteration = iter_map.get(pid, '?')
            prefix = "  " * i + ("└─ " if i > 0 else "")
            print(f"  {prefix}iter={iteration} GT={score} [{pid[:12]}...]")


if __name__ == "__main__":
    logs = [
        ("Mutation (151084)", "/home/sca63/meta_sr/out/151084.out"),
        ("Survival (144409)", "/home/sca63/meta_sr/out/144409.out"),
        ("Selection (144408)", "/home/sca63/meta_sr/out/144408.out"),
    ]
    plot_all(logs)
