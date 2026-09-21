"""Offline concurrency reconstruction from saved sacct output and run logs.

Usage: python scripts/analyze_slurm_run_load.py SNAPSHOT_DIR RUN_ID ...
The accounting.psv columns must match the collection command in the report.
This script makes no Slurm calls. Peaks use whole-second allocation intervals;
allocated memory/CPU are reservations, not measured utilization.
"""
import collections
import datetime as dt
import json
from pathlib import Path
import re
import sys


def timestamp(value):
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def task_count(job_id):
    if '_[' not in job_id:
        return 1
    spec = job_id.split('_[', 1)[1].split(']', 1)[0].split('%', 1)[0]
    count = 0
    for part in spec.split(','):
        if '-' in part:
            first, last = part.split('-')
            count += int(last) - int(first) + 1
        else:
            count += 1
    return count


def peak(rows, cutoff):
    events = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        start = timestamp(row[5])
        end = timestamp(row[6]) or cutoff
        if start is None or end <= start:
            continue
        cpus = int(row[7])
        events[start][0] += 1
        events[start][1] += cpus
        events[end][0] -= 1
        events[end][1] -= cpus
    active = cpus = 0
    best = {'tasks': 0, 'cpus': 0, 'at': None}
    for when, (delta, cpu_delta) in sorted(events.items()):
        active += delta
        cpus += cpu_delta
        if active > best['tasks']:
            best = {'tasks': active, 'cpus': cpus, 'at': when.isoformat()}
    return best


def main():
    folder = Path(sys.argv[1])
    repo = Path(__file__).resolve().parents[1]
    cutoff = dt.datetime.fromtimestamp((folder / 'accounting.psv').stat().st_mtime)
    records = [line.split('|') for line in (folder / 'accounting.psv').read_text().splitlines()]
    output = {'snapshot_local_time': cutoff.isoformat(), 'runs': {}}
    all_rows = []
    all_intervals = []
    for run in sys.argv[2:]:
        log = (repo / 'runs' / run / 'run.log').read_text()
        ids = set(re.findall(r'Submitted (?:SLURM job array|retry job): (\d+)', log))
        rows = [r for r in records if r[1].split('_')[0] in ids]
        all_rows.extend(rows)
        active = collections.defaultdict(collections.Counter)
        for row in rows:
            if row[3] in ('RUNNING', 'PENDING', 'COMPLETING'):
                active[row[1].split('_')[0]][row[3]] += task_count(row[1])
        intervals = []
        for jid in ids:
            members = [r for r in rows if r[1].split('_')[0] == jid]
            if not members:
                continue
            submitted = min(r[4] for r in members)
            ends = [timestamp(r[6]) or cutoff for r in members]
            intervals.append(['', '', '', '', '', submitted, max(ends).isoformat(), '1'])
        all_intervals.extend(intervals)
        states = collections.Counter()
        for row in rows:
            states[row[3]] += task_count(row[1])
        output['runs'][run] = {
            'submitted_arrays': len(ids),
            'accounting_task_records': len(rows),
            'states': dict(states),
            'peak_running_allocations': peak(rows, cutoff),
            'peak_outstanding_arrays': peak(intervals, cutoff)['tasks'],
            'allocated_cpu_hours': sum(int(r[10]) for r in rows) / 3600,
            'requested_memory_values': sorted(set(r[8] for r in rows)),
            'active_arrays': active,
        }
    output['combined_peak_running_allocations'] = peak(all_rows, cutoff)
    output['combined_peak_outstanding_arrays'] = peak(all_intervals, cutoff)
    rendered = json.dumps(output, indent=2)
    (folder / 'summary.json').write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
