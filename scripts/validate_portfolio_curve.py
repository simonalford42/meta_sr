#!/usr/bin/env python3
"""Validate reconstructed portfolio curves against saved restart histories."""
import csv
import json
import math
from collections import Counter

from analyze_portfolio_solve_over_time import ROOT, index_inputs, write_json


def identity(record):
    return tuple(record[k] for k in ('dataset', 'method', 'noise', 'seed'))


def main():
    output = ROOT / 'reports/portfolio_solve_over_time'
    records = json.loads((output / 'first_recovery.json').read_text())
    specs = [s for group in index_inputs().values() for s in group]
    expected = {identity(s): s for s in specs}
    assert len(records) == len(expected) == 10640
    assert Counter(map(identity, records)) == Counter(expected.keys())
    counts = Counter((r['method'], r['noise']) for r in records)
    assert len(counts) == 8 and set(counts.values()) == {1330}
    recoveries = 0
    for record in records:
        assert record['status'] == 'complete'
        raw = json.loads(open(expected[identity(record)]['path']).read())
        portfolio = raw['portfolio']
        restarts = portfolio['restarts']
        durations = [r['search_runtime_seconds'] for r in restarts]
        assert all(math.isfinite(t) and t > 0 for t in durations)
        assert math.isclose(sum(durations), record['total_search_seconds'], abs_tol=1e-7)
        assert record['restart_count'] == len(restarts)
        assert record['final_merged_solved'] == (raw.get('gt_match_score') == 1)
        recovered = record['first_solve_seconds'] is not None
        assert recovered or not record['final_merged_solved']
        if not recovered:
            assert record['first_solve_restart'] is None
            assert record['first_solve_budget_seconds'] is None
            continue
        recoveries += 1
        number = record['first_solve_restart']
        assert 1 <= number <= len(restarts)
        restart = restarts[number - 1]
        assert restart['restart_index'] + 1 == number and not restart.get('error')
        elapsed = sum(durations[:number])
        assert math.isclose(elapsed, record['first_solve_seconds'], abs_tol=1e-7)
        assert record['first_solve_budget_seconds'] == min(elapsed, 900)
        assert any(row['equation'] == record['first_solve_equation'] and
                   row.get('r2') is not None and math.isfinite(row['r2']) and row['r2'] >= .5
                   for row in restart['pareto_frontier'])
    rows = list(csv.DictReader((output / 'solve_rate.csv').open()))
    assert len(rows) == 160
    previous = {}
    seen = set()
    for row in rows:
        noise = row['noise'] if row['noise'] == 'all' else float(row['noise'])
        minute = int(row['minutes'])
        key = (noise, row['method'])
        assert (key, minute) not in seen
        seen.add((key, minute))
        subset = [r for r in records if r['method'] == row['method'] and
                  (noise == 'all' or r['noise'] == noise)]
        solved = sum(r['first_solve_budget_seconds'] is not None and
                     r['first_solve_budget_seconds'] <= minute * 60 for r in subset)
        assert int(row['solved']) == solved and int(row['expected']) == len(subset)
        assert math.isclose(float(row['percent_solved']), 100 * solved / len(subset))
        assert solved >= previous.get(key, 0)
        previous[key] = solved
    audit = {'trials': len(records), 'recovered_trials': recoveries,
             'method_noise_cells': len(counts), 'trials_per_cell': 1330,
             'csv_rows': len(rows), 'original_positives_lost': 0,
             'checks': 'identities, source status, restart timing, matching frontier membership, R2 gate, CSV counts and monotonicity'}
    write_json(output / 'curve_validation.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
