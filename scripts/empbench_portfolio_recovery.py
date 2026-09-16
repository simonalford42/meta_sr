#!/usr/bin/env python3
"""Terra binary-search reviews of EmpiricalBench first-fit and restart frontiers.

Default is offline preparation. --run explicitly enables paid API requests.
Like the previous analysis, assumes recovery persists on cumulative Pareto
frontiers; final-negative trials are not searched for transient recoveries.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frontier_aggregation import merge_frontiers
from manual_solve_check import (TARGETS, RUBRIC, REVIEW_SCHEMA, build_request, estimate_cost_upper,
    OpenRouterHTTPClient, write_json_atomic as write, calculate_cost,
    _extract_output_text, _validate_review)

MODEL = 'openai/gpt-5.6-terra'
MAX_TOKENS = 3000
MAX_COST = 15.0
OUT = ROOT / 'runs/empiricalbench_9-16_portfolio90s_terra_recovery'
RUNS = {}
EXTRA = '''\nFitted constants are free ONLY where the dataset reference permits them.
Fixed powers and relative coefficients must match exactly. Never round them to theoretical values.
For Rydberg, C-log(1/x0^2-q/x1^2) is not exact unless q is exactly 1.
For Newton and ideal gas, fixed relative logarithmic coefficients cannot vary.
For Planck, a constant, an input variable, or a Wien approximation is not exact.
For empirical Bode, the target is log(c0+c1*exp(c2*n)); judge the log-transformed family.
A fitted affine transformation is allowed only where the reference permits it.
'''


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compact(rows):
    return [dict(frontier_index=i, complexity=r['complexity'], equation=r['equation'])
            for i, r in enumerate(rows)]


def make_snapshots(result):
    """Keep actual observation times; missing reads are explicit, never backfilled."""
    portfolio = result['portfolio']
    if result.get('error') or not portfolio or not portfolio['warmup_excluded_from_budget']:
        raise ValueError('Need successful portfolios with excluded warm-up')
    restarts = portfolio['restarts']
    if not restarts:
        raise ValueError('No restarts')
    snapshots, missing = [], []
    first = restarts[0]
    trace = first.get('execution_trace') or []
    if not trace:
        raise ValueError('Missing first-restart snapshots')
    seen = set()
    for observation in sorted(trace, key=lambda o: o['elapsed_seconds']):
        scheduled = observation.get('scheduled_seconds')
        if observation.get('final') or scheduled is None or scheduled > 90:
            continue
        seen.add(scheduled)
        if observation.get('status') != 'ok' or not observation.get('equations'):
            missing.append({'scheduled_seconds': scheduled, 'status': observation.get('status')})
            continue
        snapshots.append({'seconds': observation['elapsed_seconds'], 'scheduled_seconds': scheduled,
            'kind': 'first_restart_snapshot', 'frontier': compact(observation['equations'])})
    missing.extend({'scheduled_seconds': t, 'status': 'missing'} for t in range(5, 91, 5) if t not in seen)
    cumulative, elapsed = [], 0.0
    for index, restart in enumerate(restarts):
        duration = float(restart['search_runtime_seconds'])
        if not math.isfinite(duration) or duration < 0 or restart.get('error'):
            raise ValueError('Invalid or failed restart')
        elapsed += duration
        cumulative = merge_frontiers([cumulative, restart.get('pareto_frontier') or []])
        if not cumulative:
            raise ValueError('Empty restart frontier')
        snapshots.append({'seconds': elapsed, 'kind': 'restart_end', 'restart_index': index,
                          'frontier': compact(cumulative)})
    if compact(result['frontier']) != compact(cumulative):
        raise ValueError('Reconstructed final frontier differs from saved aggregate')
    if any(a['seconds'] > b['seconds'] for a, b in zip(snapshots, snapshots[1:])):
        raise ValueError('Snapshot and restart clocks are inconsistent')
    return snapshots, missing


def initialize():
    fingerprints = {label: digest(json.loads((path/'empbench_results.json').read_text()))
                    for label, path in RUNS.items()}
    config = {'model': MODEL, 'max_tokens': MAX_TOKENS, 'extra': EXTRA,
              'targets': TARGETS, 'rubric': RUBRIC, 'schema': REVIEW_SCHEMA, 'run_paths': {k: str(v.resolve()) for k, v in RUNS.items()}}
    if (OUT/'state.json').exists():
        state = json.loads((OUT/'state.json').read_text())
        if state['fingerprints'] != fingerprints or state['config_hash'] != digest(config):
            raise ValueError('Sources or reviewer configuration changed; use a new output directory')
        return
    trials = []
    for method, path in RUNS.items():
        data = json.loads((path/'empbench_results.json').read_text())
        records = data['runs']
        datasets = data['protocol']['datasets']
        if len(datasets) != 9 or len(records) != 90 or len({(r['dataset'], r['seed']) for r in records}) != 90:
            raise ValueError('Expected all nine EmpiricalBench problems, ten seeds each')
        for dataset in datasets:
            if not dataset.startswith('empirical_') or dataset not in TARGETS:
                raise ValueError(dataset)
            if {r['seed'] for r in records if r['dataset'] == dataset} != set(range(10000,10010)):
                raise ValueError('Incomplete seed grid')
        for result in sorted(records, key=lambda r: (r['dataset'], r['seed'])):
            snapshots, missing = make_snapshots(result)
            trial = {'id': len(trials), 'method': method, 'dataset': result['dataset'],
                     'seed': result['seed'], 'n_snapshots': len(snapshots),
                     'budget_seconds': result['portfolio']['total_search_budget_seconds'],
                     'missing_snapshots': missing, 'low': 0, 'high': len(snapshots), 'history': []}
            write(OUT/'snapshots'/f"{trial['id']:03d}.json", snapshots)
            trials.append(trial)
    write(OUT/'state.json', {'trials': trials, 'cache': {}, 'round': 0, 'cost_usd': 0,
                           'fingerprints': fingerprints, 'config_hash': digest(config)})


def overrides():
    path = OUT/'review_overrides.json'
    return json.loads(path.read_text()) if path.exists() else {}


def known_exact_equations():
    known = {}
    for path in (OUT/'rounds').glob('*/reviews.json'):
        items = json.loads((path.parent/'items.json').read_text())
        for key, original in json.loads(path.read_text()).items():
            review = {**original, **overrides().get(key, {})}
            if review['classification'] == 'exact':
                known[items[key]['dataset'], review['matching_equation']] = key
    return known


def apply(trial, midpoint, review, key):
    positive = review['classification'] == 'exact'
    trial['history'].append({'snapshot_count': midpoint, 'positive': positive, 'cache_key': key})
    trial['high' if positive else 'low'] = midpoint


def step(dry_run=False):
    initialize()
    state = json.loads((OUT/'state.json').read_text())
    round_dir = OUT/'rounds'/f"{state['round']:02d}"
    batch_path=round_dir/'batch.json'
    if batch_path.exists() and dry_run:
        print('Pending batch exists; dry-run will not contact the API')
        return
    if batch_path.exists():
        saved=json.loads(batch_path.read_text())
        client=OpenRouterHTTPClient(os.environ['OPENROUTER_API_KEY'])
        batch=client.json_request('GET',f"/beta/batches/{saved['id']}")
        print('Round',state['round'],batch.get('status'),batch.get('request_counts'),flush=True)
        if batch.get('status')!='completed':
            if batch.get('status') in ('failed','expired','cancelled'): raise RuntimeError(batch['status'])
            return
        write(round_dir/'responses.json',batch)
        items=json.loads((round_dir/'items.json').read_text())
        envelopes={r['custom_id']:r for r in batch['results']}
        reviewed={}
        cost=0
        for key,item in items.items():
            envelope=envelopes[key]
            response=envelope.get('response') or {}
            body=response.get('body') or {}
            assert not envelope.get('error') and response.get('status_code',200)==200, key
            review=json.loads(_extract_output_text(body))
            _validate_review(review,len(item['frontier']))
            assert review['classification'] not in ('error','not_applicable','phenomenological_match'), key
            if review['classification']=='exact':
                assert review['matching_equation'] in [r['equation'] for r in item['frontier']]
                assert review['best_frontier_indices']
                assert any(item['frontier'][i]['equation']==review['matching_equation'] for i in review['best_frontier_indices'])
            review.update(usage=body.get('usage',{}),cost_usd=calculate_cost(body.get('usage',{}),MODEL),source='llm')
            reviewed[key]=review
            cost+=review['cost_usd']
        write(round_dir/'reviews.json',reviewed)
        state['cache'].update(reviewed)
        state['cost_usd']+=cost
        state['round']+=1
        print(f"Round cost ${cost:.4f}; cumulative ${state['cost_usd']:.4f}",flush=True)
        write(OUT/'state.json',state)
        round_dir=OUT/'rounds'/f"{state['round']:02d}"
    state['cache'] = {k:v for k,v in state['cache'].items() if v.get('source') != 'known_exact_equation'}
    for key,correction in overrides().items():
        state['cache'][key]={**state['cache'][key],**correction}
    # Reconstruct the path from cached answers so an audited correction cannot
    # leave a stale binary-search bound from an earlier round.
    for t in state['trials']:
        t.update(low=0,high=t['n_snapshots'],history=[])
    items={}
    known=known_exact_equations()
    for t in state['trials']:
        snapshots=json.loads((OUT/'snapshots'/f"{t['id']:03d}.json").read_text())
        final_key = digest([t['dataset'], snapshots[-1]['frontier']])
        if final_key not in state['cache']:
            items[final_key] = {'custom_id': final_key, 'source_hash': final_key,
                'dataset': t['dataset'], 'seed': t['seed'], 'noise': 0,
                'frontier': snapshots[-1]['frontier']}
            continue
        final = state['cache'][final_key]
        t['final_classification'] = final['classification']
        t['final_positive'] = final['classification'] == 'exact'
        if not t['final_positive']:
            continue
        while t['high']-t['low']>1:
            midpoint=(t['high']+t['low'])//2
            frontier=snapshots[midpoint-1]['frontier']
            key=digest([t['dataset'],frontier])
            if key not in state['cache']:
                matches=[r for r in frontier if (t['dataset'],r['equation']) in known]
                if matches:
                    row=matches[0]
                    state['cache'][key]={
                        'classification':'exact','matching_equation':row['equation'],
                        'best_frontier_indices':[row['frontier_index']],
                        'source':'known_exact_equation','cost_usd':0,
                        'evidence':known[t['dataset'],row['equation']],
                        'explanation':'This identical equation was already judged exact for this dataset.'}
            if key in state['cache']:
                apply(t,midpoint,state['cache'][key],key)
                continue
            items[key]={'custom_id':key,'source_hash':key,'dataset':t['dataset'],
                        'seed':t['seed'],'noise':0,'frontier':frontier}
            break
    write(OUT/'state.json',state)
    if not items:
        render(state)
        print('COMPLETE',flush=True)
        return
    requests=[build_request(item,MODEL,'medium',MAX_TOKENS) for item in items.values()]
    for req in requests: req['body']['messages'][0]['content']+=EXTRA
    estimate=estimate_cost_upper(requests,MODEL,MAX_TOKENS)
    payload={'endpoint':'/v1/chat/completions','model':MODEL,'requests':requests}
    write(round_dir/'items.json',items)
    write(round_dir/'payload.json',payload)
    write(round_dir/'estimate.json',estimate)
    print('Prepared round',state['round'],len(items),'requests;',estimate,flush=True)
    if dry_run: return
    assert state['cost_usd']+estimate['maximum_cost_usd']<=MAX_COST, f'Would exceed ${MAX_COST:g} budget'
    client=OpenRouterHTTPClient(os.environ['OPENROUTER_API_KEY'])
    batch=client.json_request('POST','/beta/batches',payload)
    write(round_dir/'batch.json',{'id':batch['id'],'status':batch.get('status')})
    print('Submitted',batch['id'],flush=True)


def render(state):
    records = []
    for trial in state['trials']:
        record = dict(trial)
        record.update(first_solve_seconds=None, first_solve_budget_seconds=None,
                      last_negative_seconds=None)
        if trial['final_positive']:
            assert trial['high'] - trial['low'] == 1
            snapshots = json.loads((OUT/'snapshots'/f"{trial['id']:03d}.json").read_text())
            positive = snapshots[trial['high']-1]
            record['first_solve_seconds'] = positive['seconds']
            # Retain actual times; budget time clips only the final fit's overshoot.
            record['first_solve_budget_seconds'] = min(positive['seconds'], trial['budget_seconds'])
            record['last_negative_seconds'] = snapshots[trial['low']-1]['seconds'] if trial['low'] else 0
            record['recovery_review'] = state['cache'][digest([trial['dataset'], positive['frontier']])]
            record['first_solve_kind'] = positive['kind']
            record['first_solve_scheduled_seconds'] = positive.get('scheduled_seconds')
        records.append(record)
    write(OUT/'first_recovery.json', records)
    with (OUT/'per_trial.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'dataset', 'seed', 'final_classification',
            'first_solve_seconds', 'first_solve_budget_seconds', 'last_negative_seconds'],
            extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    with (OUT/'solve_rate.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'seconds', 'solved', 'total', 'solve_rate'], lineterminator='\n')
        writer.writeheader()
        for method in RUNS:
            rows = [r for r in records if r['method'] == method]
            for deadline in list(range(0,91,5)) + list(range(180,3601,90)):
                count = sum(r['first_solve_budget_seconds'] is not None and
                            r['first_solve_budget_seconds'] <= deadline for r in rows)
                writer.writerow(dict(method=method, seconds=deadline, solved=count,
                                     total=len(rows), solve_rate=count/len(rows)))
    (OUT/'README.md').write_text(
        '# EmpiricalBench approximate first recovery\n\n'
        'All nine problems, ten seeds, both methods; Terra medium reasoning. '
        'Exact EmpiricalBench family recovery only; near matches excluded. '
        'Final frontiers and binary-search midpoints are reviewed; identical frontiers and previously '
        'accepted identical equations within the same dataset reuse decisions.\n\n'
        'Binary search assumes recovery persists on cumulative native-loss Pareto frontiers. '
        'Final-negative trials are not searched, so transient recoveries can be missed. '
        'An unavailable capture is excluded from the searchable sequence and recorded explicitly, '
        'not assigned an unsolved label.\n\n'
        'The first restart has five-second snapshots; later checkpoints are restart endpoints. '
        'Actual capture/search times exclude the portfolio warm-up and between-fit scoring. '
        'Raw times are preserved; only the budget-time column clips final overshoot to the one-hour budget. '
        'CSV counts use those actual times, not backdated nominal checkpoint times.\n\n'
        f"Recorded review cost at archived batch prices: ${state['cost_usd']:.4f}.\n")


def main():
    global OUT, RUNS, MAX_COST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--evolved', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--max-cost', type=float, default=15)
    parser.add_argument('--run', action='store_true', help='Enable paid requests and resume all rounds')
    args = parser.parse_args()
    if not math.isfinite(args.max_cost) or args.max_cost <= 0:
        parser.error('--max-cost must be positive and finite')
    OUT, MAX_COST = args.output_dir, args.max_cost
    RUNS = {'Baseline': args.baseline, '709715': args.evolved}
    if not args.run:
        step(dry_run=True)
        return
    while True:
        try:
            step()
        except RuntimeError as exc:
            if 'failed (404)' not in str(exc):
                raise
            print('Batch not visible yet; retrying.', flush=True)
        if (OUT/'first_recovery.json').exists():
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
