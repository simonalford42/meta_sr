"""SRBench2 adapter for checkpointed Terra binary-search timing reviews.

No requests without --run. Reuses the tested EmpiricalBench batch engine,
with SRBench2 targets, unrestricted first-fit times and the archived Bode rubric.
"""
import json
from pathlib import Path
import empbench_portfolio_recovery as engine
from splice_srbench2_first_restart import DATASETS, snapshots

engine.EXTRA = '''\nFitted constants are free only where the reference allows them.
Fixed powers and relative coefficients must match exactly; never round them.
For Rydberg, C-log(1/x0^2-q/x1^2) is not exact unless q is exactly 1.
For Newton and ideal gas, fixed relative logarithmic coefficients cannot vary.
For Planck, constants, input variables and Wien approximations are not exact.
For SRBench2 Leavitt, x0 is already log10(period); an affine function is the reference.
For Bode, use the archived broad phenomenological family c0+c1*exp(c2*x0).
Zero offset is allowed: exp(x0) is a phenomenological_match, but a constant is not.
'''


def initialize():
    fingerprints = {label: engine.digest(json.loads((path/'srbench_full_results.json').read_text()))
                    for label, path in engine.RUNS.items()}
    config = {'model': engine.MODEL, 'extra': engine.EXTRA, 'targets': engine.TARGETS,
              'rubric': engine.RUBRIC, 'schema': engine.REVIEW_SCHEMA, 'max_tokens': engine.MAX_TOKENS,
              'adapter_version': 1}
    if (engine.OUT/'state.json').exists():
        state = json.loads((engine.OUT/'state.json').read_text())
        if state['fingerprints'] != fingerprints or state['config_hash'] != engine.digest(config):
            raise ValueError('Sources/configuration changed; use a fresh review directory')
        return
    trials = []
    for method, path in engine.RUNS.items():
        data = json.loads((path/'srbench_full_results.json').read_text())
        rows = list(data['results'].values())
        if len(rows) != 90 or {(r['dataset'], r['seed']) for r in rows} != {(d,s) for d in DATASETS for s in range(10000,10010)}:
            raise ValueError('Incomplete trial grid')
        for r in sorted(rows, key=lambda r:(r['dataset'],r['seed'])):
            obs, missing = snapshots(r)
            for o in obs:
                o['frontier'] = engine.compact(o['frontier'])
            if obs[-1]['frontier'] != engine.compact(r['pareto_frontier']):
                raise ValueError('Final frontier mismatch')
            t = dict(id=len(trials), method=method, dataset=r['dataset'], seed=r['seed'],
                     n_snapshots=len(obs), budget_seconds=3600, missing_snapshots=missing,
                     low=0, high=len(obs), history=[])
            engine.write(engine.OUT/'snapshots'/f"{t['id']:03d}.json", obs)
            trials.append(t)
    engine.write(engine.OUT/'state.json', dict(trials=trials, cache={}, round=0, cost_usd=0,
                 fingerprints=fingerprints, config_hash=engine.digest(config)))


original_normalize = engine.normalize_review
original_render = engine.render


def normalize(review, item):
    if review['classification'] == 'phenomenological_match' and item['dataset'] == 'first_principles_bode':
        review = dict(review, original_classification='phenomenological_match', classification='exact',
                      normalization='Archived broad Bode exponential family counted as recovered')
    return original_normalize(review, item)


def render(state):
    original_render(state)
    (engine.OUT/'README.md').write_text(
        '# Synthetic SRBench2 1M portfolio recovery\n\n'
        'Nine EmpiricalBench-overlap tasks, ten seeds per method. New first restart with five-second '
        'captures replaces the archived first restart; all later restarts are retained. '
        'Cumulative native-loss frontiers are rebuilt and final outcomes reviewed afresh. '
        'This is a synthetic combination, not a newly observed one-hour search.\n\n'
        'Terra medium; exact reference recovery plus the archived broad Bode phenomenological family '
        '(zero offset allowed). No endpoint totals are forced. Binary search assumes persistent '
        'recovery; final-negative/transient matches can be missed. Missing captures are recorded, '
        'never backfilled. Actual fit times exclude warm-up; later times shift by the first-fit '
        'duration difference. Only final overshoot is clipped to 3600 in budget-time columns.\n\n'
        f"Review cost: ${state['cost_usd']:.4f}.\n")


if __name__ == '__main__':
    engine.initialize = initialize
    engine.normalize_review = normalize
    engine.render = render
    engine.main()
