"""Empirical solve-over-time curve after the human audit of the LLM judge.

Run: python figures/solve_over_time/solve_over_time_audited.py

The audit (reports/empbench_human_audit/) agreed with the judge everywhere except
Rydberg, where the judge rejected log(c / (1/n1^2 - q/n2^2)) whenever the fitted q
was not exactly 1 (typically q = 0.99956). This script rescores Rydberg, for both
methods and at every snapshot, with a deterministic rule that accepts that family
when |q - 1| <= Q_TOLERANCE. The rule must reproduce every human Rydberg label, and
the human must agree with the judge on every other problem, or the script stops.
All other problems keep the judge's first-recovery times. No API calls, no jobs.
"""
import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, StrMethodFormatter
import numpy as np
import sympy as sp

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT))
from solve_over_time import METHODS, plotted_seconds, save, style  # noqa: E402

REVIEW = ROOT/'runs/srbench2_9-17_1m_spliced_terra_recovery'
AUDIT = ROOT/'reports/empbench_human_audit/empbench_audit_95fcc1681b94_Simon_20261112-2116.json'
RYDBERG = 'first_principles_rydberg'
AUDITED_METHOD = '709715'
Q_TOLERANCE = 1e-3      # largest |q - 1| the human accepted was 5.0e-4; the nearest rejected-size value seen is 8.5e-3
FAMILY_RESIDUAL = 1e-8  # true members of the family fit to ~1e-15; this only absorbs rounding

LOCALS = {'square': lambda x: x**2, 'cube': lambda x: x**3, 'neg': lambda x: -x, 'inv': lambda x: 1/x}
X0, X1 = sp.symbols('x0 x1')


def _grid():
    """Integer pairs n1 < n2 <= 15 (beyond the data's n <= 7, so interpolating overfits fail) plus real-valued pairs."""
    pairs = [(a, b) for a in range(1, 15) for b in range(a + 1, 16)]
    rng = np.random.default_rng(0)
    r1 = rng.uniform(1, 8, 200)
    n1 = np.concatenate([[a for a, _ in pairs], r1])
    n2 = np.concatenate([[b for _, b in pairs], r1*rng.uniform(1.1, 3, 200)])
    return n1, n2


N1, N2 = _grid()
BASIS = np.stack([1/N1**2, 1/N2**2], axis=1)
_cache = {}


def rydberg_q(equation):
    """q if exp(-f) = a*(1/n1^2 - q/n2^2) exactly, i.e. f = c - log(1/n1^2 - q/n2^2); else None."""
    if equation not in _cache:
        q = None
        try:
            f = sp.lambdify((X0, X1), sp.sympify(equation, locals=LOCALS), 'numpy')
            with np.errstate(all='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                y = np.asarray(f(N1.astype(complex), N2.astype(complex)))*np.ones(len(N1))
                g = np.exp(-y.real)
            if np.all(np.isfinite(y)) and np.abs(y.imag).max() < 1e-9 and np.all(np.isfinite(g)):
                coef = np.linalg.lstsq(BASIS, g, rcond=None)[0]
                if coef[0] != 0 and np.linalg.norm(BASIS@coef - g) <= FAMILY_RESIDUAL*np.linalg.norm(g):
                    q = float(-coef[1]/coef[0])
        except Exception:  # unparseable or non-numeric expressions are simply not in the family
            q = None
        _cache[equation] = q
    return _cache[equation]


def recovered(frontier):
    return any((q := rydberg_q(row['equation'])) is not None and abs(q - 1) <= Q_TOLERANCE for row in frontier)


def snapshot_time(snapshot):
    return snapshot['scheduled_seconds'] if snapshot['kind'] == 'first_restart_snapshot' else snapshot['seconds']


def rescore(record):
    """The review's binary search (scripts/empbench_portfolio_recovery.py) with the rule as the judge."""
    snapshots = json.loads((REVIEW/'snapshots'/f"{record['id']:03d}.json").read_text())
    flags = [recovered(s['frontier']) for s in snapshots]
    new = dict(record, final_positive=flags[-1], final_classification='exact' if flags[-1] else record['final_classification'],
               first_solve_seconds=None, first_solve_budget_seconds=None, first_solve_kind=None,
               first_solve_scheduled_seconds=None)
    if flags[-1]:
        low, high = 0, len(snapshots)
        while high - low > 1:
            mid = (high + low)//2
            if flags[mid - 1]:
                high = mid
            else:
                low = mid
        hit = snapshots[high - 1]
        new.update(first_solve_seconds=hit['seconds'], first_solve_budget_seconds=min(hit['seconds'], record['budget_seconds']),
                   first_solve_kind=hit['kind'], first_solve_scheduled_seconds=hit.get('scheduled_seconds'))
    # Diagnostics: is recovery monotone, as the binary search assumes?
    new['_earliest'] = next((snapshot_time(s) for s, f in zip(snapshots, flags) if f), None)
    new['_transient'] = any(a and not b for a, b in zip(flags, flags[1:]))
    return new, snapshots, flags


def check_against_audit(records, rescored):
    """The figure is only justified if the rule equals the human on Rydberg and the judge equals the human elsewhere."""
    audit = json.loads(AUDIT.read_text())
    early = audit['build']['early_seconds']
    by_trial = {(r['dataset'], r['seed']): r for r in records if r['method'] == AUDITED_METHOD}
    checked = 0
    for row in audit['decisions']:
        human = row['human_label'] == 'exact'
        if row['dataset'] != RYDBERG:
            if human != row['judge_recovered']:
                raise SystemExit(f"Human and judge disagree outside Rydberg ({row['id']}); this script only rescores Rydberg")
            continue
        _, snapshots, flags = rescored[by_trial[row['dataset'], row['seed']]['id']]
        if row['checkpoint'] == '1h':
            rule = flags[-1]
        else:
            rule = [f for s, f in zip(snapshots, flags) if snapshot_time(s) <= early][-1]
        if rule != human:
            raise SystemExit(f"Rule ({rule}) does not reproduce the human label ({row['human_label']}) on {row['id']}")
        checked += 1
    return checked, audit


def curves(records):
    grid = {3600.}
    for path in (REVIEW/'snapshots').glob('*.json'):
        grid.update(float(o['scheduled_seconds']) for o in json.loads(path.read_text())
                    if o.get('scheduled_seconds') is not None and 0 < o['scheduled_seconds'] <= 3600)
    grid.update(t for r in records if (t := plotted_seconds(r)) is not None)
    xs = np.array(sorted(grid))
    out = {}
    for _, method, _, _ in METHODS:
        rows = [r for r in records if r['method'] == method]
        seeds = sorted({r['seed'] for r in rows})
        per_seed = []
        for seed in seeds:
            trial = [r for r in rows if r['seed'] == seed]
            times = [t for r in trial if (t := plotted_seconds(r)) is not None]
            per_seed.append([100*sum(t <= x for t in times)/len(trial) for x in xs])
        per_seed = np.array(per_seed)
        out[method] = dict(mean=per_seed.mean(0), sd=per_seed.std(0, ddof=1), solved=sum(r['final_positive'] for r in rows), n=len(rows))
    return xs, out


def figure(xs, data, band):
    fig, ax = plt.subplots(figsize=(7.6*0.6, 5*0.6))
    for _, method, label, color in METHODS:
        d = data[method]
        if band:
            ax.fill_between(xs/60, np.maximum(0, d['mean'] - d['sd']), np.minimum(100, d['mean'] + d['sd']),
                            step='post', color=color, alpha=.18, linewidth=0)
        ax.step(xs/60, d['mean'], where='post', color=color, linewidth=2.2, label=f"{label} ({d['solved']}/{d['n']})")
    ax.set(xscale='log', xlim=(1/60, 60), xlabel='Search time (min)')
    ax.set_xticks([.02, .1, .5, 1, 5, 15, 60])
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    style(ax)
    return fig


def main():
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11, 'savefig.facecolor': 'white'})
    original = json.loads((REVIEW/'first_recovery.json').read_text())
    rescored = {r['id']: rescore(r) for r in original if r['dataset'] == RYDBERG}
    checked, audit = check_against_audit(original, rescored)
    print(f"Rule reproduces all {checked} human Rydberg labels; human and judge agree on all "
          f"{len(audit['decisions']) - checked} audited items of the other problems")

    fields = ['method', 'seed', 'judge_final', 'judge_first_solve_seconds', 'rule_final', 'rule_first_solve_seconds',
              'rule_earliest_seconds', 'rule_recovery_transient', 'final_frontier_q_minus_1']
    with (OUT/'rydberg_rescored.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for r in original:
            if r['dataset'] != RYDBERG:
                continue
            new, snapshots, _ = rescored[r['id']]
            qs = sorted((q - 1 for row in snapshots[-1]['frontier'] if (q := rydberg_q(row['equation'])) is not None), key=abs)
            writer.writerow(dict(method=r['method'], seed=r['seed'], judge_final=r['final_classification'],
                                 judge_first_solve_seconds=plotted_seconds(r), rule_final='exact' if new['final_positive'] else 'not recovered',
                                 rule_first_solve_seconds=plotted_seconds(new), rule_earliest_seconds=new['_earliest'],
                                 rule_recovery_transient=new['_transient'], final_frontier_q_minus_1=' '.join(f'{q:+.2e}' for q in qs)))
            if new['final_positive'] != r['final_positive'] or plotted_seconds(new) != plotted_seconds(r):
                print(f"  {r['method']:>8} seed {r['seed']}: judge {r['final_classification']} @ {plotted_seconds(r)} -> "
                      f"rule {'exact' if new['final_positive'] else 'not recovered'} @ {plotted_seconds(new)}"
                      + (' (recovery not monotone)' if new['_transient'] else ''))

    records = [rescored[r['id']][0] if r['id'] in rescored else r for r in original]
    xs, data = curves(records)
    xs_before, before = curves(original)
    with (OUT/'empirical_solve_over_time_audited.csv').open('w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['method', 'seconds', 'solve_rate_percent', 'seed_sd_percent'])
        for _, method, _, _ in METHODS:
            writer.writerows([method, x, m, s] for x, m, s in zip(xs, data[method]['mean'], data[method]['sd']))
    early = audit['build']['early_seconds']

    def solved_by(grid, d, t):
        return round(d['mean'][np.searchsorted(grid, t, side='right') - 1]*d['n']/100)

    for _, method, label, _ in METHODS:
        print(f"{label}: at {early:g} s {solved_by(xs_before, before[method], early)}/90 -> {solved_by(xs, data[method], early)}/90; "
              f"at 1 h {before[method]['solved']}/90 -> {data[method]['solved']}/90")
    save(figure(xs, data, band=False), 'empirical_solve_over_time_audited.pdf')
    save(figure(xs, data, band=True), 'empirical_solve_over_time_audited_errorbars.pdf')


if __name__ == '__main__':
    main()
