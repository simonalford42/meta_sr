"""Plot observed recovery at endpoints of the original 1M-evaluation restarts.

No new searches or API reviews. Bode first-restart matches are checked against
its archived broad exponential-family rubric (zero offset is allowed).
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter, PercentFormatter
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/empirical_overlap_1m_portfolio'
TIMINGS = ROOT / 'reports/srbench2_portfolio_solve_over_time/first_recovery.json'
RUNS = {
    'Baseline': 'runs/srbench2_9-8_gt_baseline_1core_l1_portfolio_1m_no_warmup',
    '709715': 'runs/709715/srbench2_9-8_ground_truth_1core_portfolio_1m_no_warmup',
}
OVERLAP = {'hubble','ideal_gas','kepler','leavitt','newton','planck','rydberg','schechter'}


def exponential_family(equation):
    x = sp.Symbol('x0', positive=True)
    e = sp.sympify(equation, locals={'x0': x, 'square': lambda v:v**2, 'cube':lambda v:v**3}, rational=True)
    derivative = sp.diff(e, x)
    if derivative == 0:
        return False
    rate = sp.simplify(sp.diff(derivative, x)/derivative)
    return bool(not rate.free_symbols and rate.is_finite and rate.is_real and rate.is_zero is False
                and sp.simplify(sp.diff(e, x, 2)-rate*derivative) == 0)


def main():
    records = [dict(r) for r in json.loads(TIMINGS.read_text())
               if r['dataset'].removeprefix('first_principles_') in OVERLAP]
    evidence = []
    for method, dirname in RUNS.items():
        raw = json.loads((ROOT/dirname/'srbench_full_results.json').read_text())['results']
        for r in raw.values():
            if r['dataset'] != 'first_principles_bode':
                continue
            first = r['portfolio']['restarts'][0]
            assert not first.get('error')
            selected = next(row['equation'] for row in sorted(first['pareto_frontier'], key=lambda r:r['complexity'])
                            if exponential_family(row['equation']))
            t = first['search_runtime_seconds']
            evidence.append(dict(method=method, dataset=r['dataset'], seed=r['seed'],
                                 first_restart_seconds=t, equation=selected,
                                 criterion='c0+c1*exp(c2*x0), including zero offset; archived broad Bode rubric'))
            records.append(dict(method=method, dataset=r['dataset'], seed=r['seed'],
                                first_solve_budget_seconds=t, final_positive=True))
    assert len(evidence) == 20 and len(records) == 180
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'bode_first_restart_evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(8.6,5.3))
    table = []
    for method, label, color, style in [('Baseline','Base PySR','#2878B5','-'),
                                       ('709715','Evolved 709715','#D55E00','--')]:
        subset = [r for r in records if r['method']==method]
        assert len(subset)==90 and len({(r['dataset'],r['seed']) for r in subset})==90
        events = Counter(r['first_solve_budget_seconds'] for r in subset if r['final_positive'])
        xs, ys, count = [1.], [0.], 0
        for t, n in sorted(events.items()):
            count += n
            xs.append(t); ys.append(100*count/90)
        assert count == (72 if method=='Baseline' else 74)
        xs.append(3600); ys.append(100*count/90)
        ax.step(xs,ys,where='post',color=color,ls=style,lw=2.3,label=f'{label}: {count}/90 ({100*count/90:.1f}%)')
        table.extend(dict(method=label,seconds=t,solve_rate_percent=y) for t,y in zip(xs,ys))
    ax.set(xscale='log',xlim=(1,3600),ylim=(0,100),xlabel='Cumulative search time (seconds, log scale)',
           ylabel='Cumulative observed recovery rate',title='EmpiricalBench overlap in SRBench2 · 1M-evaluation restarts')
    ax.set_xticks([1,5,10,30,90,300,900,3600]); ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(alpha=.22); ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='upper left')
    fig.text(.5,.025,'9 tasks × 10 seeds; Bode uses the archived broad phenomenological criterion.\n'
             'Restart-end observations; no within-restart timing. Warm-up excluded; binary-search approximation.',
             ha='center',fontsize=8.2)
    fig.tight_layout(rect=(0,.075,1,1))
    for ext in ['png','pdf']:fig.savefig(OUT/f'solve_rate.{ext}',dpi=200)
    plt.close(fig)
    with (OUT/'curve.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
    (OUT/'trial_times.json').write_text(json.dumps([{k:r.get(k) for k in ['method','dataset','seed','first_solve_budget_seconds','final_positive']} for r in records],indent=2)+'\n')
    (OUT/'README.md').write_text('''# Existing 1M-evaluation portfolio recovery

Reproduce: `python figures/plot_empirical_overlap_1m_portfolio.py`.

This is the original SRBench2 setup, restricted to the nine EmpiricalBench-overlap problems. It is not the newer EmpiricalBench setup, and no first restarts have been replaced or simulated.

Eight tasks reuse the saved audited timing records in `reports/srbench2_portfolio_solve_over_time/first_recovery.json`. For Bode, all 20 original first-restart frontiers contain an exponential-family expression. This script verifies that form algebraically via a constant nonzero logarithmic derivative of the first derivative. `bode_first_restart_evidence.json` records the exact selected expressions and endpoint times. The archived broad phenomenological rubric allows a zero offset, including exp(x0); it does not require recovery of the full nonzero-offset Bode law.

The curve counts first observed recovery at saved restart endpoints. Before a first endpoint there is no within-restart timing information; zero observed recovery there is not proof that no equation had been discovered. First-restart durations vary by problem, seed and method. No new searches or API reviews are performed. The final totals are 72/90 baseline and 74/90 evolved under this archived combined criterion.

Timing excludes warm-up and scoring; final overshoot is capped at one hour in the inherited budget-time records. Binary search assumes persistence and may miss transient matches. Replacing a first restart could change later cumulative frontiers, so this figure is an observation of the old experiment, not a guaranteed endpoint for any proposed synthetic integration.
''')
    print(OUT)


if __name__=='__main__':
    main()
