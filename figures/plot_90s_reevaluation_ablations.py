"""Plot the Sep 21 90-second ablations. Run with --refresh to fetch source data.

Offline rerender: python figures/plot_90s_reevaluation_ablations.py
Scores are best-candidate train diagnostics (10 fresh seeds), not population means.
Seed curves are linearly interpolated over their common domain for mean/std
aggregation; no tails are extrapolated. Evaluation counts are generation-end evolution eval_idx values,
including selection reevaluations but excluding baseline/diagnostic evaluations.
"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/reevaluation_ablations_90s'
RUNS = [
    ('n1', 1, '671962', '7wdnkqss'), ('n1', 2, '750244', 'kwxib4uc'),
    ('n1-reeval', 1, '671963', 'hbonk3al'), ('n1-reeval', 2, '750245', 'iuqo0leg'),
    ('n1-TTTS', 1, '750241', 'ubg1ir57'), ('n1-TTTS', 2, '980600', 'rpgjrcdd'),
    ('n3', 1, '671964', 'm642yus3'), ('n3', 2, '980597', '0g6wvmco'),
    ('n3-reeval', 1, '750239', 'igwan6cy'), ('n3-reeval', 2, '980598', 'k3xgq41q'),
    ('n3-TTTS', 1, '750242', 'egg7h6w2'), ('n3-TTTS', 2, '980599', 'skg0nsbf'),
    ('n10', 1, '64604', 'sifnivw6'), ('n10', 2, '64605', 'wftnv7sd'),
    ('n1', 3, '64606', '4e15t4ig'),
]
RESUMED = {'980597': ('750247', '20af68tv')}
PATTERN = re.compile(r'\[train reeval\] gen (\d+) (.+?): reeval GT match rate=([\d.]+) \(live=([\d.]+), winners_curse=([+\-\d.]+)\)')


def refresh():
    import wandb
    api = wandb.Api(timeout=90)
    def read_source(job, run_id):
        run = api.run(f'simon-alford/meta-sr/{run_id}')
        history = [r for r in run.scan_history(keys=['generation', 'eval_idx', 'best_score'], page_size=1000)
                   if all(r.get(k) is not None for k in ['generation', 'eval_idx', 'best_score'])]
        counts = {int(r['generation']): int(r['eval_idx']) for r in history}
        log = (ROOT / 'runs' / job / 'run.log').read_text()
        points = []
        for g, bundle, reeval, live, curse in PATTERN.findall(log):
            g = int(g)
            assert g in counts, (job, g, 'missing generation count')
            points.append(dict(generation=g, eval_idx=counts[g], bundle=bundle,
                               train_score=float(live), train_reeval_score=float(reeval),
                               winners_curse=float(live)-float(reeval)))
        assert len({p['generation'] for p in points}) == len(points), job
        return run, history, points

    records = []
    for method, seed, job, run_id in RUNS:
        folder = ROOT / 'runs' / job
        if not (folder / 'final_eval_summary.json').exists():
            print(job, method, seed, 'not complete: excluded', flush=True)
            continue
        run, history, points = read_source(job, run_id)
        sources = [dict(job_id=job, wandb_id=run_id)]
        if job in RESUMED:
            parent_job, parent_id = RESUMED[job]
            _, prior_history, prior_points = read_source(parent_job, parent_id)
            boundary = min(int(h['generation']) for h in history)
            prior_count = max(h['eval_idx'] for h in prior_history if h['generation'] == boundary)
            resumed_count = max(h['eval_idx'] for h in history if h['generation'] == boundary)
            assert prior_count == resumed_count, (job, prior_count, resumed_count)
            # Prefer fresh resumed diagnostics if both sources have the boundary.
            merged = {p['generation']: p for p in prior_points if p['generation'] <= boundary}
            merged.update({p['generation']: p for p in points})
            points = list(merged.values())
            history = [h for h in prior_history if h['generation'] < boundary] + history
            sources.insert(0, dict(job_id=parent_job, wandb_id=parent_id))
        points.sort(key=lambda p: p['generation'])
        assert run.config['seed'] == seed
        assert run.config['generations'] == (3 if job in RESUMED else 15)
        assert run.config['timeout'] == 90 and run.config['val_n_runs'] == 10
        assert run.config['population_type'] == 'topk'
        assert run.config['population'] == run.config['offspring'] == 10
        assert run.config['n_runs'] == int(method.split('-')[0][1:])
        assert run.config['reeval'] == ('TTTS' if method.endswith('TTTS') else 'population' if method.endswith('reeval') else 'none')
        assert max(h['generation'] for h in history) == 15
        records.append(dict(method=method, seed=seed, job_id=job, wandb_id=run_id,
                            sources=sources, status='COMPLETED', failure=None,
                            last_completed_generation=15,
                            config={k:run.config.get(k) for k in ['seed','n_runs','reeval','n_reevals','reeval_budget','population_type','population','offspring','generations','timeout','budget_mode','val_n_runs','population_reeval_runs','continue_from']},
                            generation_history=history, points=points))
        print(job, method, seed, 'completed', len(points), flush=True)
    (OUT / 'data.json').write_text(json.dumps(records, indent=2)+'\n')


def aggregate(records, method, xkey, ykey):
    """Equal-weight seed mean and population SD on a shared, unextrapolated grid."""
    import numpy as np
    runs = [r for r in records if r['method'] == method and r['status'] == 'COMPLETED']
    if not runs:
        raise ValueError(f'No completed runs for {method}')
    assert len({r['seed'] for r in runs}) == len(runs), method
    curves = []
    for run in runs:
        points = sorted(run['points'], key=lambda p: p[xkey])
        x = np.array([p[xkey] for p in points], dtype=float)
        y = np.array([p[ykey] for p in points], dtype=float)
        assert len(x) and np.all(np.diff(x) > 0), (method, run['seed'], xkey)
        curves.append((x, y))
    lo = max(x[0] for x, _ in curves)
    hi = min(x[-1] for x, _ in curves)
    assert lo <= hi, (method, xkey, 'no shared domain')
    grid = (np.arange(np.ceil(lo), np.floor(hi) + 1) if xkey == 'generation'
            else np.unique(np.concatenate([x[(x >= lo) & (x <= hi)] for x, _ in curves])))
    values = np.stack([np.interp(grid, x, y) for x, y in curves])
    return grid, values.mean(axis=0), values.std(axis=0, ddof=0), len(runs)


def draw_mean(ax, records, method, xkey, ykey, color):
    x, mean, std, n = aggregate(records, method, xkey, ykey)
    ax.plot(x, mean, color=color, lw=1.8, marker='o', ms=3,
            label=f'{method} ({n} seed' + ('s)' if n != 1 else ')'))
    if n > 1:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=.18, linewidth=0)


MEAN_NOTE = 'Lines: seed mean; shading: ±1 SD (ddof=0). One seed: no band. Linear interpolation within shared seed coverage; no extrapolation.'


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = json.loads((OUT / 'data.json').read_text())
    with (OUT / 'scores.csv').open('w') as f:
        writer = csv.DictWriter(f, lineterminator='\n', fieldnames=['method','seed','job_id','status','generation','eval_idx','bundle','train_score','train_reeval_score','winners_curse'])
        writer.writeheader()
        for r in records:
            for p in r['points']:
                writer.writerow({**{k:r[k] for k in ['method','seed','job_id','status']}, **p})
    with (OUT / 'aggregate_scores.csv').open('w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['method', 'axis', 'metric', 'x', 'mean', 'std', 'n_seeds'])
        for method in sorted({r['method'] for r in records}):
            for xkey in ['generation', 'eval_idx']:
                for ykey in ['train_reeval_score', 'winners_curse']:
                    x, mean, std, n = aggregate(records, method, xkey, ykey)
                    writer.writerows((method, xkey, ykey, a, b, c, n) for a, b, c in zip(x, mean, std))
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    colors = ['#0072B2','#D55E00','#009E73']
    for row, group in enumerate(['n1','n3']):
        for col, xkey in enumerate(['generation','eval_idx']):
            draw(axes[row,col], records, group, xkey, 'train_reeval_score', colors)
            axes[row,col].set_title(f'{group}: train reevaluation vs {"generation" if col == 0 else "evaluations"}')
            axes[row,col].set_ylabel('Reevaluated train score')
            axes[row,col].set_ylim(0.30,0.90)
        draw(axes[2,row], records, group, 'generation','winners_curse', colors)
        axes[2,row].set_title(f'{group}: winner’s curse')
        axes[2,row].set_ylabel('Train score − reevaluated train score')
        axes[2,row].set_ylim(-0.06,0.36)
        axes[2,row].axhline(0,color='0.4',lw=.8)
    fig.suptitle('90-second PySR ablations · best-candidate train reevaluation',fontsize=17,y=.985)
    fig.text(.5, .94, 'Mean across completed seeds ±1 standard deviation', ha='center', fontsize=10)
    fig.text(.06,.035, MEAN_NOTE + '\nDiagnostics use 10 fresh training seeds. Evaluations count evolution seed-runs across the training task set, excluding diagnostics.', fontsize=9,va='bottom')
    fig.tight_layout(rect=(0,.105,1,.925),h_pad=2.2,w_pad=2.2)
    fig.savefig(OUT / 'train_reevaluation_six_panels.pdf')
    plt.close(fig)
    plot_combined(records)
    plot_initial_seeds(records)
    write_readme(records)


def plot_combined(records):
    import matplotlib.pyplot as plt

    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#555555', '#000000']
    methods = ['n1', 'n1-reeval', 'n1-TTTS', 'n3', 'n3-reeval', 'n3-TTTS', 'n10']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, xkey in enumerate(['eval_idx', 'generation']):
        for ax, metric, title in zip(axes[row], ['train_reeval_score', 'winners_curse'],
                                     ['Reevaluated train score', 'Winner’s curse']):
            for method, color in zip(methods, colors):
                draw_mean(ax, records, method, xkey, metric, color)
            ax.set_title(f'{title} vs {"evaluations" if row == 0 else "generation"}')
            ax.set_xlabel('Cumulative evolution evaluations (seed-runs)' if row == 0 else 'Generation')
            ax.set_ylabel('Reevaluated train score' if metric == 'train_reeval_score'
                          else 'Train score − reevaluated train score')
            ax.set_ylim((.30,.90) if metric == 'train_reeval_score' else (-.06,.36))
            ax.set_xlim((0, max(p['eval_idx'] for r in records for p in r['points']) * 1.04) if row == 0 else (-.35,15.35))
            if row == 1:
                ax.set_xticks(range(0,16,3))
            ax.grid(alpha=.2)
            ax.legend(ncol=2,fontsize=8,frameon=False,loc='upper left')
            if metric == 'winners_curse':
                ax.axhline(0,color='0.4',lw=.8)
    fig.suptitle('All seven 90-second PySR ablations · evaluation and generation comparisons',fontsize=16,y=.98)
    fig.text(.5, .94, 'Mean across completed seeds ±1 standard deviation', ha='center', fontsize=10)
    fig.text(.06,.03, MEAN_NOTE + '\nEvaluations include selection reevaluations and exclude diagnostics. One seed-run covers the training task set.', fontsize=9)
    fig.tight_layout(rect=(0,.085,1,.91), h_pad=2.5)
    fig.savefig(OUT / 'all_methods_eval_axis.pdf')
    plt.close(fig)


def plot_initial_seeds(records):
    import matplotlib.pyplot as plt

    methods = {'n1': '#0072B2', 'n3': '#CC79A7', 'n10': '#000000'}
    selected = [r for r in records if r['method'] in methods]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, xkey in zip(axes, ['generation', 'eval_idx']):
        for method, color in methods.items():
            draw_mean(ax, records, method, xkey, 'train_reeval_score', color)
        ax.set_title('Reevaluated train score vs ' + ('generation' if xkey == 'generation' else 'evaluations'))
        ax.set_xlabel('Generation' if xkey == 'generation' else 'Cumulative evolution evaluations (seed-runs)')
        ax.set_ylabel('Reevaluated train score')
        ax.set_ylim(.30, .90)
        if xkey == 'generation':
            ax.set_xlim(-.35, 15.35)
            ax.set_xticks(range(0, 16, 3))
        else:
            ax.set_xlim(0, max(p['eval_idx'] for r in selected for p in r['points']) * 1.04)
        ax.grid(alpha=.2)
        ax.legend(ncol=3, frameon=False, loc='upper left')
    fig.suptitle('n1 vs n3 vs n10 · no selection reevaluation', fontsize=16, y=.98)
    fig.text(.5, .90, 'Mean across completed seeds ±1 standard deviation', ha='center', fontsize=10)
    fig.text(.06, .025, MEAN_NOTE + '\nBest-candidate train diagnostics use 10 fresh seeds. Evaluations exclude diagnostics.', fontsize=9)
    fig.tight_layout(rect=(0, .105, 1, .85))
    fig.savefig(OUT / 'n1_n3_n10_train_reevaluation.pdf')
    plt.close(fig)


def write_readme(records):
    from datetime import datetime, timezone
    text = f"""# 90-second PySR reevaluation ablations

Updated {datetime.now(timezone.utc).isoformat()}. Includes {len(records)} completed method/seed combinations.

`train_reevaluation_six_panels.pdf` compares n1 and n3 with population reevaluation
and TTTS, without n10. It shows reevaluated
train score versus generation and cumulative evolution evaluations, plus winner's
curse versus generation. `all_methods_eval_axis.pdf` compares all seven methods
on both axes. All panels show an equally weighted mean across completed seeds,
with ±1 population standard deviation (ddof=0), not a standard error or confidence
interval. Legend counts state the number of independent seeds. A one-seed method
has no band; its variability cannot be estimated.

For each method and x-axis, each seed is linearly interpolated onto a common grid
within the intersection of its observed range with all other seeds. Generation
grids are integer-valued; evaluation grids use the union of observed counts inside
the shared range. There is no extrapolation or changing seed count along a curve.
Thus n3's generation curve stops at 14 because seed 1 has no generation-15 diagnostic.
Winner's curse is computed per seed before averaging. `aggregate_scores.csv`
records the plotted means, SDs and seed counts for both metrics and x-axes.

`n1_n3_n10_train_reevaluation.pdf` compares n1, n3 and n10 without selection
reevaluation: reevaluated train score versus generation on the left and cumulative
evolution evaluations on the right, using the same completed seeds.

Scores are best-candidate training diagnostics on 10 fresh seeds, parsed from
`[train reeval]` records in local run.log files (four-decimal logging precision).
These are not validation scores or population averages. Winner's curse is the
contemporaneous live train score minus the reevaluated score. Raw diagnostics remain
unchanged; interpolation is applied only for aggregating the plotted curves.

Unsampled W&B generation records supply evolution eval_idx at the submitted
generation, not the diagnostic completion step. Counts include initial population,
offspring and selection reevaluations; they exclude baseline, diagnostic and final
evaluations. A seed-run covers the entire training task set. The evaluation-axis
range includes the full n10 budget.

All runs use topk selection, population/offspring 10, best2 models, 90-second PySR
budgets and 15 generations. Population reevaluation tops up n1 to 3 seeds and n3
to 10; TTTS budgets are 10 and 30 per generation. n10 uses 10 initial seeds per
candidate without selection reevaluation.

The n3 seed-2 continuation merges original job 750247 with 980597. Their generation-12
evaluation counters agree at 390; no offset is applied. This is one seed, not two.
Failed attempts superseded by fresh retries are excluded. Completion requires a
local final_eval_summary.json and a generation-15 W&B record.

| Method | Seed | Source jobs | Last diagnostic generation |
|---|---:|---|---:|
"""
    for r in records:
        sources = ' → '.join(s['job_id'] for s in r['sources'])
        text += f"| {r['method']} | {r['seed']} | {sources} | {r['points'][-1]['generation']} |\n"
    included = {r['job_id'] for r in records}
    excluded = [f'{method} seed {seed} (job {job})' for method, seed, job, _ in RUNS if job not in included]
    text += '\nConfigured but not complete at refresh: ' + (', '.join(excluded) or 'none') + '.\n'
    text += """
Reproduce offline: `python figures/plot_90s_reevaluation_ablations.py`.
Add `--refresh` to reread local logs and W&B. `data.json` preserves the source
configuration, generation counters and diagnostics; `scores.csv` contains plotted
observations. Outputs are PDF only.
"""
    (OUT / 'README.md').write_text(text)


def draw(ax, records, group, xkey, ykey, colors):
    methods = [group,group+'-reeval',group+'-TTTS']
    for method, color in zip(methods, colors):
        draw_mean(ax, records, method, xkey, ykey, color)
    ax.set_xlabel('Generation' if xkey=='generation' else 'Cumulative evolution evaluations (seed-runs)')
    if xkey=='generation':
        ax.set_xlim(-.35,15.35)
        ax.set_xticks(range(0,16,3))
    ax.grid(alpha=.2)
    ax.legend(loc='upper left',ncol=2,fontsize=8,frameon=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.refresh:
        refresh()
    plot()
