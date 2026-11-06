#!/usr/bin/env python3
"""Generate portfolio recovery figures from completed trial records.

Run: python figures/plot_portfolio_solve_over_time.py
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'figures/portfolio_solve_over_time'
METHODS = ('Base PySR', '709715')

def render_noise_average(output, records):
    """Average the four noise-level recovery curves with equal weight."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    noises = [0.0, 0.001, 0.01, 0.1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {'Base PySR': '#3264ad', '709715': '#d45b24'}
    for method in METHODS:
        subsets = [[r for r in records if r['method'] == method and r['noise'] == noise]
                   for noise in noises]
        assert all(subsets), f'Missing noise level for {method}'
        events = sorted((r['first_solve_budget_seconds'], 25 / len(subset))
                        for subset in subsets for r in subset
                        if r['first_solve_budget_seconds'] is not None)
        xs, ys = [0.1], [0]
        for seconds, weight in events:
            xs.append(seconds)
            ys.append(ys[-1] + weight)
        xs.append(900)
        ys.append(ys[-1])
        ax.step(xs, ys, where='post', color=colors[method], linewidth=2,
                label=f'{method} ({ys[-1]:.2f}% at 15 min)')
    ax.set(xscale='log', xlim=(0.1, 900), ylim=(0, 100),
           xlabel='Cumulative search time (seconds, log scale)',
           ylabel='Trials recovered at least once (%)',
           title='15-minute portfolios · average across noise levels')
    ax.set_xticks([0.1, 1, 10, 30, 100, 300, 900],
                  labels=['0.1', '1', '10', '30', '100', '300', '900'])
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc='upper left')
    fig.text(0.5, 0.055, 'Equal weight for noise 0, 0.001, 0.01, 0.1 · 133 tasks × 10 seeds per level',
             ha='center', fontsize=9)
    fig.text(0.5, 0.02, 'Recovery credited at restart completion; warm-up excluded; final overshoot mapped to 15 min.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'solve_rate_noise_average.{ext}', dpi=180)
    plt.close(fig)



def render_panels(output, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    noises = [0.0, 0.001, 0.01, 0.1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    colors = {'Base PySR': '#3264ad', '709715': '#d45b24'}
    for ax, noise in zip(axes.flat, noises):
        for method in METHODS:
            subset = [r for r in records if r['method'] == method and r['noise'] == noise]
            times = sorted(r['first_solve_budget_seconds'] / 60 for r in subset
                           if r['first_solve_budget_seconds'] is not None)
            xs = [0] + times + [15]
            ys = [0] + [100 * i / len(subset) for i in range(1, len(times) + 1)]
            ys.append(ys[-1])
            ax.step(xs, ys, where='post', label=method, color=colors[method], linewidth=2)
        ax.set_title(f'Target noise: {noise:g}')
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
        ax.set_xticks([0, 3, 6, 9, 12, 15])
    axes[0, 0].legend(frameon=False)
    fig.supxlabel('Cumulative search time (minutes)', y=0.045)
    fig.supylabel('Trials recovered at least once (%)')
    n_datasets = len({r['dataset'] for r in records})
    fig.suptitle(f'15-minute restart portfolios · {n_datasets} tasks × 10 seeds per noise level')
    fig.text(0.5, 0.015, 'Recovery credited at restart completion; warm-up excluded; final budget overshoot mapped to 15 min.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0.02, 0.08, 1, 0.97))
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'solve_rate.{ext}', dpi=180)
    plt.close(fig)


def render_figures(records, output=DEFAULT_OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    render_noise_average(output, records)
    render_panels(output, records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'reports/portfolio_solve_over_time/first_recovery.json')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render_figures(json.loads(args.input.read_text()), args.output)
