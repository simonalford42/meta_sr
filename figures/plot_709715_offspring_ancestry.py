#!/usr/bin/env python3
"""Plot offspring fitness, recorded ancestry, and operator origins for run 709715."""

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '709715_offspring_ancestry'
TYPES = ('mutation', 'survival', 'selection', 'loss')
BASELINE = ('add_constant_offset', 'age_regularized_survival', 'tournament_selection', 'mse_loss')
COLORS = dict(mutation='#d62728', loss='#2878c8', selection='#e4bc24', survival='#2c9b49')


def draw_ancestry_legend(ax):
    """Align category labels at left, with a square bracket grouping operators."""
    for i, (label, color) in enumerate(COLORS.items()):
        y = 0.255 - i * 0.045
        ax.scatter([0.86], [y], s=46, c=color, edgecolors='none',
                   transform=ax.transAxes, zorder=6)
        ax.text(0.878, y, label.capitalize(), transform=ax.transAxes,
                va='center', fontsize=9, zorder=6)
    ax.scatter([0.86], [0.06], s=46, c='#bdbdbd', alpha=0.70,
               edgecolors='none', transform=ax.transAxes, zorder=6)
    ax.text(0.815, 0.06, 'Other offspring', transform=ax.transAxes,
            ha='right', va='center', fontsize=9, zorder=6)
    x, top, bottom = 0.835, 0.274, 0.101
    mid = (top + bottom) / 2
    vertices = [(x+0.010, top), (x, top), (x, bottom), (x+0.010, bottom)]
    ax.add_patch(PathPatch(MplPath(vertices, [MplPath.MOVETO]+[MplPath.LINETO]*3),
                           transform=ax.transAxes, fill=False, edgecolor='#555555',
                           linewidth=1, zorder=6))
    ax.plot([x-0.012, x], [mid, mid], transform=ax.transAxes,
            color='#555555', linewidth=1, zorder=6)
    ax.text(0.815, mid, 'Ancestor', transform=ax.transAxes,
            ha='right', va='center', fontsize=9, zorder=6)


def extract(source):
    data = json.loads(source.read_text())
    def compact(bundle):
        return {k: bundle[k] for k in ('score', 'meta_mutation_counts')} | {
            'operators': {t: {k: op.get(k) for k in ('name', 'generation', 'parent_name', 'mode')}
                          for t, op in bundle['operators'].items()}}
    records = {'generations': [{'generation': g['generation'],
                               **{kind: [compact(b) for b in g[kind]]
                                  for kind in ('population', 'offspring')}}
                              for g in data['generations']],
               'val_results': {n: {'avg_score': v.get('avg_score')}
                               for n, v in data['val_results'].items()}}
    (OUT / 'lineage_records.json').write_text(json.dumps(records, indent=2) + '\n')


def reconstruct(data):
    operators, bundles, rows, offspring = {}, {}, [], []
    def key(b):
        return tuple(b['operators'][t]['name'] for t in TYPES)
    for g in data['generations']:
        for kind in ('population', 'offspring'):
            for index, original in enumerate(g[kind]):
                b = dict(original, seen=g['generation'], kind=kind, index=index)
                b['key'] = key(b)
                rows.append(b)
                bundles.setdefault(b['key'], b)
                for t, op in b['operators'].items():
                    operators[op['name']] = dict(op, type=t)
                if kind == 'offspring':
                    offspring.append(b)
    def birth(b):
        return max(op['generation'] for op in b['operators'].values())
    def event(b):
        edited = [operators[n] for n in b['key'] if operators[n]['generation'] == birth(b)
                  and n not in BASELINE]
        if len(edited) != 1:
            raise ValueError(f'Ambiguous creation event: {b["key"]}')
        return edited[0]
    names = {' | '.join(k): k for k in bundles}
    eligible = [(n, v['avg_score']) for n, v in data['val_results'].items()
                if n in names and v.get('avg_score') is not None]
    selected_name, val_score = max(eligible, key=lambda item: item[1])
    selected = bundles[names[selected_name]]
    assert selected['operators']['selection']['name'] == 'streamlined_niche_clone_tournament_gen43_3'

    def parent(b):
        op = event(b)
        counts = copy.deepcopy(b['meta_mutation_counts'])
        counts[op['type']][op['mode']] -= 1
        found = {}
        for candidate in rows:
            if candidate['seen'] >= birth(b) or candidate['meta_mutation_counts'] != counts:
                continue
            if any(candidate['operators'][t]['name'] != b['operators'][t]['name']
                   for t in TYPES if t != op['type']):
                continue
            if op['mode'] in ('refine', 'simplify') and candidate['operators'][op['type']]['name'] != op['parent_name']:
                continue
            found.setdefault(candidate['key'], candidate)
        if len(found) != 1:
            raise ValueError(f'Expected unique recorded bundle parent: {b["key"]}; found {len(found)}')
        return next(iter(found.values()))

    # Operator-origin coloring follows the explicit parent chain of each final component.
    origins = {}
    for t in TYPES:
        name = selected['operators'][t]['name']
        visited = set()
        while name and name not in BASELINE:
            if name in visited:
                raise ValueError('Cycle in operator ancestry')
            visited.add(name)
            origins[name] = t
            name = operators[name]['parent_name']
    # Also follow bundle inheritance and recorded operator donors recursively.
    creators = {}
    for b in bundles.values():
        if b['key'] != BASELINE:
            creators.setdefault(event(b)['name'], b)
    ancestors = set()
    def visit(b):
        if b['key'] in ancestors:
            return
        ancestors.add(b['key'])
        if birth(b) > 0:
            visit(parent(b))
        op = event(b)
        if op['mode'] in ('refine', 'simplify', 'crossover'):
            donor = creators.get(op['parent_name'])
            if donor is not None:
                visit(donor)
    visit(selected)
    # Initial candidates have no separate generation-0 snapshot. Use earliest saved
    # population fitness for surviving initial bundles and explicitly flag it.
    initial = [b for b in bundles.values() if birth(b) == 0 and b['key'] != BASELINE]
    plotted = []
    for b in initial + offspring:
        op = event(b)
        score = b['score']
        if score is None or not math.isfinite(float(score)):
            raise ValueError(f'Missing/nonfinite fitness for {op["name"]}')
        plotted.append(dict(generation=birth(b), fitness=float(score), mode=op['mode'],
                            edited_operator=op['type'], operator_name=op['name'],
                            status='operator_origin' if op['name'] in origins else
                                   'ancestor' if b['key'] in ancestors else 'not_recorded_ancestor',
                            color_operator=origins.get(op['name'], ''),
                            score_source='first_saved_population' if birth(b) == 0 else 'offspring',
                            score_snapshot_generation=b['seen'], bundle=' | '.join(b['key'])))
    assert sorted(p['generation'] for p in plotted if p['color_operator'] == 'loss') == [0, 8, 34]
    return plotted, dict(selected_bundle=selected_name, validation_score=val_score,
                         recorded_ancestor_bundles=len(ancestors),
                         colored_origins={t: sorted(p['generation'] for p in plotted if p['color_operator'] == t)
                                          for t in TYPES})


def render(points):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.97)
    for status in ('not_recorded_ancestor', 'ancestor', 'operator_origin'):
        group = [p for p in points if p['status'] == status]
        colors = [COLORS[p['color_operator']] if p['color_operator'] else
                  '#555555' if status == 'ancestor' else '#bdbdbd' for p in group]
        ax.scatter([p['generation'] for p in group], [p['fitness'] for p in group],
                   marker='o', s=66 if status == 'operator_origin' else 50 if status == 'ancestor' else 38,
                   c=colors, edgecolors='none', alpha=0.70 if status == 'not_recorded_ancestor' else 1,
                   zorder=4 if status == 'operator_origin' else 3 if status == 'ancestor' else 2)
    ax.set(xlabel='Generation', ylabel='Fitness (GT)', xlim=(-1, 46), ylim=(0, 1))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(color='#e9ecf0', lw=0.8)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_color('#bfc5cd')
    draw_ancestry_legend(ax)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'offspring_ancestry.{ext}', dpi=200, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-from', type=Path, help='Re-extract compact metadata from run_data.json')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.refresh_from:
        extract(args.refresh_from)
    points, summary = reconstruct(json.loads((OUT / 'lineage_records.json').read_text()))
    render(points)
    with (OUT / 'plotted_offspring.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    (OUT / 'ancestry_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Plotted {len(points)} creation events to {OUT}')


if __name__ == '__main__':
    main()
