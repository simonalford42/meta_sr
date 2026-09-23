#!/usr/bin/env python3
"""Plot offspring fitness, recorded ancestry, and operator origins for run 709715."""

import argparse
import copy
import csv
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

# Scale the entire figure: canvas, fonts, dots, and line widths.
SCALE = 1.0

HIDDEN_COMMENTARY_LABELS = {
    'Cost-aware age replacement',
    'Simplified rational motif templates',
    'Simplified cyclic motif coupling',
    'Clone suppression and niche rescue',
    'Quality-gated niche exploration',
}

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '709715_offspring_ancestry'
TYPES = ('mutation', 'survival', 'selection', 'loss')
BASELINE = ('add_constant_offset', 'age_regularized_survival', 'tournament_selection', 'mse_loss')
COLORS = dict(mutation='#d62728', loss='#2878c8', selection='#e4bc24', survival='#2c9b49')


def draw_ancestry_legend(ax, fontsize):
    """Right-align category labels beside the operator markers."""
    # Scale the custom legend in physical units when canvas or font size changes.
    width, height = ax.figure.get_size_inches()
    sx = (fontsize / 9) * (10.2 / width)
    sy = (fontsize / 9) * (5.6 / height)
    def lx(value):
        return 1.0 - (0.97 - value) * sx
    def ly(value):
        return 0.025 + (value - 0.025) * sy
    ax.add_patch(Rectangle((lx(0.73), ly(0.025)), 0.24 * sx, 0.27 * sy,
                           transform=ax.transAxes, facecolor='white',
                           edgecolor='grey', linewidth=0.5 * SCALE, alpha=1, zorder=5))
    for i, (label, color) in enumerate(COLORS.items()):
        y = 0.255 - i * 0.045
        ax.scatter([lx(0.86)], [ly(y)], s=46 * SCALE**2, c=color, edgecolors='none',
                   transform=ax.transAxes, zorder=6)
        ax.text(lx(0.878), ly(y), label.capitalize(), transform=ax.transAxes,
                va='center', fontsize=fontsize, zorder=6)
    ax.scatter([lx(0.86)], [ly(0.06)], s=46 * SCALE**2, c='#bdbdbd', alpha=0.70,
               edgecolors='none', transform=ax.transAxes, zorder=6)
    ax.text(lx(0.845), ly(0.06), 'Other offspring', transform=ax.transAxes,
            ha='right', va='center', fontsize=fontsize, zorder=6)
    mid = (0.255 + 0.120) / 2
    ax.text(lx(0.845), ly(mid), 'Ancestor', transform=ax.transAxes,
            ha='right', va='center', fontsize=fontsize, zorder=6)


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


def recover_initial_population(data):
    """Recover the ten original scores from the initial evaluation log block."""
    source = ROOT.parent / 'runs/709715/run.log'
    known = {op['name']: op for g in data['generations']
             for kind in ('population', 'offspring') for b in g[kind]
             for op in b['operators'].values()}
    population, evidence = [], []
    in_initial = False
    for line_number, line in enumerate(source.read_text().splitlines(), 1):
        if 'Evaluating initial population (10 bundles)' in line:
            in_initial = True
        if in_initial and '[timing] initial-pop evaluation:' in line:
            break
        if not in_initial:
            continue
        match = re.match(r'  Avg ([0-9.]+) (.+): \[([^]]+)\] solved', line)
        if not match:
            continue
        printed_mean, bundle_name, scores = match.groups()
        names = bundle_name.split(' | ')
        assert len(names) == len(TYPES)
        seed_scores = [float(score) for score in scores.split(',')]
        assert len(seed_scores) == 3
        # The logged scores are fractions solved out of 20 datasets per seed.
        assert all(abs(score * 20 - round(score * 20)) < 1e-8 for score in seed_scores)
        score = sum(seed_scores) / len(seed_scores)
        assert abs(score - float(printed_mean)) <= 0.00005
        counts = {t: dict.fromkeys(('explore', 'refine', 'simplify', 'crossover'), 0) for t in TYPES}
        operators = {}
        for t, name, baseline in zip(TYPES, names, BASELINE):
            operators[t] = known.get(name, dict(name=name, generation=0, parent_name=None, mode='explore'))
            assert operators[t]['generation'] == 0
            if name != baseline:
                counts[t]['explore'] = 1
        population.append(dict(score=score, meta_mutation_counts=counts, operators=operators))
        evidence.append(dict(bundle=bundle_name, fitness=score, seed_scores=seed_scores,
                             source=str(source.relative_to(ROOT.parent)), line=line_number, text=line))
    assert len(population) == 10
    assert len({r['bundle'] for r in evidence}) == 10
    (OUT / 'initial_population_scores.json').write_text(json.dumps(evidence, indent=2) + '\n')
    return dict(generation=0, population=population, offspring=[])


def reevaluated_scores():
    """Prefer independent train reevaluations, then population seed top-ups."""
    population, independent = {}, {}
    generation = 0
    for line_number, line in enumerate((ROOT.parent / 'runs/709715/run.log').read_text().splitlines(), 1):
        gen = re.match(r'Generation (\d+)/\d+', line)
        if gen:
            generation = int(gen.group(1))
        match = re.match(r'  (?:\[extras\] )?Avg ([0-9.]+) (.+): \(seeds=(\d+)\)', line)
        if match:
            score, bundle, seeds = match.groups()
            if int(seeds) > 3:
                population[bundle] = dict(fitness=float(score), score_source='population_reevaluation_log',
                                          score_snapshot_generation=generation, score_log_line=line_number)
        match = re.match(r'\[train reeval\] gen (\d+) (.+): reeval GT match rate=([0-9.]+)', line)
        if match:
            gen, bundle, score = match.groups()
            independent[bundle] = dict(fitness=float(score), score_source='independent_train_reevaluation_log',
                                       score_snapshot_generation=int(gen), score_log_line=line_number)
    return population | independent


def reconstruct(data):
    operators, bundles, rows, offspring = {}, {}, [], []
    def key(b):
        return tuple(b['operators'][t]['name'] for t in TYPES)
    initial_generation = recover_initial_population(data)
    for g in [initial_generation] + data['generations']:
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

    # The cache audit recovers both inputs, including donors omitted by parent_name.
    recovery_path = ROOT.parent / 'analysis/709715_crossover_recovery/crossover_parents.json'
    recovered = json.loads(recovery_path.read_text())
    if len(recovered) != 73 or any(r['status'] != 'confirmed_exact_code' for r in recovered):
        raise ValueError('Expected exact recovery for all 73 crossovers')
    crossover_parents = {r['child']: (r['parent1'], r['parent2']) for r in recovered}

    def operator_parents(op):
        if op['mode'] == 'crossover':
            parents = crossover_parents[op['name']]
            assert parents[0] == op['parent_name']
            return parents
        return (op['parent_name'],) if op['parent_name'] else ()

    # Color every operator contributing to a final component through either input.
    origins = {}
    active = set()
    def visit_operator(name, operator_type):
        if name in BASELINE:
            return
        if name in active:
            raise ValueError('Cycle in operator ancestry')
        if name in origins:
            return
        active.add(name)
        op = operators[name]
        assert op['type'] == operator_type
        for donor in operator_parents(op):
            visit_operator(donor, operator_type)
        active.remove(name)
        origins[name] = operator_type
    for t in TYPES:
        visit_operator(selected['operators'][t]['name'], t)
    # Also follow bundle inheritance and both recovered operator donors recursively.
    creators = {}
    for b in bundles.values():
        if b['key'] != BASELINE:
            creators.setdefault(event(b)['name'], b)
    ancestors = set()
    def visit(b):
        if b['key'] in ancestors:
            return
        ancestors.add(b['key'])
        if b['key'] == BASELINE:
            return
        if birth(b) > 0:
            visit(parent(b))
        op = event(b)
        if op['mode'] in ('refine', 'simplify', 'crossover'):
            for donor_name in operator_parents(op):
                donor = creators.get(donor_name)
                if donor is not None:
                    visit(donor)
    visit(selected)
    # Use original logged fitness for all nine initial proposals and the baseline.
    initial = [b for b in bundles.values() if birth(b) == 0]
    plotted = []
    for b in initial + offspring:
        op = (dict(name='baseline', mode='baseline', type='')
              if b['key'] == BASELINE else event(b))
        score = b['score']
        if score is None or not math.isfinite(float(score)):
            raise ValueError(f'Missing/nonfinite fitness for {op["name"]}')
        plotted.append(dict(generation=birth(b), fitness=float(score), mode=op['mode'],
                            edited_operator=op['type'], operator_name=op['name'],
                            status='operator_origin' if op['name'] in origins else
                                   'ancestor' if b['key'] in ancestors else 'not_recorded_ancestor',
                            color_operator=origins.get(op['name'], ''),
                            score_source='initial_evaluation_log' if birth(b) == 0 else 'offspring',
                            score_snapshot_generation=b['seen'], bundle=' | '.join(b['key'])))
    reevaluations = reevaluated_scores()
    for point in plotted:
        point['original_fitness'] = point['fitness']
        point['score_log_line'] = None
        if point['bundle'] in reevaluations:
            point.update(reevaluations[point['bundle']])
    assert sorted(p['generation'] for p in plotted if p['color_operator'] == 'loss') == [0, 8, 34]
    return plotted, dict(selected_bundle=selected_name, validation_score=val_score,
                         recorded_ancestor_bundles=len(ancestors),
                         crossover_parent_source=str(recovery_path.relative_to(ROOT.parent)),
                         recovered_crossovers=len(recovered),
                         initial_bundles=len(initial),
                         initial_score_source='runs/709715/run.log',
                         score_policy='independent training reevaluation, then population reevaluation, then original',
                         reevaluated_points=sum(p['score_log_line'] is not None for p in plotted),
                         colored_origins={t: sorted(p['generation'] for p in plotted if p['color_operator'] == t)
                                          for t in TYPES})


def add_ancestor_commentary(fig, ax, points):
    """Pack wrapped lineage labels into rows above the unchanged plotting area."""
    changes = json.loads((ROOT.parent / 'analysis/709715_crossover_recovery/ancestor_changes.json').read_text())
    ancestors = sorted((p for p in points if p['color_operator']), key=lambda p: p['generation'])
    assert {p['operator_name'] for p in ancestors} == set(changes)
    ancestors = [p for p in ancestors
                 if changes[p['operator_name']]['label'] not in HIDDEN_COMMENTARY_LABELS]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    rows, labels = [], []
    for point in ancestors:
        label = textwrap.fill(changes[point['operator_name']]['label'], width=19,
                              break_long_words=False, break_on_hyphens=False)
        text = ax.text(point['generation'], 1.04, label, fontsize=7.5 * SCALE,
                       ha='center', va='bottom', transform=ax.get_xaxis_transform(),
                       clip_on=False, linespacing=1.15, zorder=7)
        box = text.get_window_extent(renderer)
        # Keep edge labels inside the plot width while retaining vertical connectors.
        shift = max(bounds.x0 - box.x0, 0) - max(box.x1 - bounds.x1, 0)
        xpixel = ax.transData.transform((point['generation'], 0))[0] + shift
        text.set_x(ax.transData.inverted().transform((xpixel, bounds.y0))[0])
        left, right = box.x0 + shift, box.x1 + shift
        padding = 7 * SCALE * fig.dpi / 72
        row = next((i for i, intervals in enumerate(rows)
                    if all(right + padding < a or left > b + padding for a, b in intervals)), len(rows))
        if row == len(rows):
            rows.append([])
        rows[row].append((left, right))
        labels.append((point, text, row, box.height))
    # Add physical space above the old axes; keep its size and all existing elements.
    width, height = fig.get_size_inches()
    old = ax.get_position()
    row_height = max(item[3] for item in labels) / fig.dpi + 0.14 * SCALE
    extra = len(rows) * row_height + 0.18 * SCALE
    fig.set_size_inches(width, height + extra)
    ax.set_position([old.x0, old.y0 * height / (height + extra), old.width,
                     old.height * height / (height + extra)])
    axes_height = old.height * height
    for point, text, row, _ in labels:
        y = 1 + (0.14 * SCALE + row * row_height) / axes_height
        text.set_y(y)
        ax.plot([point['generation'], point['generation']], [point['fitness'], y - 0.012],
                transform=ax.get_xaxis_transform(), color='black',
                linewidth=0.55 * SCALE, alpha=1.0, clip_on=False, zorder=1.8)
    # Verify packed labels stay inside the exported canvas and don't overlap.
    fig.canvas.draw()
    boxes = [text.get_window_extent(fig.canvas.get_renderer()) for _, text, _, _ in labels]
    for i, box in enumerate(boxes):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
        assert not any(box.overlaps(other) for other in boxes[i + 1:])


def render(points, edges=None, filename="offspring_ancestry.pdf", commentary=False):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10 * SCALE,
                         'pdf.fonttype': 42, 'axes.labelsize': 12 * SCALE,
                         'xtick.labelsize': 10 * SCALE, 'ytick.labelsize': 10 * SCALE,
                         'axes.linewidth': 0.8 * SCALE,
                         'xtick.major.size': 3.5 * SCALE, 'ytick.major.size': 3.5 * SCALE,
                         'xtick.major.width': 0.8 * SCALE, 'ytick.major.width': 0.8 * SCALE,
                         'xtick.major.pad': 3.5 * SCALE, 'ytick.major.pad': 3.5 * SCALE,
                         'axes.labelpad': 4 * SCALE})
    fig, ax = plt.subplots(figsize=(7.2 * SCALE, 4.2 * SCALE))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.97)
    if edges is not None:
        by_id = {p['node_id']: p for p in points}
        for emphasis in (False, True):
            group = [e for e in edges if e['final_ancestry'] == emphasis
                     and e.get('line_width', 1) > 0]
            segments = [[(by_id[e[end]]['generation'], by_id[e[end]]['fitness'])
                         for end in ('parent', 'child')] for e in group]
            ax.add_collection(LineCollection(
                segments, colors='#666666' if emphasis else '#999999',
                linewidths=[e.get('line_width', 0.65 if emphasis else 0.4) * SCALE for e in group],
                alpha=0.40 if emphasis else 0.13, zorder=1.7 if emphasis else 1.6,
                linestyles=['dashed' if e['certainty'] == 'ambiguous' else 'solid' for e in group]))
    final_name = 'streamlined_niche_clone_tournament_gen43_3'
    for status in ('not_recorded_ancestor', 'ancestor', 'operator_origin'):
        group = [p for p in points if p['status'] == status and p['operator_name'] != final_name]
        colors = [COLORS[p['color_operator']] if p['color_operator'] else
                  '#bdbdbd' for p in group]
        ax.scatter([p['generation'] for p in group], [p['fitness'] for p in group],
                   marker='o', s=(66 if status == 'operator_origin' else 28) * SCALE**2,
                   c=colors, edgecolors='none', alpha=1 if status == 'operator_origin' else 0.70,
                   zorder=4 if status == 'operator_origin' else 2)
    final = [p for p in points if p['operator_name'] == final_name]
    assert len(final) == 1
    point = final[0]
    ax.scatter([point['generation']], [point['fitness']], marker='*', s=155 * SCALE**2,
               c=COLORS[point['color_operator']], edgecolors='black', linewidths=0.7 * SCALE, zorder=4.5)
    ax.set(xlabel='Generation', ylabel='Fitness (GT)', xlim=(-1, 46), ylim=(0, 1))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(color='#e9ecf0', lw=0.8 * SCALE)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_color('#bfc5cd')
    draw_ancestry_legend(ax, 8 * SCALE)
    if commentary:
        add_ancestor_commentary(fig, ax, points)
    fig.savefig(OUT / filename, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-from', type=Path, help='Re-extract compact metadata from run_data.json')
    args = parser.parse_args()
    if SCALE <= 0:
        raise ValueError('SCALE must be positive')
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
