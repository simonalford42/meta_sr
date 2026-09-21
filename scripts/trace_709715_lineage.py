"""Reconstruct the default evaluation bundle's recorded lineage without evaluation."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/709715'
TYPES = ('mutation', 'survival', 'selection', 'loss')
BASELINE = dict(zip(TYPES, ('add_constant_offset', 'age_regularized_survival',
                           'tournament_selection', 'mse_loss')))


def main():
    data = json.loads((RUN / 'run_data.json').read_text())
    operators, bundles, rows = {}, {}, []
    def key(b):
        return tuple(b['ops'][t] for t in TYPES)
    def name(b):
        return ' | '.join(key(b))
    def collect(entry, generation, kind):
        b = {'ops': {t: entry['operators'][t]['name'] for t in TYPES},
             'counts': entry['meta_mutation_counts'], 'seen': generation,
             'kind': kind, 'score': entry.get('score')}
        for t, op in entry['operators'].items():
            operators[op['name']] = {k: op.get(k) for k in
                                    ('name', 'generation', 'parent_name', 'mode')}
            operators[op['name']]['type'] = t
        rows.append(b)
        bundles.setdefault(name(b), b)
    for generation in data['generations']:
        for kind in ('population', 'offspring'):
            for entry in generation[kind]:
                collect(entry, generation['generation'], kind)
    if data.get('best_bundle'):
        collect(data['best_bundle'], 45, 'best_bundle')
    eligible = [(n, v) for n, v in data['val_results'].items()
                if n in bundles and v.get('avg_score') is not None]
    selected_name, val = max(eligible, key=lambda item: item[1]['avg_score'])
    selected = bundles[selected_name]
    # This follows bundle_loader._select_best_by_val, including first-entry ties.
    assert selected['ops']['selection'] == 'streamlined_niche_clone_tournament_gen43_3'

    def birth(b):
        return max(operators[n]['generation'] for n in key(b))
    def event(b):
        g = birth(b)
        edited = [operators[n] for n in key(b) if operators[n]['generation'] == g]
        assert len(edited) == 1, (g, edited)
        return edited[0]
    def parents(b):
        op = event(b)
        t, mode = op['type'], op['mode']
        counts = copy.deepcopy(b['counts'])
        counts[t][mode] -= 1
        found = {}
        for p in rows:
            if p['seen'] >= birth(b) or p['counts'] != counts:
                continue
            if any(p['ops'][k] != b['ops'][k] for k in TYPES if k != t):
                continue
            if mode in ('refine', 'simplify') and p['ops'][t] != op['parent_name']:
                continue
            found.setdefault(key(p), p)
        return list(found.values())

    chain = [selected]
    while birth(chain[-1]) > 0:
        candidates = parents(chain[-1])
        assert len(candidates) == 1, (name(chain[-1]), len(candidates))
        chain.append(candidates[0])
    chain.reverse()
    assert len(chain) == 14

    lines = ['# Run 709715: lineage of the default evaluation bundle', '',
        '`srbench_full_eval.py --evolve-results runs/709715` defaults to '
        '`--select-by val`. The selected bundle was created in **generation 43**, '
        f'with persisted validation score **{val["avg_score"]:.4f}**.', '',
        ' | '.join(f'**{n}**' for n in key(selected)), '',
        '## Bundle lineage, in creation order', '',
        'Each row descends from the previous row. Bold marks the operator introduced '
        'at that step; the other three operators are inherited. Generations absent '
        'from this table made no edit on this particular path. Scores are omitted '
        'because reevaluation changes them over time.', '',
        '| Generation | Creation method | Mutation | Survival | Selection | Loss |',
        '| --- | --- | --- | --- | --- | --- |',
        '| baseline | default | ' + ' | '.join(BASELINE.values()) + ' |']
    for b in chain:
        g = birth(b)
        if g == 0:
            edited = next(t for t in TYPES if b['ops'][t] != BASELINE[t])
            method = f'explore → {edited} (initial population)'
        else:
            op = event(b)
            edited = op['type']
            method = f'{op["mode"]} → {edited}'
        cells = [f'**{b["ops"][t]}**' if t == edited else b['ops'][t] for t in TYPES]
        lines.append(f'| {g} | {method} | ' + ' | '.join(cells) + ' |')

    lines += ['', '## Operator ancestry and crossover inputs', '',
        'Bundle inheritance and operator ancestry differ: crossover can draw an '
        'operator from another bundle. The following tables follow the saved '
        '`parent_name` for each final component. For crossover, this is only '
        'parent 1; parent 2 was not persisted. Saved prompts stop after generation '
        '3, so the crossover inputs at generations 18, 19, and 20 cannot be fully '
        'recovered from these records. Explore creates a new proposal; a recorded '
        'baseline reference is not a refine or crossover event.', '']
    for t in TYPES:
        lineage = []
        n = selected['ops'][t]
        while n:
            assert n in operators, n
            op = operators[n]
            lineage.append(op)
            n = op['parent_name']
        lines += [f'### {t.capitalize()}', '',
                  '| Generation | Method | Operator | Recorded operator parent |',
                  '| --- | --- | --- | --- |']
        for op in reversed(lineage):
            method = 'baseline' if op['name'] in BASELINE.values() else op['mode']
            parent = op['parent_name'] or '—'
            if method == 'crossover':
                parent += '; second parent not recorded'
            label = op['name']
            if label == selected['ops'][t]:
                label = f'**{label}**'
            lines.append(f'| {op["generation"]} | {method} → {t} | {label} | {parent} |')
        lines.append('')

    lines += ['## Recorded descendants of the selected operators', '',
        'These are all direct and indirect descendants reachable through '
        '`parent_name` for the four selected components, through generation 45. '
        'They can predate the assembly of the final bundle in generation 43. '
        'Crossover descendants connected only through an unrecorded second parent '
        'cannot be identified. These tables describe operator descent, not '
        'necessarily descent of the complete selected bundle.', '']
    total_descendants = 0
    for t in TYPES:
        root = selected['ops'][t]
        reached = {root}
        while True:
            expanded = reached | {n for n, op in operators.items()
                                  if op['parent_name'] in reached}
            if expanded == reached:
                break
            reached = expanded
        descendants = sorted((operators[n] for n in reached - {root}),
                             key=lambda op: (op['generation'], op['name']))
        total_descendants += len(descendants)
        lines += [f'### {t.capitalize()}: {len(descendants)} descendants', '',
                  f'Root: **{root}**.', '']
        if not descendants:
            lines += ['No descendants recorded through generation 45.', '']
            continue
        lines += ['| Generation | Creation method | Descendant | Recorded parent |',
                  '| --- | --- | --- | --- |']
        for op in descendants:
            lines.append(f'| {op["generation"]} | {op["mode"]} → {t} | '
                         f'{op["name"]} | {op["parent_name"]} |')
        lines.append('')

    later = {name(b): b for b in rows if b['kind'] == 'offspring'
             and b['seen'] > birth(selected)}
    reached_bundles = {key(selected)}
    definite, possible = [], []
    for b in sorted(later.values(), key=birth):
        candidates = parents(b)
        matching = [p for p in candidates if key(p) in reached_bundles]
        if matching and len(matching) == len(candidates):
            definite.append(b)
            reached_bundles.add(key(b))
        elif matching:
            possible.append(b)
    lines += ['## Later descendants of the complete generation 43 bundle', '']
    if not definite and not possible:
        lines += ['No later bundle descendants are supported by the saved '
                  'inheritance and edit-count records for generations 44–45.', '']
    else:
        for label, entries in [('Confirmed', definite), ('Ambiguous', possible)]:
            for b in entries:
                op = event(b)
                lines += [f'- {label}, generation {birth(b)}: '
                          f'{op["mode"]} → {op["type"]}: {name(b)}.']
        lines.append('')
    lines += ['## Sources and reconstruction', '',
        '- [run_data.json](../runs/709715/run_data.json): generation populations, '
        'offspring, operator generation/mode/parent, inherited `meta_mutation_counts`, '
        'and validation results.',
        '- [srbench_full_eval.py](../srbench_full_eval.py): default `--select-by val`.',
        '- [bundle_loader.py](../bundle_loader.py): `_select_best_by_val` selection rule.',
        '- [operator_types.py](../operator_types.py): `OperatorBundle.copy_with` '
        'inherits three operators and increments one edit count.',
        '- [evolve_pysr.py](../evolve_pysr.py): crossover chooses operator parents '
        'independently of the bundle supplying unchanged components.',
        '- [Saved prompts](../runs/709715/prompts): available only through generation 3.',
        '- [Report generator](../scripts/trace_709715_lineage.py).', '',
        'The bundle path is reconstructed by matching the three unchanged components '
        'and all inherited edit counts against earlier records. Refine/simplify also '
        'require the replaced component to match the saved operator parent. Each '
        'of the 13 transitions after initialization has exactly one matching parent '
        'bundle. Initialization is the generation 0 loss exploration from the baseline. '
        'The operator tables use explicit parent metadata rather than guesses from names.', '']
    output = ROOT / 'out/709715-lineage.md'
    output.write_text('\n'.join(lines))
    print(f'Wrote {output}: {len(chain)} steps, {total_descendants} operator descendants; '
          f'{len(definite)} confirmed and {len(possible)} ambiguous later bundle descendants.')


if __name__ == '__main__':
    main()
