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

    recovery = ROOT / 'analysis/709715_crossover_recovery'
    crossovers = json.loads((recovery / 'crossover_parents.json').read_text())
    assert len(crossovers) == 73 and all(r['status'] == 'confirmed_exact_code' for r in crossovers)
    crossover_parents = {r['child']: [r['parent1'], r['parent2']] for r in crossovers}
    library = json.loads((recovery / 'operator_library.json').read_text())
    for n, op in library.items():
        operators.setdefault(n, {k: v for k, v in op.items() if k != 'code'})
    changes = json.loads((recovery / 'ancestor_changes.json').read_text())

    def operator_parents(n):
        return crossover_parents.get(n, [operators[n]['parent_name']] if operators[n]['parent_name'] else [])

    def ancestors(root):
        reached = set()
        def visit(n):
            if n in reached:
                return
            reached.add(n)
            for parent in operator_parents(n):
                visit(parent)
        visit(root)
        return sorted((operators[n] for n in reached), key=lambda op: (op['generation'], op['name']))

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

    ancestor_names = {op['name'] for t in TYPES for op in ancestors(selected['ops'][t])}
    evolved = ancestor_names - set(BASELINE.values())
    assert evolved == set(changes), (evolved - set(changes), set(changes) - evolved)
    lines += ['', '## What changed at each ancestral step', '',
        'These 15 evolved operators include both crossover branches, not just the '
        'bundle-inheritance path above. Labels summarize code changes, not measured '
        'fitness gains; simplification steps can remove mechanisms rather than add them. '
        'Generation 16 is a donor branch; generation 11 contributes through the second '
        'parent of the generation 20 crossover.', '',
        '| Generation | Operator | Short description | One-sentence explanation |',
        '| --- | --- | --- | --- |']
    for n in sorted(evolved, key=lambda n: (operators[n]['generation'], n)):
        op, change = operators[n], changes[n]
        assert 1 <= len(change['label'].split()) <= 5
        source = next((path for path in sorted((RUN / 'operators').glob(f"gen{op['generation']}_{op['type']}*.jl"))
                       if n in path.read_text()), None)
        assert source, n
        lines.append(f"| {op['generation']} | [{op['type']}: `{n}`](../{source.relative_to(ROOT)}) | "
                     f"{change['label']} | {change['explanation']} |")

    lines += ['', '## Operator ancestry and crossover inputs', '',
        'Both crossover parents are recovered for **all 73 crossovers** by matching '
        'cached response code to the saved child (undoing its generated function-name '
        'suffix), then matching both parent code blocks in that request to saved '
        'operators. Every match is exact after trimming outer whitespace. '
        'The generation 19 survival crossover used the same baseline operator for '
        'both inputs. Explore events with no parent metadata are new proposals.', '',
        'See the [complete crossover audit](../analysis/709715_crossover_recovery/README.md) '
        'for all parent pairs and the corresponding prompt evidence.', '']
    for t in TYPES:
        lines += [f'### {t.capitalize()}', '',
                  '| Generation | Method | Operator | Operator parent(s) |',
                  '| --- | --- | --- | --- |']
        for op in ancestors(selected['ops'][t]):
            method = 'baseline' if op['name'] in BASELINE.values() else op['mode']
            parent = '; '.join(operator_parents(op['name'])) or '—'
            label = op['name']
            if label == selected['ops'][t]:
                label = f'**{label}**'
            lines.append(f'| {op["generation"]} | {method} → {t} | {label} | {parent} |')
        lines.append('')

    lines += ['## Recorded descendants of the selected operators', '',
        'These are all direct and indirect descendants reachable through '
        'saved parents and both recovered crossover inputs for the four selected components, through generation 45. '
        'They can predate the assembly of the final bundle in generation 43. '
        'These tables describe operator descent, not '
        'necessarily descent of the complete selected bundle.', '']
    total_descendants = 0
    for t in TYPES:
        root = selected['ops'][t]
        reached = {root}
        while True:
            expanded = reached | {n for n, op in operators.items()
                                  if any(p in reached for p in operator_parents(n))}
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
                         f'{op["name"]} | {"; ".join(operator_parents(op["name"]))} |')
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
        '- [Crossover recovery](../scripts/recover_709715_crossover_parents.py): exact cached request/response matching.',
        '- [Change descriptions](../analysis/709715_crossover_recovery/ancestor_changes.json): manually reviewed source-code summaries.',
        '- [Report generator](../scripts/trace_709715_lineage.py).', '',
        'The bundle path is reconstructed by matching the three unchanged components '
        'and all inherited edit counts against earlier records. Refine/simplify also '
        'require the replaced component to match the saved operator parent. Each '
        'of the 13 transitions after initialization has exactly one matching parent '
        'bundle. Initialization is the generation 0 loss exploration from the baseline. '
        'The operator tables combine explicit parent metadata with exact cached-code evidence for both crossover inputs.', '']
    output = ROOT / 'out/709715-lineage.md'
    output.write_text('\n'.join(lines))
    print(f'Wrote {output}: {len(chain)} steps, {total_descendants} operator descendants; '
          f'{len(definite)} confirmed and {len(possible)} ambiguous later bundle descendants.')


if __name__ == '__main__':
    main()
