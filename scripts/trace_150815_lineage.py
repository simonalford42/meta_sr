"""Generate the BasicSR 150815 generation summary and recorded lineage reports."""
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bundle_loader import load_skeleton_bundle  # noqa: E402


def main():
    run = ROOT / 'runs/150815'
    data = json.loads((run / 'run_data.json').read_text())
    log = (run / 'run.log').read_text()
    selected = load_skeleton_bundle(str(run))
    slots = tuple(data['config']['operator_slots'])
    operators, bundles, offspring = {}, {}, []

    def identity(entry):
        return tuple(entry['functions'][slot]['name'] for slot in slots)

    for generation in data['generations']:
        for kind in ('population', 'offspring'):
            for entry in generation[kind]:
                names = identity(entry)
                bundles.setdefault(names, entry)
                for function in entry['functions'].values():
                    metadata = {k: v for k, v in function.items() if k != 'code'}
                    if function['name'] in operators:
                        assert operators[function['name']] == metadata
                    operators[function['name']] = metadata
                if kind == 'offspring':
                    offspring.append((generation['generation'], names))
    baseline = tuple(next(n for n, op in operators.items()
                          if op['slot'] == slot and op['parent_name'] is None)
                     for slot in slots)
    target = tuple(selected.functions[slot].name for slot in slots)
    assert target in bundles
    initial = re.search(r'^Best initial bundle: (.+) \(score=([\d.]+)\)$', log, re.M)
    records = [(0, tuple(initial[1].split(' | ')), initial[2])]
    for generation in data['generations']:
        g = generation['generation']
        match = re.search(rf'^Generation {g} complete: best=(.+) \(score=([\d.]+)\)', log, re.M)
        assert match and match[1] == generation['best_name']
        assert f'{generation["best_score"]:.4f}' == match[2]
        records.append((g, tuple(match[1].split(' | ')), match[2]))
    assert [g for g, _, _ in records] == list(range(31))
    summary = ['summary:',
               f'baseline: {" | ".join(baseline)} (score: {data["baseline"]["score"]:.4f})']
    previous = baseline
    for g, names, score in records:
        if names == previous:
            description = 'no improvement'
        else:
            description = ' | '.join(
                f'**{n}**' if n not in baseline and operators[n]['generation'] == g else n
                for n in names) + f' (score: {score})'
        summary.append(f'{g}: {description}')
        previous = names
    summary += ['Slot order: ' + ' | '.join(slots) + '.',
                'Scores are the logged training **gt-r2** fitness, not pure GT match rates. '
                '“No improvement” means the best bundle is unchanged from the preceding generation. '
                'Bold marks a function introduced in the indicated generation. '
                'Source: [run.log](../runs/150815/run.log).']
    (ROOT / 'out/150815-scratch2.md').write_text('\n\n'.join(summary) + '\n')

    def event(names):
        evolved = [operators[n] for n in names if operators[n]['parent_name']]
        return max(evolved, key=lambda op: op['generation']) if evolved else None

    def parent(names):
        op = event(names)
        result = list(names)
        result[slots.index(op['slot'])] = op['parent_name']
        return tuple(result)

    chain = []
    names = target
    while names != baseline:
        op = event(names)
        assert op is not None
        chain.append((names, op))
        predecessor = parent(names)
        assert predecessor in bundles or predecessor == baseline
        expected = copy.deepcopy(bundles[names]['meta_mutation_counts'])
        expected[op['slot']][op['mode']] -= 1
        if predecessor in bundles:
            assert expected == bundles[predecessor]['meta_mutation_counts']
        else:
            assert all(v == 0 for modes in expected.values() for v in modes.values())
        names = predecessor
    chain.reverse()
    assert len(chain) == 18
    selected_generation = event(target)['generation']
    report = ['# Run 150815: lineage of the default BasicSR evaluation bundle', '',
        '`srbench_full_eval.py --evolve-results runs/150815` defaults to '
        '`--select-by val`. The actual `load_skeleton_bundle` loader selects the '
        f'**generation {selected_generation}** bundle below: validation '
        f'**{selected.val_score:.4f}**, saved training score **{selected.score:.4f}**.', '',
        'This run uses **gt-r2** fitness and eight BasicSR function slots. '
        'These scores are not pure GT match rates. The recorded training budget '
        'was 500 seconds and the validation budget was 1,500 seconds.', '',
        '| Slot | Selected function |', '| --- | --- |']
    for slot, n in zip(slots, target):
        report.append(f'| {slot} | **{n}** |')
    report += ['', '## Bundle lineage, in creation order', '',
        'Each generation entry descends from the preceding bundle. The creation '
        'method identifies the edited slot; the new function is bolded. '
        'Function order is: ' + ' | '.join(slots) + '.', '',
        '**Baseline:** ' + ' | '.join(baseline), '']
    for names, op in chain:
        label = ' (initial population)' if op['generation'] == 0 else ''
        report += [f'**Generation {op["generation"]}{label}: '
                   f'{op["mode"]} → {op["slot"]}**', '',
                   ' | '.join(f'**{n}**' if n == op['name'] else n for n in names), '']
    report += ['## Recorded component ancestry', '',
        '`parent_name` records the function replaced in the parent bundle, including '
        'for explore events. Explore receives full-bundle context but no dedicated '
        'parent implementation to edit. For crossover, the recorded parent is '
        'the first input; the second input was not saved. All crossover events on '
        'this path occur after generation 3, when prompt logging had stopped. '
        'Their second parents cannot be recovered from the saved metadata/prompts.', '',
        'The **crossover method** combines implementations of a function slot. '
        'The **crossover slot** controls how BasicSR crosses symbolic expressions; '
        'these are distinct uses of the word.', '']
    for slot, root in zip(slots, target):
        ancestors = []
        n = root
        while n:
            op = operators[n]
            ancestors.append(op)
            n = op['parent_name']
        report += [f'### {slot}', '',
                   '| Generation | Creation method | Function | Recorded parent |',
                   '| --- | --- | --- | --- |']
        for op in reversed(ancestors):
            method = 'baseline' if op['parent_name'] is None else f'{op["mode"]} → {slot}'
            p = op['parent_name'] or '—'
            if op['parent_name'] and op['mode'] == 'crossover':
                p += '; second parent not recorded'
            label = f'**{op["name"]}**' if op['name'] == root else op['name']
            report.append(f'| {op["generation"]} | {method} | {label} | {p} |')
        report.append('')
    report += ['## Recorded descendants of the selected functions', '',
        'All direct and indirect descendants reachable via `parent_name` through '
        'generation 30 are listed below. A descendant of an individual selected '
        'function may predate the complete generation 29 bundle. Explore edges '
        'mean replacement within an inherited bundle, rather than direct code '
        'refinement. Crossover relationships through unrecorded second parents '
        'are unavailable.', '']
    count = 0
    for slot, root in zip(slots, target):
        reached = {root}
        while True:
            expanded = reached | {n for n, op in operators.items() if op['parent_name'] in reached}
            if reached == expanded:
                break
            reached = expanded
        descendants = sorted((operators[n] for n in reached - {root}),
                             key=lambda op: (op['generation'], op['name']))
        count += len(descendants)
        report += [f'### {slot}: {len(descendants)} descendants', '', f'Root: **{root}**.', '']
        if not descendants:
            report += ['No descendants recorded through generation 30.', '']
            continue
        report += ['| Generation | Creation method | Descendant | Recorded parent |',
                   '| --- | --- | --- | --- |']
        for op in descendants:
            report.append(f'| {op["generation"]} | {op["mode"]} → {slot} | '
                          f'{op["name"]} | {op["parent_name"]} |')
        report.append('')
    report += ['## Later descendants of the complete selected bundle', '']
    reached = {target}
    later = []
    for g, names in sorted(offspring):
        if g > selected_generation and parent(names) in reached:
            reached.add(names)
            later.append((g, names, event(names)))
    for g, names, op in later:
        report += [f'**Generation {g}: {op["mode"]} → {op["slot"]}**', '',
                   ' | '.join(f'**{n}**' if n == op['name'] else n for n in names), '']
    if not later:
        report += ['No descendants of the complete generation 29 bundle are recorded '
                   'in generation 30.', '']
    report += ['## Sources and verification', '',
        '- [run_data.json](../runs/150815/run_data.json): functions, parent links, '
        'generation/mode metadata, inherited edit counts, validation scores.',
        '- [run.log](../runs/150815/run.log): initial population and generation summaries.',
        '- [bundle_loader.py](../bundle_loader.py): actual `load_skeleton_bundle` '
        'default validation selection, matched by rendered bundle content hash.',
        '- [evolve_fullsr.py](../evolve_fullsr.py): `_finish_candidate` records '
        'the replaced function as parent; crossover also draws a second implementation.',
        '- [skeleton_operator_types.py](../skeleton_operator_types.py): '
        '`SkeletonBundle.copy_with` inherits seven functions and increments one edit count.',
        '- [Saved prompts](../runs/150815/prompts): available through generation 3.',
        '- [Report generator](../scripts/trace_150815_lineage.py).', '',
        f'All {len(chain)} lineage steps were checked against recorded bundle identities '
        'and inherited edit counts. The final step back to the baseline was checked '
        'against zero edit counts. The generation summary covers 0–30 and agrees '
        'with both the log and saved best scores.', '']
    (ROOT / 'out/150815-lineage.md').write_text('\n'.join(report))
    print(f'Created both reports: {len(chain)} lineage steps, {count} function descendants, '
          f'{len(later)} later bundle descendants.')


if __name__ == '__main__':
    main()
