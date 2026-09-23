#!/usr/bin/env python3
"""Reconstruct and plot all offspring edges; preserve the dots-only PDF."""
import copy
import json
from collections import Counter

import plot_709715_offspring_ancestry as plot

# Maximum line width in points, before the shared figure SCALE is applied.
MAX_LINE_WIDTH = 1.0


def weight_edges(points, edges):
    """Count unique reachable nodes through each child; normalize by child generation."""
    if MAX_LINE_WIDTH <= 0:
        raise ValueError('MAX_LINE_WIDTH must be positive')
    by_id = {p['node_id']: p for p in points}
    children = {n: set() for n in by_id}
    for e in edges:
        if e['certainty'] == 'confirmed':
            children[e['parent']].add(e['child'])
    descendants = {n: set() for n in by_id}
    for p in sorted(points, key=lambda p: p['generation'], reverse=True):
        n = p['node_id']
        for child in children[n]:
            assert by_id[child]['generation'] > p['generation']
            descendants[n].add(child)
            descendants[n].update(descendants[child])
        p['descendant_count'] = len(descendants[n])
    maxima = {}
    for e in edges:
        # Include the child itself: an edge ending in a leaf has count 1 and vanishes.
        count = 1 + len(descendants[e['child']])
        generation = by_id[e['child']]['generation']
        e['branch_descendant_count'] = count
        maxima[generation] = max(maxima.get(generation, 0), count)
    for e in edges:
        maximum = maxima[by_id[e['child']]['generation']]
        e['generation_max_descendant_count'] = maximum
        e['line_width'] = (MAX_LINE_WIDTH * (e['branch_descendant_count'] - 1) / (maximum - 1)
                           if maximum > 1 else 0.0)


def main():
    data = json.loads((plot.OUT / 'lineage_records.json').read_text())
    points, _ = plot.reconstruct(data)
    initial = plot.recover_initial_population(data)
    generations = [initial] + data['generations']
    def key(bundle):
        return ' | '.join(bundle['operators'][t]['name'] for t in plot.TYPES)
    by_bundle = {p['bundle']: p for p in points}
    assert len(by_bundle) == len(points)
    for g in generations:
        entries = g['population'] if g['generation'] == 0 else g['offspring']
        for index, b in enumerate(entries):
            p = by_bundle[key(b)]
            p['offspring_index'] = index
            p['node_id'] = f"({g['generation']},{index})"
    by_id = {p['node_id']: p for p in points}
    creators = {p['operator_name']: p for p in points}
    baseline = creators['baseline']
    recovery = json.loads((plot.ROOT.parent / 'analysis/709715_crossover_recovery/crossover_parents.json').read_text())
    crossovers = {r['child']: (r['parent1'], r['parent2']) for r in recovery}
    relationships, ambiguous, unresolved = [], [], []

    def add(parent, child, kind, certainty='confirmed', **extra):
        assert parent['generation'] < child['generation']
        relationships.append(dict(parent=parent['node_id'], child=child['node_id'],
                                  kind=kind, certainty=certainty, **extra))

    for previous, g in zip(generations, generations[1:]):
        assert g['generation'] == previous['generation'] + 1
        for b in g['offspring']:
            child = by_bundle[key(b)]
            t, mode = child['edited_operator'], child['mode']
            op = b['operators'][t]
            counts = copy.deepcopy(b['meta_mutation_counts'])
            counts[t][mode] -= 1
            # Parents are selected from the surviving population entering this generation.
            candidates = {key(p): p for p in previous['population']
                          if p['meta_mutation_counts'] == counts and
                          all(p['operators'][other]['name'] == b['operators'][other]['name']
                              for other in plot.TYPES if other != t)}
            # All bundles here have four explicit operators, so refine/simplify
            # use the edited component of the selected bundle (no missing-op fallback).
            if mode in ('refine', 'simplify'):
                candidates = {k: p for k, p in candidates.items()
                              if p['operators'][t]['name'] == op['parent_name']}
            certainty = 'confirmed' if len(candidates) == 1 else 'ambiguous'
            if not candidates:
                unresolved.append(child['node_id'])
            elif len(candidates) > 1:
                ambiguous.append(dict(child=child['node_id'],
                                      candidates=[by_bundle[k]['node_id'] for k in candidates]))
            for k in candidates:
                add(by_bundle[k], child, 'bundle_inheritance', certainty)
            if mode in ('refine', 'simplify', 'crossover'):
                donors = crossovers[op['name']] if mode == 'crossover' else (op['parent_name'],)
                for slot, donor in enumerate(donors, 1):
                    parent = baseline if donor in plot.BASELINE else creators[donor]
                    add(parent, child, 'operator_input', operator=t, parent_slot=slot,
                        operator_name=donor)

    # Coalesce identical endpoints, preserving every relationship/parent slot in JSON.
    # A confirmed relationship makes a segment definite even if another role is ambiguous.
    segments = {}
    for r in relationships:
        k = (r['parent'], r['child'])
        e = segments.setdefault(k, dict(parent=k[0], child=k[1], certainty=r['certainty']))
        if r['certainty'] == 'confirmed':
            e['certainty'] = 'confirmed'
    selected = next(p for p in points if p['operator_name'] == 'streamlined_niche_clone_tournament_gen43_3')
    reached = {selected['node_id']}
    while True:
        expanded = reached | {e['parent'] for e in segments.values()
                              if e['child'] in reached and e['certainty'] == 'confirmed'}
        if expanded == reached:
            break
        reached = expanded
    for e in segments.values():
        e['final_ancestry'] = (e['child'] in reached and e['certainty'] == 'confirmed')
    assert not unresolved, unresolved
    assert len(points) == 459 and len(by_id) == 459
    assert len({r['child'] for r in relationships if r['kind'] == 'bundle_inheritance'}) == 449
    weight_edges(points, list(segments.values()))
    summary = dict(nodes=len(points), relationships=len(relationships),
                   total_segments=len(segments),
                   drawn_segments=sum(e['line_width'] > 0 for e in segments.values()),
                   max_line_width=MAX_LINE_WIDTH,
                   width_normalization='child generation; (branch count - 1) / (generation maximum - 1)',
                   ambiguous_bundle_parents=ambiguous,
                   unresolved_bundle_parents=unresolved,
                   relationship_types=dict(Counter(r['kind'] for r in relationships)))
    graph = dict(summary=summary, nodes=points, relationships=relationships,
                 segments=list(segments.values()))
    (plot.OUT / 'offspring_dag.json').write_text(json.dumps(graph, indent=2) + '\n')
    plot.render(points, list(segments.values()), filename='offspring_ancestry_dag.pdf')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
