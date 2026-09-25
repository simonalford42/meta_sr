# Offspring ancestry graph

Regenerate from the repository root:

```sh
python figures/plot_709715_offspring_dag.py
```

Output: `offspring_ancestry_dag.pdf`. The dots-only variant is `figures/pysr_evolution.pdf` (formerly `offspring_ancestry.pdf`). Both use the same point positions and styling; `SCALE` in `figures/plot_709715_offspring_ancestry.py` controls their size.

`offspring_dag.json` stores 459 nodes and typed relationships. Node IDs are `(generation, index)`, with zero-based indices into the saved offspring array; generation 0 uses the initial evaluation log order, including the baseline. These indices are not necessarily the generation slot suffixes in operator names.

Edges mean:

- **Bundle inheritance:** the previous generation's surviving bundle matching the three unchanged operators and decremented edit counts; refine/simplify also require the edited operator to match the recorded parent.
- **Operator input:** the creation event of the source operator for refine/simplify, or each of the two exactly recovered crossover inputs. This identifies the origin of the code, not which later bundle carried that code when sampled. Baseline operator inputs point to the initial baseline node.

Edges point forward in generation. Endpoints use each node's plotted reevaluated fitness where available, not its fitness at the time it was selected as a parent. Initial candidates are roots. Identical endpoint pairs are drawn once, but their separate roles and crossover parent slots remain in the JSON.

There are 447 uniquely identified bundle parents and two ambiguous generation-1 exploration events: `(1,2)` has four possible parents; `(1,3)` has three. Their alternative inheritance edges are dashed rather than choosing a parent arbitrarily. All 449 offspring have at least one bundle-parent candidate. All 73 crossovers have both operator inputs.

The graph contains 801 distinct segments; the weighted PDF draws 204. Each edge's branch count is its child plus all unique descendants of that child, following confirmed edges of both relationship types. Shared descendants reached through several paths count once. Ambiguous relationships do not propagate descendant counts; an ambiguous candidate edge is weighted by its child's confirmed descendants and remains dashed.

An edge ending in a leaf has branch count 1 and is hidden. For each child generation separately, width is:

```text
MAX_LINE_WIDTH * (branch_count - 1) / (largest_branch_count_in_that_generation - 1)
```

Generations containing only leaf children have no lines. Set `MAX_LINE_WIDTH = 1.0` (points, before figure `SCALE`) in `figures/plot_709715_offspring_dag.py` to change the maximum thickness. Normalization groups edges by their child's generation, not their parent's age or the generations crossed by a long edge.

All dots remain. Confirmed paths into the final selected bundle retain darker grey, and other edges remain transparent grey. Opacity can be adjusted in `render()` in `figures/plot_709715_offspring_ancestry.py`. Counts, normalization denominators, and widths are saved in `offspring_dag.json`.

The selected final algorithm at generation 43 is shown as a larger yellow star; other offspring use smaller grey dots.

Bundle-only ancestors use the same light grey dot style as other offspring; only ancestors of the final operators receive colored markers. Their ancestry relationships remain in the graph.

The DAG PDF includes 7 of the 15 short ancestor descriptions plus a “Final best algorithm” label from `analysis/709715_crossover_recovery/ancestor_changes.json`, shared with the lineage report. Wrapped labels are packed into nonoverlapping rows above the original-size plotting area, with vertical black connectors. Adjust font size, wrapping width, and row spacing in `add_ancestor_commentary()` in the shared plotting script.

Fitness now prefers the latest logged independent training reevaluation for each exact bundle, then the latest logged population seed top-up, then original evaluation if neither exists. This updates 156 of 459 points; 303 have no recorded reevaluation. `original_fitness`, `score_source`, `score_snapshot_generation`, and `score_log_line` preserve provenance in the CSV and graph JSON. Generation positions remain creation generations. Validation scores are not used on the fitness axis.

`HIDDEN_COMMENTARY_LABELS` in the shared plotting script controls omitted annotations; currently generations 9, 10, 18, 19, 27, 34, and 40 are omitted; generation 43 uses “Final best algorithm” instead of its change description. Their dots and DAG edges remain. The final star has a black outline.
