# SRBench black-box frontiers

Reproduce with `python figures/plot_srbench_black_box_frontiers.py`.

- Base PySR: runs/290227. Evolved BasicSR GT-R2: runs/271625, sourced from runs/150815.
- Both have 122 tasks × 10 trials, 1M evaluation budgets and 1500-second black-box timeouts.
- At each integer C from 1 to 40, take the maximum finite saved test R² among equations with native complexity ≤ C, independently per trial. Median over trials, then median (main figure) or mean (companion) over tasks. No clipping, missing-trial omission, or extrapolation beyond C=40. All trials have a complexity-1 candidate. The mean version is a mean of trial medians, not a mean of all trials.
- Bands are pointwise 95% intervals from 2000 task-bootstrap samples. They describe variation across tasks, not uncertainty of selecting an equation.
- Reference diamonds use the published per-task trial medians in srbench/docs/csv/blackbox_results_datasets.csv, then the same across-task statistic. All 13 complete symbolic methods are included. AIFeynman is excluded because it lacks 15 tasks; non-symbolic ML methods are outside this equation-complexity comparison.
- Reference x coordinates aggregate each method's chosen model sizes; they are NOT a bound applying to every task. Published data has selected models, not within-trial frontiers. Therefore these points cannot establish dominance against the local size-constrained curves.
- Local curves select using held-out test R² (an oracle envelope, as requested); published models use the original selection protocol. Search budgets, operators and split protocols also differ. The overlay is descriptive, not a controlled method ranking.
- The paper defines complexity as operators + features + constants. These are conceptually tree sizes, but native binary operators, powers, simplification and scaling can change counts. Local saved native complexity and released SRBench model_size are retained without claiming exact equivalence; recounting only the saved test-Pareto models could miss discarded candidates under a new measure.
- The main median aggregation follows the local SRBench Figure 1 plotting notebook (estimator=np.median); its paper caption instead says mean of medians, which the companion implements. This is a replacement for the requested rank plot, not a reproduction of Figures 1–2 or their training-time panel.

Source paper: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c0c7c76d30bd3dcaefc96f40275bdc0a-Paper-round1.pdf

The CSVs retain exact values (including negatives), sample counts and reference confidence intervals. The left panels zoom to R² ∈ [0, 1]; the right panels show all reference point estimates. Source hashes and evaluation metadata are in provenance.json.
