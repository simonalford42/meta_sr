"""Build the offline, per-seed one-hour frontier audit (no API calls or jobs).

Run from any directory: python figures/build_empirical_1m_audit.py
Requires sympy and tectonic. Sources are the archived September 8 portfolios.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/empirical_1m_equation_audit"
RUNS = {
    "Base PySR": ("srb_base_port", "Baseline", "runs/srbench2_9-8_gt_baseline_1core_l1_portfolio_1m_no_warmup"),
    "Evolved PySR (709715)": ("srb_evo_port", "709715", "runs/709715/srbench2_9-8_ground_truth_1core_portfolio_1m_no_warmup"),
}
# References use the actual feature order of these SRBench2 runs.
TARGETS = {
    "hubble": ("Hubble", r"v=c x_0", r"x_0=D"),
    "ideal_gas": ("Ideal gas", r"\log P=c_0+\log x_0+\log x_1-\log x_2", r"(x_0,x_1,x_2)=(n,T,V)"),
    "kepler": ("Kepler", r"P=c x_0^{3/2}", r"x_0=a"),
    "leavitt": ("Leavitt", r"M=c_0+c_1x_0", r"x_0=\log_{10}P"),
    "newton": ("Newton", r"\log F=c_0+\log x_1+\log x_2-2\log x_0", r"(x_0,x_1,x_2)=(r,m_1,m_2)"),
    "planck": ("Planck", r"\log B=\log\!\left(\frac{c_0 x_0^3}{\exp(c_1x_0/x_1)-1}\right)", r"(x_0,x_1)=(\nu,T)"),
    "rydberg": ("Rydberg", r"\log\lambda=c_0-\log\!\left(x_0^{-2}-x_1^{-2}\right)", r"(x_0,x_1)=(n_1,n_2),\quad n_2>n_1>0"),
    "schechter": ("Schechter", r"\log\phi=c_0+c_1\log x_0+c_2x_0", r"x_0=L"),
    "bode": ("Bode", r"a=c_0+c_1\exp(c_2x_0)\quad\text{(archived broad criterion allows }c_0=0\text{)}", r"x_0=n"),
}


def read(rel):
    return json.loads((ROOT / rel).read_text())


def esc(value):
    value = ''.join(c for c in str(value) if ord(c) >= 32 or c == '\n')
    value = value.replace('−', '-').replace('·', '*').replace('→', ' -> ').replace('²', '^2')
    table = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
             '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(table.get(c, c) for c in value)


def latex_equation(equation):
    # Only the display copy is rounded. Exact source strings are printed below it.
    local = {'square': lambda x: sp.Pow(x, 2, evaluate=False),
             'cube': lambda x: sp.Pow(x, 3, evaluate=False)}
    expr = sp.sympify(equation, locals=local, evaluate=False)
    expr = expr.xreplace({f: sp.Float(str(f), 8) for f in expr.atoms(sp.Float)})
    return sp.latex(expr, ln_notation=True, mul_symbol='dot')


def number(x):
    return f'{x:.8g}' if isinstance(x, (int, float)) and math.isfinite(x) else 'unavailable'


def candidate_block(title, idx, row):
    return '\n'.join([
        r'\subsubsection*{' + esc(title) + '}',
        esc(f"Frontier index {idx}; complexity {row['complexity']}; training R2 = {number(row.get('r2'))}; native loss = {number(row.get('loss'))}."),
        r'\begin{center}\begin{adjustbox}{max width=\linewidth,max totalheight=1.0in}$\displaystyle ' + latex_equation(row['equation']) + r'$\end{adjustbox}\end{center}',
        r'{\scriptsize Exact saved expression (use this for coefficient checks):}',
        r'\begin{lstlisting}', row['equation'], r'\end{lstlisting}',
    ])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit_rel = 'analysis/benchmark_positive_audit_2026-09-08.json'
    timing_rel = 'reports/srbench2_portfolio_solve_over_time/first_recovery.json'
    audit = {(r['setup'], r['dataset'], r['seed']): r for r in read(audit_rel)['records']}
    final = {(r['method'], r['dataset'], r['seed']): r for r in read(timing_rel)}
    records, sources = [], [audit_rel, timing_rel, 'manual_solve_check.py']
    for method, (setup, timing_name, rel) in RUNS.items():
        sources += [rel + '/srbench_full_results.json', rel + '/manual_solve_check_results.json', rel + '/manifest.json']
        manifest = read(rel + '/manifest.json')
        protocol = manifest['serial_restart_portfolio']
        assert protocol['restart_max_evals'] == 1_000_000
        assert protocol['total_search_budget_seconds'] == 3600
        assert protocol['warmup_excluded_from_budget']
        reviews = {(r['dataset'], r['seed']): r for r in read(rel + '/manual_solve_check_results.json')['reviews']}
        results = read(rel + '/srbench_full_results.json')['results']
        for result in results.values():
            dataset, seed = result['dataset'], result['seed']
            task = dataset.removeprefix('first_principles_')
            if task not in TARGETS:
                continue
            assert not result.get('error')
            frontier = result['pareto_frontier']
            old = reviews[dataset, seed]
            checked = audit[setup, dataset, seed]
            label = checked['audited']
            if task != 'bode':
                assert final[timing_name, dataset, seed]['final_classification'] == label
            selected = checked['selected_equation']
            reason = checked['reason'] if checked['checked'] else old['explanation']
            selection_source = 'Archived review / positive-equation audit'
            if selected is None:
                # The only unselected miss in the nine-task subset. Choose the
                # compact log(n1) + ratio correction, not an invented witness.
                assert task == 'rydberg' and seed == 10000 and method == 'Base PySR'
                selected = frontier[7]['equation']
                assert selected == '((cube(x0 / x1) + -8.119437542038874) + log(x0)) / 0.501958254706728'
                selection_source = 'Report-selected structural proxy (review selected none)'
                reason += (' Report proxy: compact log(n1) plus a ratio-dependent correction. '
                           'Its cubic correction is not -log(1-(n1/n2)^2), and its log coefficient '
                           'is not exactly 2. This is a qualitative choice, not a proven nearest expression; '
                           'the full frontier is included in the appendix. The archived miss label is retained.')
            indices = [i for i, row in enumerate(frontier) if row['equation'] == selected]
            assert indices, (method, dataset, seed, selected)
            idx = indices[0]
            if old['matching_equation']:
                assert idx in old['best_frontier_indices']
            eligible = [i for i, row in enumerate(frontier) if isinstance(row.get('r2'), (int, float)) and math.isfinite(row['r2'])]
            best = max(eligible, key=lambda i: (frontier[i]['r2'], -frontier[i]['complexity']))
            records.append(dict(method=method, task=task, dataset=dataset, seed=seed,
                archived_classification=old['classification'], audited_classification=label,
                structural_index=idx, best_r2_index=best, selection_source=selection_source,
                explanation=reason, original_review=old, frontier=frontier,
                source_run=rel, restart_count=len(result['portfolio']['restarts']),
                actual_search_seconds=sum(r['search_runtime_seconds'] for r in result['portfolio']['restarts'])))
    assert len(records) == 180
    assert len({(r['method'], r['task'], r['seed']) for r in records}) == 180
    for method, expected in zip(RUNS, (72, 74)):
        assert sum(r['audited_classification'] in ('exact', 'phenomenological_match') for r in records if r['method'] == method) == expected
    hashes = {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources}
    (OUT / 'audit_data.json').write_text(json.dumps(dict(source_sha256=hashes, records=records), indent=2) + '\n')
    tex = [r'''\documentclass[10pt,letterpaper,landscape]{article}
\usepackage[margin=0.48in]{geometry}
\usepackage{amsmath,amssymb,graphicx,adjustbox,booktabs,listings,xcolor,hyperref,fancyhdr}
\hypersetup{colorlinks=true,linkcolor=blue,pdftitle={One-hour PySR equation audit: 1M-evaluation restarts}}
\pagestyle{fancy}\fancyhf{}\fancyfoot[L]{\footnotesize EmpiricalBench overlap in SRBench2 | 1M-evaluation restarts}
\fancyfoot[R]{\thepage}\renewcommand{\headrulewidth}{0pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
\lstset{basicstyle=\ttfamily\fontsize{7.5}{9}\selectfont,breaklines=true,breakatwhitespace=false,columns=fullflexible,keepspaces=true,aboveskip=3pt,belowskip=5pt}
\begin{document}
\section*{One-hour equation audit: base vs. evolved PySR}
Nine EmpiricalBench-overlap tasks in the \textbf{archived SRBench2 setup}; ten paired seeds (10000--10009).
Each search uses one core, a 3600-second search budget, and restarts capped at $10^6$ evaluations.
Warm-up/scoring are excluded. These are saved final portfolio frontiers; the final restart can overshoot the budget.
This is the comparison behind \textbf{72/90 vs. 74/90}, not the later native EmpiricalBench or 90-second-restart experiments.

\textbf{How to read each page.} Left: base PySR; right: evolved PySR (run 709715).
The \emph{structural candidate} is the equation selected by the archived review/audit, including near matches.
The \emph{best numerical fit} is independently selected by maximum saved training $R^2$ on the final frontier (ties: lower complexity).
It need not be the witness supporting recovery, the search's chosen output, or the minimum-native-loss equation.
Native losses differ between methods and are not directly comparable. No constants are refitted here.

\textbf{Checking correctness.} Exact means the accepted variable dependence, with only the reference's free constants adjustable.
Fixed powers/relative coefficients, tiny extra terms, and transformed targets matter. Near/miss do not count as recovery.
Bode is special: the archived broad phenomenological criterion allows a zero offset; its 20 positives are not strict recovery of the full nonzero-offset law.
Formula displays use eight significant digits for readability; \textbf{the exact saved expression is printed beneath each formula}.
Use those full-precision strings to inspect coefficients near 1 or 2. These pages preserve archived decisions, not a new exhaustive correctness certificate.

\begin{center}\begin{tabular}{lrrl}\toprule
Task & Base / 10 & Evolved / 10 & Jump to first seed \\\midrule''']
    by_key = {(r['task'], r['seed'], r['method']): r for r in records}
    for task, (name, _, _) in TARGETS.items():
        counts = [sum(r['audited_classification'] in ('exact', 'phenomenological_match') for r in records if r['task'] == task and r['method'] == m) for m in RUNS]
        tex.append(f'{name} & {counts[0]} & {counts[1]} & '+r'\hyperref['+task+r'-10000]{'+name+r'} \\')
    tex += [r'\midrule Total (including broad Bode matches) & 72 & 74 & \\\bottomrule\end{tabular}\end{center}',
            r'All 180 frontiers, original reviews, selections, and source fingerprints are saved in \texttt{audit\_data.json}. One page per task--seed pair follows. The unselected Rydberg miss also has a full-frontier appendix.']
    for task, (name, target, variables) in TARGETS.items():
        # Check feature mapping against the actual dataset header, archived with the report.
        path = ROOT / f'pmlb/datasets/first_principles_{task}/first_principles_{task}.tsv.gz'
        with gzip.open(path, 'rt') as handle:
            header = handle.readline().strip()
        for seed in range(10000, 10010):
            tex += [r'\clearpage\phantomsection\label{'+f'{task}-{seed}'+'}',
                    r'\section*{'+f'{name} | seed {seed}'+'}',
                    r'\textbf{Accepted target:} $\displaystyle '+target+r'$\hfill $'+variables+r'$\par',
                    r'{\small Dataset columns: '+esc(header.replace('\t', ', '))+r'. Natural logs unless specified. Constants $c_i$ denote the accepted free parameters.}\par\medskip']
            for j, method in enumerate(RUNS):
                rec = by_key[task, seed, method]
                label = rec['audited_classification'].replace('_', ' ')
                tex += [r'\begin{minipage}[t]{0.485\textwidth}', r'\subsection*{'+esc(method)+'}',
                        r'\textbf{Archived/audited label: '+esc(label)+r'.} '+esc(f"{rec['restart_count']} restarts; actual search {rec['actual_search_seconds']:.1f} s."),
                        r'{\small '+esc(rec['selection_source'])+'}',
                        candidate_block('Recovery witness' if label in ('exact', 'phenomenological match') else 'Closest structural candidate', rec['structural_index'], rec['frontier'][rec['structural_index']]),
                        r'{\small\textbf{Review rationale:} '+esc(rec['explanation'])+'}']
                if rec['structural_index'] == rec['best_r2_index']:
                    tex += [r'\subsubsection*{Best numerical fit}', r'Same equation as above (maximum saved training $R^2$).']
                else:
                    tex.append(candidate_block('Best numerical fit (maximum training R2)', rec['best_r2_index'], rec['frontier'][rec['best_r2_index']]))
                if rec['archived_classification'] != rec['audited_classification']:
                    tex.append(r'{\small Original label: '+esc(rec['archived_classification'])+'; changed by the saved audit.}')
                tex.append(r'\end{minipage}' + (r'\hfill' if j == 0 else ''))
    miss = by_key['rydberg', 10000, 'Base PySR']
    for start in range(0, len(miss['frontier']), 4):
        tex += [r'\clearpage\section*{Appendix: Rydberg, base PySR, seed 10000}',
                r'The archived reviewer returned miss without selecting an equation. All saved frontier rows follow; the main page uses index 7 as a compact structural proxy. Target: $\log\lambda=c_0-\log(x_0^{-2}-x_1^{-2})$.']
        for i in range(start, min(start+4, len(miss['frontier']))):
            tex.append(candidate_block(f'Candidate {i}', i, miss['frontier'][i]))
    tex += [r'\clearpage\section*{Sources and reproduction}',
            r'Rebuild from the repository root: \texttt{python figures/build\_empirical\_1m\_audit.py}. This reads saved artifacts and runs Tectonic locally; it submits no jobs and makes no LLM requests.',
            r'\textbf{Source files (SHA-256 fingerprints in audit\_data.json):}']
    tex += [r'\begin{lstlisting}'+ '\n'+s+'\n'+r'\end{lstlisting}' for s in sources]
    tex += [r'\textbf{Audit limits.} Selected positive witnesses were previously checked algebraically. Archived near/miss decisions were not exhaustively re-reviewed. The closest structural candidate is qualitative, not a globally optimal symbolic distance. All displayed equations are verified to occur verbatim at their stated zero-based final-frontier indices. The numeric-fit selection uses saved training metrics, not held-out performance.', r'\end{document}']
    path = OUT / 'equation_audit.tex'
    path.write_text('\n'.join(tex) + '\n')
    subprocess.run(['tectonic', '--keep-logs', '--outdir', str(OUT), str(path)], check=True)
    print(f'Wrote {OUT / "equation_audit.pdf"}; 180 trials, 90 paired audit pages.')


if __name__ == '__main__':
    main()
