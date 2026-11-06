#!/usr/bin/env python3
"""Build a full-frontier LaTeX comparison from saved seed-10009 portfolios."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_supernova_zr_2026_09_08 import ROOT, RUNS, TASK, predict, score

SEED = 10009
STEM = "supernova_zr_pareto_seed10009"


def number(value, digits=6):
    s = f"{value:.{digits}g}"
    if "e" in s:
        mantissa, power = s.split("e")
        return rf"{mantissa}\times 10^{{{int(power)}}}"
    return s


def math_tree(equation):
    """Render the stored syntax tree; round displayed constants only."""
    def visit(n):
        if isinstance(n, ast.Constant):
            return number(n.value)
        if isinstance(n, ast.Name) and n.id == "x0":
            return "t"
        if isinstance(n, ast.UnaryOp):
            arg = visit(n.operand)
            if isinstance(n.operand, ast.BinOp):
                arg = rf"\left({arg}\right)"
            return ("-" if isinstance(n.op, ast.USub) else "+") + arg
        if isinstance(n, ast.BinOp):
            left, right = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Div):
                return rf"\frac{{{left}}}{{{right}}}"
            if isinstance(n.op, ast.Pow):
                return rf"\left({left}\right)^{{{right}}}"
            if isinstance(n.op, ast.Mult):
                if isinstance(n.left, ast.BinOp) and isinstance(n.left.op, (ast.Add, ast.Sub)):
                    left = rf"\left({left}\right)"
                if isinstance(n.right, ast.UnaryOp) or (isinstance(n.right, ast.BinOp) and isinstance(n.right.op, (ast.Add, ast.Sub))):
                    right = rf"\left({right}\right)"
                return left + r"\cdot " + right
            if isinstance(n.op, ast.Sub):
                if isinstance(n.right, ast.UnaryOp) or (isinstance(n.right, ast.BinOp) and isinstance(n.right.op, (ast.Add, ast.Sub))):
                    right = rf"\left({right}\right)"
                return left + " - " + right
            if isinstance(n.op, ast.Add):
                if isinstance(n.right, ast.UnaryOp):
                    right = rf"\left({right}\right)"
                return left + " + " + right
        if isinstance(n, ast.Call) and len(n.args) == 1:
            arg = visit(n.args[0])
            func = n.func.id
            if func == "exp":
                return rf"\exp\!\left({arg}\right)"
            if func == "log":
                return rf"\ln\!\left({arg}\right)"
            if func == "sqrt":
                return rf"\sqrt{{{arg}}}"
            if func in {"square", "cube"}:
                exponent = 2 if func == "square" else 3
                return rf"\left({arg}\right)^{{{exponent}}}"
        raise ValueError(ast.dump(n))
    return visit(ast.parse(equation, mode="eval").body)


PREAMBLE = r"""\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{amsmath,amssymb,graphicx,xcolor,booktabs,array,fancyhdr,hyperref}
\definecolor{ink}{HTML}{16324F}
\definecolor{muted}{HTML}{52616B}
\definecolor{baseline}{HTML}{246A91}
\definecolor{evolved}{HTML}{A65B20}
\definecolor{highlight}{HTML}{FFF2C6}
\definecolor{highlightedge}{HTML}{B98B17}
\definecolor{card}{HTML}{F4F6F8}
\definecolor{cardedge}{HTML}{D9E0E5}
\hypersetup{colorlinks=true,urlcolor=baseline,linkcolor=baseline}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\color{muted}Supernova ZR / full saved Pareto frontiers}
\fancyhead[R]{\small\color{muted}Seed 10009}
\fancyfoot[L]{\small\color{muted}September 8 runs / prepared September 9, 2026}
\fancyfoot[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.2pt}
\newsavebox{\equationbox}
\newcommand{\fitformula}[1]{%
  \sbox{\equationbox}{$\displaystyle #1$}%
  \ifdim\wd\equationbox>0.97\linewidth
    \resizebox{0.97\linewidth}{!}{\usebox{\equationbox}}%
  \else\usebox{\equationbox}\fi}
\newcommand{\entry}[8]{%
  \begingroup
  \setlength{\fboxsep}{8pt}%
  \noindent\fcolorbox{#7edge}{#7}{%
    \begin{minipage}{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}
    {\small\bfseries #1}\hfill{\small Complexity #2}\hfill
    {\small Loss $=#3$}\hfill{\small $R^2=#4$}\hfill
    {\small $R^2_{\mathrm{cal}}=#5$}\\[8pt]
    \centering\fitformula{f(t)=#6}\par
    #8
    \end{minipage}}%
  \endgroup\par\vspace{8pt}}
\begin{document}
\thispagestyle{empty}
{\LARGE\bfseries\color{ink}Supernova ZR: baseline vs. evolved}\par
{\large Full portfolio Pareto frontiers, seed 10009}\par
One core; 60-minute total search budget; restarts capped at $10^6$ evaluations;
no max-size warmup. Baseline uses L1 loss. Evolved uses the validation-selected
bundle from run 709715 and its custom affine-profile loss.

\section*{Reference family and the highlighted candidates}
The task fits normalized observed flux as a function of time $t$ (days relative
to the observed peak). The 236 rows span $-18.04$ to $86.75$ days. The accepted
reference is the empirical Bazin family, not a known physical generating law:
\[
  F(t;A,B,C,D)=\frac{A}{B\exp(Ct)+\exp(-Dt)}.
\]
The yellow boxes identify the saved review's \emph{closest structural match},
not the lowest-loss or highest-$R^2$ member. B06 is the simplest exact-family
match cited by the baseline review. E13 is the evolved review's near match.
All 15 baseline and all 20 evolved frontier rows follow, in increasing complexity.

\textbf{Baseline B06 (exact):}
\[
 \frac{\exp(a t)}{\exp(t)+b}
 =\frac{1/b}{(1/b)\exp((1-a)t)+\exp(-a t)},\qquad
 a\simeq0.955950,\quad b\simeq2.25986\times10^{-5}.
\]
\textbf{Evolved E13 (near):}
\[
 \frac{(0.0571477t)^3+0.976272}
 {\exp(-0.127003t)+1.27564\exp(0.119704t)}.
\]
Its denominator has the reference structure, but its numerator contains a real
nonconstant cubic term. Rounding here is for display only; the raw syntax-tree
version appears in the frontier.

\section*{Metrics}
Let $p_i=f(t_i)$, $S_y=\sum_i(y_i-\bar y)^2$, and let
$(\hat a,\hat b)$ minimize $\sum_i(y_i-a p_i-b)^2$. Then
\[
 R^2=1-\frac{\sum_i(y_i-p_i)^2}{S_y},\qquad
 R^2_{\mathrm{cal}}=1-\frac{\sum_i(y_i-\hat a p_i-\hat b)^2}{S_y}.
\]
The native losses are
\[
 L_{\mathrm{L1}}=\frac1n\sum_i|y_i-p_i|,\qquad
 L_{\mathrm{custom}}=\sqrt{\operatorname{clip}(1-R^2_{\mathrm{cal}},0,1)}
 +\frac{1}{256}\frac{\sqrt{1-R^2}}{1+\sqrt{1-R^2}}.
\]
The custom expression matches the implementation on this dataset (its tiny
target-normalization floor is inactive). Calibration fits scale and offset only,
not the equation's internal constants. For constant predictions, calibration
returns the target mean and $R^2_{\mathrm{cal}}=0$.

\textbf{Reading the tables.} Loss is each method's own search loss; the two loss
columns have different meanings and should not be compared numerically. Every
$R^2$ is recomputed on the same 236 training rows, without clipping negative
values. Calibrated $R^2$ is a diagnostic, not a held-out score or the stored raw
equation's prediction quality. Constants are displayed to six significant digits;
all metrics use the full-precision stored expressions. Row IDs are zero-based
positions in the merged frontier, not the original per-restart PySR indices.

{\footnotesize Reference context: \href{https://arxiv.org/html/2402.04298v4\#S5.SS3}
{Russeil et al., Multi-View Symbolic Regression, section 5.3 and Table 4}.}
"""


def main():
    frame = pd.read_csv(ROOT/f"pmlb/datasets/{TASK}/{TASK}.tsv.gz", sep="\t")
    x, y = frame.iloc[:, 0].to_numpy(), frame.target.to_numpy()
    document = [PREAMBLE]
    audit = {"seed": SEED, "dataset": TASK, "n_rows": len(x), "methods": []}
    for label, prefix, kind in [("Baseline portfolio", "B", "exact"), ("Evolved portfolio", "E", "near")]:
        root = ROOT/RUNS[label]
        source = root/"srbench_full_results.json"
        result = json.loads(source.read_text())["results"][f"{TASK}|{SEED}|0"]
        reviews = json.loads((root/"manual_solve_check_results.json").read_text())["reviews"]
        review = next(r for r in reviews if r["dataset"] == TASK and r["seed"] == SEED)
        assert review["classification"] == kind
        rows = []
        baseline = prefix == "B"
        for index, f in enumerate(result["pareto_frontier"]):
            p = predict(f["equation"], x)
            assert np.isfinite(p).all()
            scores = score(p, y)
            native = float(np.mean(abs(p-y))) if baseline else scores["evolved_loss"]
            assert abs(native-f["loss"]) < 1e-10, (label, index, native, f["loss"])
            highlighted = index in review["best_frontier_indices"]
            if highlighted:
                assert f["equation"] == review["matching_equation"]
            rows.append({"id": f"{prefix}{index:02d}", "frontier_index": index,
                         "complexity": f["complexity"], "equation": f["equation"],
                         "loss_kind": "L1" if baseline else "custom", "saved_loss": f["loss"],
                         "recomputed_native_loss": native, **scores,
                         "closest_match": highlighted, "match_classification": kind if highlighted else None})
        assert len(rows) == (15 if baseline else 20)
        restarts = result["portfolio"]["restart_count_successful"]
        pages = (len(rows)+4)//5
        for page in range(pages):
            chunk = rows[5*page:5*(page+1)]
            document += [r"\clearpage", rf"{{\Large\bfseries\color{{{'baseline' if baseline else 'evolved'}}}{label}}}\par",
                rf"{{\large Native loss: {'mean absolute error (L1)' if baseline else 'custom affine-profile loss'}}}\par",
                rf"Seed {SEED}; {restarts} successful restarts; full merged frontier. "
                rf"Rows {chunk[0]['id']}--{chunk[-1]['id']} ({page+1}/{pages}).\par\vspace{{6pt}}"]
            for row in chunk:
                note = (rf"\par\vspace{{6pt}}{{\small\bfseries Closest structural match: {kind.upper()}. "
                        + ("Algebraically in the Bazin family." if baseline else "Extra cubic numerator; not in the Bazin family.")
                        + "}") if row["closest_match"] else ""
                args = [row["id"], str(row["complexity"]), number(row["saved_loss"]),
                        number(row["raw_r2"]), number(row["affine_r2"]), math_tree(row["equation"]),
                        "highlight" if row["closest_match"] else "card", note]
                document.append(r"\entry"+"".join("{"+a+"}" for a in args))
            if page == pages-1:
                document += [r"\vfill{\footnotesize\color{muted}Source run: ",
                    r"\nolinkurl{"+RUNS[label]+"}",
                    r"\par Full-precision equations and metrics are retained in the companion JSON.}"]
        audit["methods"].append({"method": label, "source": str(source.relative_to(ROOT)),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "successful_restarts": restarts,
            "rows": rows})
    document.append(r"\end{document}")
    (ROOT/f"analysis/{STEM}.tex").write_text("\n".join(document)+"\n")
    (ROOT/f"analysis/{STEM}.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(f"Wrote {STEM}.tex and .json; verified losses for all 35 equations.")


if __name__ == "__main__":
    main()
