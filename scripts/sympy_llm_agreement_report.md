# SymPy vs GPT-5.2 Equivalence Check: Agreement Report

## Scope
This note documents how we compared symbolic equivalence decisions from:
- `SymPy`-based checker (`check_pysr_symbolic_match`)
- `LLM` checker (`check_sympy_equivalence_with_llm`, model `openai/gpt-5.2`, thinking `high`)

on PySR best equations for the `1e8` zero-noise results.

## What We Ran
Main run over all tasks:
```bash
python scripts/check_topk_r2_with_llm.py \
  -k 10000 \
  --save-jsonl results/symbolic_checks_pysr_1e8/all_best_sympy_llm_results.jsonl
```

Human-readable disagreement extraction:
```bash
# built from all_best_sympy_llm_results.jsonl
results/symbolic_checks_pysr_1e8/all_sympy_llm_disagreements_report.txt
```

## Main Result Summary
From the full run (`122` tasks):
- Combined matches (`sympy OR llm`): `80/122`
- SymPy matches: `79/122`
- LLM matches: `69/122`

The raw disagreement report initially had `12` disagreements.

## Why The Initial Disagreements Happened
The disagreement report showed two main patterns:
1. Expression-vs-equation formatting mismatch:
   - e.g. LLM compared `x1*y1 + ...` against `A = x1*y1 + ...` literally.
   - SymPy path already compares expression forms and treated these as matches.
2. Tiny coefficient/numeric precision effects:
   - Cases where coefficients differ at very small scale (roughly `1e-8` to `1e-10` level).
   - These can cause strict symbolic decisions to differ depending on parsing/rounding path.

## Fix Applied
We updated LLM-side comparison to strip assignment LHS (`... =`) and compare RHS expression to RHS expression before asking the model.

After this change, disagreement rate dropped substantially on targeted reruns of prior `llm=False` cases; most equation-format disagreements disappeared.

## Practical Takeaway
For this workflow, GPT-5.2 thinking is effectively consistent with SymPy in almost all meaningful cases once normalization is aligned (RHS vs RHS).

Remaining disagreements are mostly edge cases driven by very small numeric coefficient differences ("1e-8 type" issues), not broad conceptual mismatch.

## Files to Reference
- Full run JSONL:
  - `results/symbolic_checks_pysr_1e8/all_best_sympy_llm_results.jsonl`
- Disagreement report:
  - `results/symbolic_checks_pysr_1e8/all_sympy_llm_disagreements_report.txt`
- Targeted rerun report (RHS-only normalization path):
  - `results/symbolic_checks_pysr_1e8/topk_llm_false_rerun_report_rhs_only.txt`
