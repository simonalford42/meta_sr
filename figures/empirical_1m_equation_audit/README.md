# One-hour equation audit

[Open the PDF](equation_audit.pdf). The editable source is [equation_audit.tex](equation_audit.tex).

This is the original **SRBench2 experiment restricted to the nine EmpiricalBench-overlap tasks**, with one core, a 3600-second search budget excluding warm-up, and restarts capped at 1,000,000 evaluations. It is the comparison with 72/90 base and 74/90 evolved recoveries, including the archived broad Bode criterion. The final saved restart can overshoot 3600 seconds; actual cumulative search time is printed per trial.

Each paired seed gets a landscape page with the target, feature mapping, both methods' structural candidates, full-precision saved expressions, review labels/reasons, and the highest-training-R² equation on each final frontier. The latter is explicitly a numerical-fit selection, not necessarily the search-selected output or the structural recovery witness. No constants are refitted; native losses differ between methods.

| Pages | Contents |
|---|---|
| 1 | Protocol, selection rules, counts, clickable task links |
| 2–11 | Hubble, seeds 10000–10009 |
| 12–21 | Ideal gas |
| 22–31 | Kepler |
| 32–41 | Leavitt |
| 42–51 | Newton |
| 52–61 | Planck |
| 62–71 | Rydberg |
| 72–81 | Schechter |
| 82–91 | Bode |
| 92–96 | Complete frontier for the base Rydberg seed-10000 miss |
| 97 | Sources and limitations |

For near matches, the report uses the archived review's structural selection. For the sole miss without a selected equation (base Rydberg seed 10000), it shows index 7 as a compact log(n1)-plus-ratio structural proxy, explicitly identified as a report choice, and includes every frontier candidate in the appendix. This is not a mathematically defined closest-expression search. Archived labels are preserved, including the saved positive-audit correction to evolved Rydberg seed 10004.

Rounded typeset formulas are for readability only. The full-precision saved strings beneath them are authoritative for coefficient checks. Bode's archived criterion allows zero offset and should not be interpreted as strict recovery of the full nonzero-offset law. The report does not newly certify archived near/miss decisions or guarantee that an alternative exact witness was never missed.

`audit_data.json` contains all 180 final frontiers, original reviews, selected indices, audit explanations, and SHA-256 fingerprints of source files. The build verifies that witnesses occur verbatim at their archived indices, confirms the run protocol and seed coverage, reconciles labels with the saved recovery analysis, and reproduces the 72/90 and 74/90 totals.

Rebuild locally from the repository root:

```bash
python figures/build_empirical_1m_audit.py
```

Requires the archived run files, SymPy, and Tectonic. No LLM requests, new searches, or SLURM submissions.
