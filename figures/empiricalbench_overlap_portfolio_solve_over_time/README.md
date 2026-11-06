# EmpiricalBench overlap in SRBench2: portfolio recovery

Plots use the completed SRBench2 recovery records; no new reviews were made. This is a subset of SRBench2 runs, not separate EmpiricalBench evaluations.

8 shared reference-family tasks; Bode excluded (phenomenological in SRBench2).

Included tasks: hubble, ideal_gas, kepler, leavitt, newton, planck, rydberg, schechter.

Mean tasks solved is the number of recovered task–seed trials divided by ten seeds. All tasks/seeds remain in the denominator, including final negatives. The binary-search approximation can miss temporary recoveries.

Reproduce: `python scripts/plot_srbench2_portfolio_recovery.py --empirical-only`
