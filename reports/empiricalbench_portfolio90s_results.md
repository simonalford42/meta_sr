# EmpiricalBench one-hour portfolio results

Both search runs completed all 90 trials without recorded errors. Terra's completed final-frontier batch was recovered from saved responses; no duplicate review requests were made.

## Recovery at one hour

Counts are problem–seed trials, ten seeds per problem. Count exact accepted families, including equivalent phenomenological labels for Bode, Leavitt and Schechter; exclude near matches.

| Problem | Base PySR | Evolved 709715 |
|---|---:|---:|
| Bode | 10/10 | 10/10 |
| Hubble | 10/10 | 10/10 |
| Ideal Gas | 10/10 | 10/10 |
| Kepler | 10/10 | 10/10 |
| Leavitt | 10/10 | 10/10 |
| Newton | 10/10 | 10/10 |
| Planck | 0/10 | 0/10 |
| Rydberg | 1/10 | 1/10 |
| Schechter | 10/10 | 10/10 |
| **Total** | **71/90 (78.9%)** | **71/90 (78.9%)** |

## Review corrections

Terra returned 28 `phenomenological_match` labels for Bode, Leavitt and Schechter despite recognizing their accepted fitted families. Inspection of all 28 selected equations confirmed their stated family form. Their normalized classification is `exact`, with `original_classification` retained. Near matches remain excluded.

One Schechter review correctly selected frontier index 6 but omitted its log(x0) factor when copying the expression. Its explanation explicitly described that factor. The saved override restores the exact expression at index 6, which has the accepted c0+c1*log(x0)+c2*x0 form. Original response and original copied equation are preserved.

The completed final-frontier batch cost $0.707940 at the archived batch rates. Binary-search timing reviews are resumed in `runs/empiricalbench_9-16_portfolio90s_terra_recovery/`.
