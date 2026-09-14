# Recorded frontier scores

Final generation 30 population, filtered for exact LOC/score nondominance. Scores are percentages of ground-truth matches; errors remain failures. Noise levels are relative target noise. See comparison.md for interpretation.

| LOC / Julia bundle | Seeds | All noise | 0 | 0.001 | 0.01 | 0.1 | Errors at noise 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [234](frontier_loc234.jl) | 2 | 65.000% | 57.50% | 82.50% | 67.50% | 52.50% | 10/40 |
| [200](frontier_loc200.jl) | 1 | 62.500% | 60.00% | 70.00% | 70.00% | 50.00% | 5/20 |
| [189](frontier_loc189.jl) | 1 | 58.750% | 70.00% | 55.00% | 60.00% | 50.00% | 3/20 |
| [156](frontier_loc156.jl) | 2 | 54.375% | 52.50% | 62.50% | 57.50% | 45.00% | 9/40 |
| [153](frontier_loc153.jl) | 2 | 48.750% | 52.50% | 55.00% | 47.50% | 40.00% | 8/40 |
| [150](frontier_loc150.jl) | 2 | 47.500% | 57.50% | 57.50% | 47.50% | 27.50% | 8/40 |
| [141](frontier_loc141.jl) | 2 | 44.375% | 40.00% | 50.00% | 40.00% | 47.50% | 5/40 |
| [108](frontier_loc108.jl) | 2 | 8.125% | 7.50% | 15.00% | 2.50% | 7.50% | 3/40 |
| [102](frontier_loc102.jl) | 1 | 0.000% | 0.00% | 0.00% | 0.00% | 0.00% | 6/20 |

## Ten-seed fresh identification

Only these final-frontier bundles were included in the fresh identification pass. The original frontier continues to use final-population scores.

| LOC | Fresh all-noise | Noise 0 | Noise 0.001 | Noise 0.01 | Noise 0.1 | Errors at noise 0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 234 | 57.000% | 57.50% | 61.50% | 63.00% | 46.00% | 45/200 |
| 200 | 53.375% | 66.50% | 51.50% | 54.00% | 41.50% | 18/200 |

## Matched zero-noise tasks without recorded errors

Each row uses its own subset of the 20 training tasks: retain a task only if all recorded evaluations for both bundles are free of recorded errors. The reference is the separate 709715 90-second, ten-seed evaluation. These subsets differ by row and are not a new ranking.

| LOC | Shared tasks | 709715 (261 LOC) | 941340 candidate |
| ---: | ---: | ---: | ---: |
| 234 | 10 | 75.00% | 80.00% |
| 200 | 15 | 80.00% | 80.00% |
| 189 | 17 | 81.18% | 82.35% |
| 156 | 12 | 70.83% | 62.50% |
| 153 | 13 | 76.15% | 61.54% |
| 150 | 12 | 75.00% | 75.00% |
| 141 | 16 | 71.88% | 40.62% |
| 108 | 17 | 71.76% | 5.88% |
| 102 | 14 | 72.14% | 0.00% |

## Same-run reference, shared error-free task/noise cells

The original 709715 validation winner was also evaluated inside 941340. Each row below retains only task/noise cells for which both candidates have no recorded errors across their recorded seeds. Each cell is equally weighted. These are sensitivity checks, not corrected population scores.

| LOC | Shared cells (of 80) | Original 261 LOC | Candidate |
| ---: | ---: | ---: | ---: |
| 234 | 38 | 78.95% | 81.58% |
| 200 | 38 | 78.95% | 89.47% |
| 189 | 45 | 70.00% | 71.11% |
| 156 | 35 | 71.43% | 67.14% |
| 153 | 32 | 65.62% | 57.81% |
| 150 | 34 | 64.71% | 54.41% |
| 141 | 40 | 68.75% | 46.25% |
| 108 | 44 | 70.45% | 11.36% |
| 102 | 45 | 70.00% | 0.00% |
