# PySR on fully observable NeuronBench

PDF: `/home/sca63/meta_sr/reports/neuronbench_pysr_fully_observable.pdf`

Protocol: 6 worlds × 2 methods × 3 seeds, max_evals=1,000,000.

| method | runs | recovered | exact symbolic | median raw NRMSE | median affine-calibrated NRMSE |
|---|---:|---:|---:|---:|---:|
| vanilla PySR | 18 | 1 | 0 | 4.089e-03 | 4.018e-03 |
| evolved PySR (538190) | 18 | 0 | 0 | 4.627e-03 | 3.514e-03 |

Paired: evolved wins 9/18; baseline wins 9/18; median evolved/baseline NRMSE ratio 0.967.

Conclusion: evolved 538190 does not clearly improve full dynamics recovery. Vanilla has the only strict recovery and the lower median raw NRMSE; the paired comparison is 9–9. Evolved 538190 is modestly better on affine-calibrated shape NRMSE, consistent with its affine-invariant custom loss.

## Best raw NRMSE by world

| world | vanilla (3 seeds) | evolved 538190 (3 seeds) |
|---|---:|---:|
| z_rebound | 4.379e-03, 3.385e-03, 3.457e-03 | 3.662e-03, 1.212e-02, 3.125e-03 |
| h_sag | 3.469e-03, 3.550e-03, 1.001e-02 | 6.499e-03, 1.681e-03, 3.737e-03 |
| na_fatigue | 9.342e-04, 3.436e-07, 3.049e-05 | 1.004e-04, 1.731e-03, 1.741e-03 |
| ca_rebound | 1.002e-02, 4.282e-03, 9.791e-03 | 8.577e-03, 3.917e-03, 9.989e-03 |
| d_type | 4.173e-03, 4.005e-03, 2.169e-02 | 2.569e-02, 5.336e-03, 7.862e-04 |
| textbook_M | 5.353e-03, 3.849e-03, 9.232e-03 | 6.758e-03, 6.548e-03, 7.375e-03 |
