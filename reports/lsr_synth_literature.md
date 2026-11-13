# LSR-Synth (LLM-SRBench) literature numbers: what's published and how comparable it is

Compiled 2026-11-10. Sources: arXiv HTML and PDF pages, the official GitHub repos, the HF dataset API, plus one secondary summary for RESTART, which is clearly marked. The numbers below were copied from the papers' HTML tables, which I parsed with a script, or from the PDF text. **No numbers were invented or recomputed.** Bold and underline marks in the source tables were lost in extraction and are not reproduced.

LSR-Synth domains, in the HF config naming: `chem_react` (36), `bio_pop_growth` (24), `phys_osc` (44), `matsci` (25), for 129 problems in total.

## 0. TL;DR

| Paper | arXiv | Venue | Reports LSR-Synth? | Metrics on LSR-Synth | Own method rows | Baseline rows |
|---|---|---|---|---|---|---|
| LLM-SRBench (Shojaee et al.) | [2504.10415](https://arxiv.org/abs/2504.10415) | ICML 2025 (Oral) | yes | SA, Acc_0.1, NMSE (ID); OOD NMSE in a figure only | n/a (benchmark paper): LLM-SR, LaSR, SGA, Direct × 3 LLMs, **PySR (App. F, PDF only)** | n/a |
| SR-Scientist (Xia, Sun, Liu) | [2510.11661](https://arxiv.org/abs/2510.11661) | ICLR 2026 | yes | Acc_0.01, Acc_0.001 (5% worst points dropped); SA; R²; NMSE (physics only); OOD Acc_0.01 in figures only | SR-Scientist × 5 LLMs, + RL | **re-run** by the authors: GPLearn, E2E, NeSymReS, DSR, uDSR, **PySR**, LaSR, LLM-SR |
| PiT-PO (Wang et al.) | [2602.10576](https://arxiv.org/abs/2602.10576) | KDD 2026 | yes, Llama-3.1-8B only | SA, Acc_all-0.1, NMSE | PiT-PO | **copied verbatim** from the LLM-SRBench Llama-3.1-8B rows |
| RESTART (Li & Pan) | none (OpenReview [z9TKJhLVKj](https://openreview.net/forum?id=z9TKJhLVKj)) | ICLR 2026 poster | yes, but **no Chemistry** (93 synth tasks) | NMSE, ACC (τ unknown), ID and OOD | RESTART (Qwen3-8B) | LLM-SR (Qwen3-8B) |
| GenSR | [2602.20557](https://arxiv.org/abs/2602.20557) | ICLR 2026 | **no** | SRBench Feynman, Strogatz, black-box only | n/a | n/a |
| MOD-SR (Xiang, Wei, Yan) | none (OpenReview [p06oR4sXgz](https://openreview.net/forum?id=p06oR4sXgz)) | ICML 2026 | **not found** | the poster shows only R² on Nguyen/Keijzer/Koza/Constant/Livermore/Vladislavleva/R/Jin + SRBench Feynman/Strogatz | n/a | n/a |
| LLM-SR (Shojaee et al.) | [2404.18400](https://arxiv.org/abs/2404.18400) | ICLR 2025 | no (it predates LSR-Synth) | LLM-SR's LSR-Synth numbers come from LLM-SRBench and SR-Scientist | | |

**PySR rows on LSR-Synth exist in two places:**
- LLM-SRBench, App. F Table 3: SA and Acc_0.1. This table appears only in the PDF, not the HTML.
- SR-Scientist, Table 1: Acc_0.01 and Acc_0.001. SA appears in Table 2.

PiT-PO's PySR row is on the 4-problem LLM-SR Suite only.

**Ungated data:** [`pkuHaowei/llm-srbench`](https://huggingface.co/datasets/pkuHaowei/llm-srbench) is an ungated third-party re-pack of LLM-SRBench as per-subset parquet files.
- It contains train, test (ID) and OOD splits plus the ground-truth expression for all 129 LSR-Synth problems.
- A range request returned HTTP 206 with no auth. The same request against the original `nnheui/llm-srbench` hdf5 returns 401.
- Details are in section 7.

---

## 1. LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models

- arXiv [2504.10415](https://arxiv.org/abs/2504.10415) (v2, 7 Jun 2025). HTML: <https://arxiv.org/html/2504.10415v2>. ICML 2025 (Oral, per the HF mirror README). The HTML version omits the appendices; I read those from the PDF.
- Code: <https://github.com/deep-symbolic-mathematics/llm-srbench>. A local copy is at `baselines/llm-srbench-src/`.

### Metrics

As defined in the paper (Sec. 2.3 and App. B.1):

- **Acc_τ = 1( max_{1≤i≤N_test} |(ŷ_i − y_i)/y_i| ≤ τ )**. It is the strict max over all test points. **No point dropping** is mentioned. τ = 0.1 is used in all tables.
- **NMSE = Σ(ŷ−y)² / Σ(y−ȳ)²** on the test set.
- **SA (symbolic accuracy):** GPT-4o judges equivalence after removing parameters and constants. GPT-4o agreed with two authors 94.6% of the time on 130 sampled problems.
- **OOD:** a dedicated OOD test set exists for LSR-Synth only. ID vs OOD NMSE appears only in **Fig. 5** (GPT-4o-mini), with no numeric table.

**Code note.** The official repo (`bench/pipelines.py::compute_output_base_metrics`) computes only `mse`, `nmse`, `r2`, `kdt` (Kendall τ) and `mape`. It **does not compute Acc_τ at all**. It also drops NaN predictions before computing NMSE, using `np.mean(...)/np.var(y)`, which is equivalent to the formula above. So Acc_τ must have been computed by unreleased scripts.

### Data

LSR-Synth data generation (App. A):
- 5000 evenly spaced samples per problem.
- Dynamical domains: `solve_ivp` RK45 over t∈[0,60]. MatSci: ε∈[0,0.6], T∈[273,573] K.
- OOD = the last 500 points (by time, or by temperature for matsci). The remaining 4500 points are ID train and validation, from which the HDF5 holds the `train_data` and `id_test_data` splits.

### Protocol

- **Backbones:** Llama-3.1-8B-Instruct, GPT-3.5-turbo, GPT-4o-mini.
- **Budget:** 1k LLM calls per problem for every method.
- **LLM-SR settings:** temperature 0.8, 4 programs per prompt, 4 evaluators, 10 islands, k=2 in-context examples, ≤10 params, BFGS, 30 s and 2 GB per program.
- **LaSR settings:** 25 iterations × 550 cycles, 10 populations × 33, maxsize 30, operators + * − / ^ exp log sqrt sin cos tan cosh, LLM weights 0.005 each, top-20 concepts.
- **Seeds and runs:** not stated. Many Acc values are not k/N, which suggests averaging over runs or some other unstated aggregation. Examples: Chem 50.22 and 38.92 (k/36 gives 50.00 and 38.89); MatSci 88.12, 88.28, 64.22 (k/25 gives multiples of 4).
- **Final equation selection:** not stated in the paper. In the code, the LLM-SR searcher returns the single best-scoring program, where score = −train MSE after BFGS.
- **Problem counts:** the paper text says LSR-Synth has 128 problems (36/24/**43**/25). The released data has 129 (phys_osc = 44), and SR-Scientist also uses 129 with physics = 44.

### Table 1: LSR-Synth columns (all rows are baselines run by the benchmark authors)

| Method | LLM | Chem SA | Chem Acc0.1 | Chem NMSE | Bio SA | Bio Acc0.1 | Bio NMSE | Phys SA | Phys Acc0.1 | Phys NMSE | MatSci SA | MatSci Acc0.1 | MatSci NMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Direct (DataBlind) | Llama-3.1-8B | 0.0 | 0.0 | 0.0644 | 0.0 | 0.0 | 0.5481 | 0.0 | 0.0 | 0.0459 | 0.0 | 0.0 | 0.0826 |
| Direct (DataBlind) | GPT-3.5-turbo | 0.0 | 8.33 | 0.0023 | 0.0 | 4.16 | 0.5990 | 0.0 | 2.27 | 0.0274 | 0.0 | 0.0 | 0.0277 |
| Direct (DataBlind) | GPT-4o-mini | 0.0 | 13.88 | 0.0221 | 0.0 | 4.16 | 0.4648 | 4.54 | 9.09 | 0.0647 | 0.0 | 0.0 | 0.0484 |
| SGA | Llama-3.1-8B | 0.0 | 8.33 | 0.0458 | 0.0 | 0.0 | 0.2416 | 0.0 | 2.27 | 0.1549 | 0.0 | 12.12 | 0.0435 |
| SGA | GPT-3.5-turbo | 0.0 | 8.33 | 0.0071 | 0.0 | 8.33 | 0.1279 | 2.27 | 4.54 | 0.0249 | 0.0 | 28.10 | 0.0019 |
| SGA | GPT-4o-mini | 0.0 | 16.66 | 5.46e-4 | 4.16 | 12.51 | 0.0128 | 4.54 | 9.09 | 0.0511 | 0.0 | 36.11 | 6.02e-4 |
| LaSR | Llama-3.1-8B | 0.0 | 27.77 | 2.77e-4 | 4.16 | 16.66 | 2.73e-4 | 4.54 | 25.02 | 0.0018 | 8.21 | 64.22 | 7.44e-5 |
| LaSR | GPT-3.5-turbo | 0.0 | 38.89 | 1.51e-4 | 0.0 | 16.66 | 2.31e-4 | 6.81 | 22.71 | 0.0011 | 20.66 | 64.09 | 3.77e-5 |
| LaSR | GPT-4o-mini | 2.77 | 38.92 | 9.11e-5 | 8.33 | 20.83 | 1.53e-4 | 9.91 | 31.81 | 9.94e-4 | 28.12 | 72.04 | 9.23e-6 |
| LLM-SR | Llama-3.1-8B | 8.33 | 66.66 | 8.01e-6 | 25.30 | 58.33 | 1.04e-6 | 6.97 | 34.09 | 1.23e-4 | 4.10 | 88.12 | 1.15e-7 |
| LLM-SR | GPT-3.5-turbo | 0.0 | 50.22 | 2.87e-5 | 0.0 | 25.03 | 2.33e-5 | 0.0 | 25.12 | 8.84e-4 | 12.42 | 82.14 | 2.75e-8 |
| LLM-SR | GPT-4o-mini | 11.11 | 52.77 | 4.12e-6 | 16.66 | 29.16 | 3.06e-6 | 9.91 | 36.36 | 7.62e-5 | 20.24 | 88.28 | 3.21e-9 |

For reference, the LSR-Transform columns (SA / Acc0.1 / NMSE):

| Method | Llama-3.1-8B | GPT-3.5-turbo | GPT-4o-mini |
|---|---|---|---|
| Direct | 3.61 / 1.801 / 0.3697 | 2.10 / 1.801 / 0.3553 | 7.21 / 6.306 / 0.2631 |
| SGA | 2.70 / 0.909 / 0.3519 | 0.0 / 0.909 / 0.3465 | 9.91 / 8.11 / 0.2321 |
| LaSR | 5.41 / 45.94 / 0.0021 | 12.61 / 47.74 / 0.0015 | 6.31 / 50.45 / 0.0011 |
| LLM-SR | 30.63 / 38.55 / 0.0101 | 10.81 / 10.81 / 0.1449 | 31.53 / 39.64 / 0.0091 |

Oddity: Phys SA = 9.91 for both LaSR and LLM-SR with GPT-4o-mini equals 11/111, a Transform-sized fraction. It is not k/44 (4/44 = 9.09, 5/44 = 11.36), so it may be a transcription error in the paper.

### App. F Table 3 (PDF only): PySR vs best LLM methods, SA / Acc0.1

From the paper: "PySR's default configuration with the same computational budget (equivalent number of evaluations) as the LLM-based methods". "(best)" appears to be the best value per metric across the three backbones, so SA and Acc can come from different backbones. For example, the LLM-SR Chem cell takes SA 11.11 from GPT-4o-mini and Acc 66.66 from Llama.

| Dataset | LLM-SR (best) | LaSR (best) | SGA (best) | **PySR** |
|---|---|---|---|---|
| LSR-Transform | 31.53 / 39.64 | 12.61 / 50.45 | 9.91 / 8.11 | 8.11 / 56.76 |
| LSR-Synth Chemistry | 11.11 / 66.66 | 2.77 / 38.92 | 0 / 16.66 | **0 / 41.67** |
| LSR-Synth Biology | 25.30 / 58.33 | 8.33 / 20.83 | 4.16 / 12.51 | **0 / 25.0** |
| LSR-Synth Physics | 9.91 / 36.36 | 9.91 / 31.81 | 4.54 / 9.09 | **4.54 / 29.55** |
| LSR-Synth Material Science | 20.24 / 88.28 | 28.12 / 72.04 | 0 / 36.11 | **0 / 68.0** |

The PySR values are clean k/N fractions: 15/36, 6/24, 13/44, 17/25. That suggests a single run.

---

## 2. SR-Scientist: Scientific Equation Discovery With Agentic AI

- arXiv [2510.11661](https://arxiv.org/abs/2510.11661). HTML: <https://arxiv.org/html/2510.11661v1>. ICLR 2026.
- Code: <https://github.com/GAIR-NLP/SR-Scientist>. The repo has no LSR-Synth data; it downloads `nnheui/llm-srbench` and converts it to parquet in `utils/data_preprocessing.py`. The HF dataset `GAIR/SR-Scientist` is their RL training data, not LSR-Synth.

### Metrics

**Acc_τ**, as written in the paper (Eq. 4):
- Acc_τ = 1(max_i |(ŷ_i − y_i)/y_i| ≤ τ), at τ = 0.01 and τ = 0.001.
- "**We discard the 5% worst predictions** to prevent outliers due to the presence of the max operator, following Biggio et al. 2021 and Kamienny et al. 2022."

**Acc_τ as implemented** (`inference/infer/get_results.py`):
- The code counts the fraction of test points with relative error **< τ** (strict inequality).
- Points with y≈0 (`np.isclose(y,0)`) count as correct only if `np.isclose(ŷ,0,atol=1e-9)`.
- Acc = 1 if that fraction is **> 0.95**. On the test set the comparison is strict `>`; in the train-set variant it is `>=`.
- Constants are refit by BFGS (10 params, init 1.0) on the **train** set, then the equation is evaluated on test.

**Other metrics:**
- **SA:** GPT-OSS-120B judges each problem 10 times. Non-unanimous cases go to humans. They report 98.3% LLM-human agreement on 121 unanimous cases.
- **R²:** reported in App. Table 7.
- **NMSE:** reported in App. Table 6, physics only.
- **OOD:** Acc_0.01 on the OOD split, shown only in bar charts (Fig. 2, 9–12), with no numbers.
- **Noise robustness:** σ ∈ {0.01, 0.05, 0.1}, in Fig. 3 only.

### Protocol

- **Backbones:** Qwen3-Coder-480B-A35B, GLM-4.5-Air, GPT-OSS-120B, GPT-OSS-20B and Qwen3-Coder-30B-A3B, plus RL fine-tuning.
- **SR-Scientist settings:** 40 iterations × ≤25 turns, about 1k LLM calls. MAPE goal starts at 0.1%; termination at 0.0001% MAPE. Temperature 0.7, top-K = 3 equations from the experience buffer, ≤10 params, BFGS on MSE.
- **LLM baselines:** ≤1,000 LLM calls and ≤1,000 candidates per problem.
- **Non-LLM baselines:** ≤100,000 candidates per problem.
- **Seeds:** "repeat it three times and report the average". This is why the values are non-integer fractions of N.
- **Final equation selection:** "select the equation with the best performance on the observed data". The footnote says lowest NMSE; the code takes the lowest **train** MSE. That is train-only selection, with no validation split.
- **Train data:** the LLM-SRBench `train_data` split. ID test is `id_test_data`; OOD test is `ood_test_data`.

**PySR config** (App. A):
- niterations=125, ncycles_per_iteration=550, populations=15 × population_size=33, maxsize=30, weight_randomize=0.1.
- Binary operators + − * / pow. Unary operators exp, log, sqrt, sin, cos.
- nested_constraints: sin/cos may not nest sin/cos; exp/log may not nest exp/log; sqrt may not nest sqrt.
- constraints: sin 10, cos 10, exp 20, log 20, sqrt 20, pow (−1, 20).

**LaSR config:** 25 iterations × 550 cycles, 10 populations × 33, maxsize 30, 20 concepts, LLM weights 0.02 each. **LLM-SR:** llm-srbench default with 1,000 calls.

### Table 1: Acc_0.01 / Acc_0.001 on LSR-Synth (%), mean of 3 runs

Own method = SR-Scientist and +RL. All other rows are baselines **re-run by the SR-Scientist authors**, not copied.

| Method | Overall 0.01 | Overall 0.001 | MatSci 0.01 | MatSci 0.001 | Chem 0.01 | Chem 0.001 | Bio 0.01 | Bio 0.001 | Phys 0.01 | Phys 0.001 |
|---|---|---|---|---|---|---|---|---|---|---|
| *Without LLMs* | | | | | | | | | | |
| GPLearn | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| E2E | 0.26 | 0.00 | 1.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| NeSymReS | 3.10 | 0.78 | 8.00 | 4.00 | 2.78 | 0.00 | 4.17 | 0.00 | 0.00 | 0.00 |
| DSR | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| uDSR | 29.46 | 12.40 | 36.00 | 8.00 | 50.00 | 25.00 | 29.17 | 8.33 | 9.09 | 6.82 |
| **PySR** | **29.46** | **14.47** | **53.33** | **22.67** | **25.93** | **11.11** | **16.67** | **6.95** | **25.76** | **16.67** |
| *Qwen3-Coder-480B-A35B* | | | | | | | | | | |
| LaSR | 11.89 | 7.49 | 13.33 | 6.67 | 16.67 | 9.26 | 6.95 | 6.95 | 9.85 | 6.82 |
| LLM-SR | 41.08 | 18.09 | 80.00 | 52.00 | 36.11 | 9.26 | 30.56 | 18.06 | 28.79 | 6.06 |
| SR-Scientist | 49.09 | 24.55 | 86.67 | 69.33 | 40.74 | 5.56 | 50.00 | 26.39 | 34.09 | 13.64 |
| *GLM-4.5-Air* | | | | | | | | | | |
| LaSR | 14.21 | 8.53 | 18.67 | 12.00 | 16.67 | 10.18 | 11.11 | 5.56 | 11.36 | 6.82 |
| LLM-SR | 35.92 | 14.47 | 61.33 | 38.67 | 30.56 | 3.71 | 18.06 | 4.17 | 35.61 | 15.15 |
| SR-Scientist | 48.32 | 25.06 | 81.33 | 70.67 | 45.37 | 11.11 | 40.28 | 16.66 | 36.37 | 15.15 |
| *GPT-OSS-120B* | | | | | | | | | | |
| LaSR | 16.02 | 10.08 | 20.00 | 12.00 | 18.52 | 12.96 | 9.72 | 5.56 | 15.15 | 9.09 |
| LLM-SR | 28.16 | 11.37 | 64.00 | 38.67 | 22.22 | 3.71 | 11.11 | 2.78 | 21.97 | 6.82 |
| SR-Scientist | 63.57 | 49.35 | 74.67 | 60.00 | 81.48 | 64.81 | 66.67 | 43.05 | 40.91 | 34.09 |
| *GPT-OSS-20B* | | | | | | | | | | |
| LaSR | 12.66 | 8.53 | 14.67 | 8.00 | 15.74 | 12.96 | 11.11 | 6.95 | 9.85 | 6.06 |
| LLM-SR | 33.33 | 12.92 | 70.67 | 46.67 | 28.71 | 5.56 | 22.22 | 2.78 | 21.97 | 5.30 |
| SR-Scientist | 42.64 | 23.00 | 62.67 | 40.00 | 49.07 | 23.15 | 34.72 | 19.44 | 30.30 | 15.15 |
| *Qwen3-Coder-30B-A3B* | | | | | | | | | | |
| LaSR | 12.66 | 8.27 | 20.00 | 10.67 | 17.59 | 12.04 | 8.33 | 6.95 | 6.82 | 4.55 |
| LLM-SR | 24.55 | 7.24 | 29.33 | 6.67 | 30.55 | 10.18 | 15.28 | 5.55 | 21.97 | 6.06 |
| SR-Scientist | 32.30 | 16.02 | 81.33 | 52.00 | 22.22 | 5.56 | 22.22 | 8.33 | 18.18 | 8.33 |
| SR-Scientist + RL | 40.92 | 20.69 | 85.33 | 65.33 | 37.38 | 7.46 | 29.17 | 11.11 | 25.00 | 11.37 |

### App. Table 5: smaller backbones, Acc_0.01 / Acc_0.001

Values look like single runs; for example 22.48 = 29/129.

| Backbone | Method | Overall 0.01 | Overall 0.001 | MatSci 0.01 | MatSci 0.001 | Chem 0.01 | Chem 0.001 | Bio 0.01 | Bio 0.001 | Phys 0.01 | Phys 0.001 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-32B | LLM-SR | 30.49 | 9.04 | 34.67 | 18.67 | 38.89 | 5.56 | 15.28 | 4.17 | 29.55 | 9.09 |
| Qwen3-32B | SR-Scientist | 29.46 | 15.50 | 72.00 | 48.00 | 22.22 | 8.33 | 20.83 | 4.17 | 15.91 | 9.09 |
| Qwen3-32B | +RL | 39.53 | 20.16 | 80.00 | 64.00 | 38.89 | 8.33 | 25.00 | 4.17 | 25.00 | 13.64 |
| Qwen3-14B | LLM-SR | 27.91 | 9.30 | 56.00 | 32.00 | 16.67 | 2.78 | 8.33 | 0.00 | 31.82 | 6.82 |
| Qwen3-14B | SR-Scientist | 18.60 | 6.98 | 48.00 | 28.00 | 13.89 | 0.00 | 12.50 | 4.17 | 9.09 | 2.27 |
| Qwen3-14B | +RL | 23.26 | 10.08 | 52.00 | 40.00 | 16.67 | 2.78 | 20.83 | 0.00 | 13.64 | 4.55 |
| Qwen3-8B | LLM-SR | 22.48 | 8.53 | 48.00 | 28.00 | 16.67 | 5.56 | 8.33 | 0.00 | 20.45 | 4.55 |
| Qwen3-8B | SR-Scientist | 15.50 | 4.65 | 40.00 | 16.00 | 8.33 | 5.56 | 16.67 | 0.00 | 6.82 | 0.00 |
| Qwen3-8B | +RL | 20.93 | 9.30 | 68.00 | 44.00 | 16.67 | 0.00 | 4.17 | 0.00 | 6.82 | 2.27 |

### Table 2: symbolic accuracy (%) over all 129 LSR-Synth problems

| uDSR | **PySR** | LLM-SR (GLM) | LLM-SR (GPT) | LLM-SR (Qwen) | SR-Scientist (GLM) | SR-Scientist (GPT) | SR-Scientist (Qwen) |
|---|---|---|---|---|---|---|---|
| 0.77 | **4.65** | 5.43 | 4.65 | 5.43 | 7.75 | 7.00 | 7.00 |

GLM = GLM-4.5-Air, GPT = GPT-OSS-120B, Qwen = Qwen3-Coder-480B.

### App. Tables 6 and 7 (Qwen3-Coder-480B)

NMSE, physics only:

| Method | NMSE avg | NMSE max | NMSE min | NMSE median |
|---|---|---|---|---|
| LaSR | 1.53e-02 | 1.67e-01 | 2.66e-13 | 1.84e-03 |
| LLM-SR | 3.19e-03 | 7.83e-02 | 1.72e-14 | 8.87e-06 |
| SR-Scientist | 1.29e-03 | 4.06e-02 | 1.57e-14 | 2.64e-06 |

R²:

| Method | Overall | MatSci | Chem | Bio | Phys |
|---|---|---|---|---|---|
| LaSR | 0.991 | 0.998 | 0.996 | 0.988 | 0.985 |
| LLM-SR | 0.998 | 0.996 | 1.000 | 1.000 | 0.997 |
| SR-Scientist | 0.998 | 0.996 | 1.000 | 0.995 | 0.999 |

---

## 3. PiT-PO: LLM-Based Scientific Equation Discovery via Physics-Informed Token-Regularized Policy Optimization

- arXiv [2602.10576](https://arxiv.org/abs/2602.10576). HTML: <https://arxiv.org/html/2602.10576>. KDD 2026.
- Code: <https://github.com/CAS-CLab/PiT-PO>. The repo holds only LLM-SR Suite data (oscillator1/2, bactgrow, stressstrain CSVs). It has no LSR-Synth data.

### Metrics

- **Acc_all(τ) = 1(max_i |(ŷ_i−y_i)/y_i| ≤ τ)**: strict max, with no point dropping stated. This is the metric used on LLM-SRBench.
- **Acc_avg(τ) = (1/N) Σ_i 1(|(ŷ_i−y_i)/y_i| ≤ τ)**: the fraction of points. It is used on the LLM-SR Suite only.
- **NMSE** = mean((ŷ−y)²)/Var(y).
- **SA:** follows the LLM-SRBench protocol with GPT-4o.
- **OOD:** NMSE_OOD is reported for the LLM-SR Suite only (Table 4). There is no OOD number for LSR-Synth.

### Protocol

- **Budget:** 2,500 search iterations, with hyperparameters otherwise at LLM-SR defaults. Note that this differs from LLM-SRBench's 1k-call budget, while the baseline rows are copied from LLM-SRBench.
- **Model and training:** GRPO in-search fine-tuning, lr 1e-6, G=4, N=4 islands, KL 0.01, LoRA r=16. **4-bit quantized** Llama-3.1-8B, Llama-3.2-3B and Llama-3.2-1B on one RTX 3090.
- **Seeds:** the LLM-SR Suite curves show "median over seeds", but the number of seeds is not stated. No seed count is given for LLM-SRBench.
- **Final equation selection:** not stated.

### Table 2: LLM-SRBench with Llama-3.1-8B-Instruct

**The Direct, SGA, LaSR and LLM-SR rows are identical to the LLM-SRBench Llama-3.1-8B rows**: copied, not re-run. Only the PiT-PO row is the authors' own.

| Method | Transform SA | Transform Acc0.1 | Transform NMSE | Chem SA | Chem Acc0.1 | Chem NMSE | Bio SA | Bio Acc0.1 | Bio NMSE | Phys SA | Phys Acc0.1 | Phys NMSE | MatSci SA | MatSci Acc0.1 | MatSci NMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Direct Prompting (copied) | 3.61 | 1.801 | 0.3697 | 0.00 | 0.00 | 0.0644 | 0.00 | 0.00 | 0.5481 | 0.00 | 0.00 | 0.0459 | 0.00 | 0.00 | 0.0826 |
| SGA (copied) | 2.70 | 0.909 | 0.3519 | 0.00 | 8.33 | 0.0458 | 0.00 | 0.00 | 0.2416 | 0.00 | 2.27 | 0.1549 | 0.00 | 12.12 | 0.0435 |
| LaSR (copied) | 5.41 | 45.94 | 0.0021 | 0.00 | 27.77 | 2.77e-4 | 4.16 | 16.66 | 2.73e-4 | 4.54 | 25.02 | 0.0018 | 8.21 | 64.22 | 7.44e-5 |
| LLM-SR (copied) | 30.63 | 38.55 | 0.0101 | 8.33 | 66.66 | 8.01e-6 | 25.30 | 58.33 | 1.04e-6 | 6.97 | 34.09 | 1.23e-4 | 4.10 | 88.12 | 1.15e-7 |
| **PiT-PO (own)** | 34.23 | 46.84 | 0.0056 | 13.89 | 77.78 | 4.13e-7 | 29.17 | 70.83 | 9.37e-8 | 11.36 | 40.91 | 6.57e-5 | 12.00 | 92.00 | 1.18e-8 |

The PiT-PO values are clean k/N fractions (5/36, 28/36, 7/24, 17/24, 5/44, 18/44, 3/25, 23/25), which suggests a single run.

**PiT-PO's PySR row is on the LLM-SR Suite only** (Table 1). It does not cover LSR-Synth:

| | Osc1 Acc_avg0.001 | Osc1 NMSE | Osc2 Acc_avg0.001 | Osc2 NMSE | E. coli Acc_avg0.1 | E. coli NMSE | Stress-Strain Acc_avg0.1 | Stress-Strain NMSE |
|---|---|---|---|---|---|---|---|---|
| PySR | 3.80 | 0.0003 | 7.02 | 0.0002 | 2.80 | 0.4068 | 70.60 | 0.0347 |

---

## 4. RESTART: Robust Equation Structure Learning with Adaptive Refinement (Li & Pan)

- **No arXiv version found** (searched the arXiv API by title). It exists as an ICLR 2026 poster: OpenReview <https://openreview.net/forum?id=z9TKJhLVKj>.
- **I could not read the PDF.** OpenReview returns a bot challenge to curl and WebFetch, for both the `/pdf` and API attachment endpoints.
- Code: <https://github.com/Liyunlun/RESTART>, built on the llm-srbench code. It has no data and downloads `nnheui/llm-srbench`.

### From the code (primary source)

`bench/pipelines.py`:
- Acc_τ for τ ∈ {0.01, 0.05, 0.1, 0.2}, computed as **max relative error ≤ τ after discarding the 5% worst points**.
- The rule is: sort by |ŷ−y|/(|y|+1e-10) and keep `int(0.95·N)` points. It only applies when N > 20.
- NMSE uses var + 1e-10, and NaN predictions are dropped.

`configs/RESTART_qwen3_8b.yaml`:
- Qwen3-8B via vLLM, thinking disabled.
- `global_max_sample_num: 1000` (1k samples), 4 samples per prompt, 10 islands.
- E2E (Kamienny et al.) initialization and residual analysis, plus a concept library.

### From the secondary summary only (papernotes.org, NOT verified against the paper)

Source: <https://en.papernotes.org/ICLR2026/others/robust_equation_structure_learning_with_adaptive_refinement/>. Treat these as unverified until the PDF is read.

- LSR-Synth uses 93 tasks, i.e. **Chemistry is excluded** (24 + 44 + 25).
- The backbone is Qwen3-8B, and **the LLMSR baseline is apparently re-run with Qwen3-8B**.
- The ACC τ is not given in the summary.
- Bio 77.08 = 18.5/24, which suggests averaging over at least 2 runs.

| Dataset | LLMSR NMSE / ACC | RESTART NMSE / ACC |
|---|---|---|
| LSR-Transform (ID) | 0.160 / 74.32 | 0.157 / 74.77 |
| Biology (ID) | 0.016 / 70.83 | 0.001 / 77.08 |
| Material Science (ID) | 0.003 / 96.00 | 0.001 / 96.00 |
| Physics (ID) | 0.002 / 84.09 | 0.003 / 85.23 |
| Biology (OOD, NMSE capped at 100) | 6.667 / 45.83 | 5.087 / 52.08 |
| Material Science (OOD) | 0.084 / 96.00 | 0.075 / 100.00 |
| Physics (OOD) | 14.808 / 65.91 | 8.167 / 71.59 |

---

## 5. GenSR: Symbolic Regression Based in Equation Generative Space

- arXiv [2602.20557](https://arxiv.org/abs/2602.20557). HTML: <https://arxiv.org/html/2602.20557v1>. ICLR 2026 (repo `tokaka22/ICLR26-GENSR`, which contains only a README).
- **It does not evaluate on LLM-SRBench or LSR-Synth.** Its experiment tables cover only the SRBench black-box, Feynman and Strogatz sets (R², time, complexity at noise 0, 0.001, 0.01 and 0.1), plus latent-space ablations. It is a non-LLM method: a CVAE latent space searched with CMA-ES.
- There are no LSR-Synth numbers to compare against.

## 6. MOD-SR: Unifying Multimodal Learning and Direct Optimization with Gradient-Guided Diffusion for Symbolic Regression

- **No arXiv version found.** It is an ICML 2026 paper: OpenReview <https://openreview.net/forum?id=p06oR4sXgz> (bot-challenged, so I could not read the PDF). Code: <https://github.com/KROX777/MOD-SR>, with a local copy at `baselines/mod-sr/`.
- It is a non-LLM diffusion model. The ICML poster (<https://icml.cc/media/icml-2026/Slides/61549.pdf>) shows one results table, "Results (without post-processing)", with R² and complexity on:
  - Nguyen, Keijzer, Koza, Constant, Livermore, Vladislavleva, R and Jin;
  - SRBench Feynman and Strogatz.
- The repo assets contain only those benchmark lists. **I found no LSR-Synth results for MOD-SR**, though the full PDF could not be checked. A MOD-SR comparison on LSR-Synth would have to be run by us.

## 7. Data availability (LSR-Synth without the gated nnheui repo)

- **Original:** `nnheui/llm-srbench` is HF "gated: auto", i.e. auto-approved after you accept the terms while logged in. An unauthenticated request for `lsr_bench_data.hdf5` (239,063,948 B, which includes LSR-Transform) returns 401.
- **Ungated mirror: [`pkuHaowei/llm-srbench`](https://huggingface.co/datasets/pkuHaowei/llm-srbench).** Not gated: a range GET returned HTTP 206 with no auth.
  - Configs and example counts match exactly: `lsr_synth_chem_react` 36, `lsr_synth_bio_pop_growth` 24, `lsr_synth_phys_osc` 44, `lsr_synth_matsci` 25, `lsr_transform` 111.
  - Fields: `instance_id, description, train_input, train_output, test_input, test_output, ood_input, ood_output, input_vars, output_vars, gt_expression, symbols, subset`.

  | Published file | Size (B) | URL |
  |---|---|---|
  | chem_react | 3,249,980 | <https://huggingface.co/datasets/pkuHaowei/llm-srbench/resolve/main/lsr_synth_chem_react/train-00000-of-00001.parquet> |
  | bio_pop_growth | 2,167,007 | <https://huggingface.co/datasets/pkuHaowei/llm-srbench/resolve/main/lsr_synth_bio_pop_growth/train-00000-of-00001.parquet> |
  | phys_osc | 5,675,432 | <https://huggingface.co/datasets/pkuHaowei/llm-srbench/resolve/main/lsr_synth_phys_osc/train-00000-of-00001.parquet> |
  | matsci | 1,527,209 | <https://huggingface.co/datasets/pkuHaowei/llm-srbench/resolve/main/lsr_synth_matsci/train-00000-of-00001.parquet> |
  | lsr_transform | 172,033,305 + 181,374,427 | two shards |

  - Caveat: this is a third-party re-pack. Before trusting it, check that train/test/OOD sizes and values match the original HDF5 (`train_data`, `id_test_data`, `ood_test_data`), e.g. once someone accepts the nnheui terms.
- **Companion dataset:** [`pkuHaowei/llm-srbench-trajectories`](https://huggingface.co/datasets/pkuHaowei/llm-srbench-trajectories), also ungated. It holds per-instance candidate equations with nmse, acc01, sa and r2 for several agents and models, plus `candidates/PySR/1..6/results.jsonl` files of about 60 KB each. Provenance and run settings are unknown, so it is not a published PySR result.
- **Paper repos:** SR-Scientist, RESTART, PiT-PO, GenSR and MOD-SR contain **no copy of LSR-Synth data**. SR-Scientist and RESTART download it from nnheui. PiT-PO ships only the 4 LLM-SR Suite CSVs (e.g. `data/oscillator1/train.csv`, about 0.6 MB).

## 8. Cross-paper mismatches and comparability notes

1. **The Acc_τ definition differs:**
   - LLM-SRBench and PiT-PO use the strict max over all points.
   - SR-Scientist drops the worst 5%, implemented as "more than 95% of points within τ", with a special rule for y≈0.
   - RESTART's code also drops the worst 5%, via sort-and-keep 95% when N > 20.
   - A PySR evaluation should report both: strict max (comparable to LLM-SRBench and PiT-PO) and 5%-trimmed (comparable to SR-Scientist and RESTART).
   - The official llm-srbench repo does not implement Acc_τ.
2. **τ differs:** LLM-SRBench and PiT-PO use 0.1; SR-Scientist uses 0.01 and 0.001; RESTART's τ is unknown, and its code computes 0.01, 0.05, 0.1 and 0.2.
3. **LLM-SR numbers are not shared across papers:**
   - LLM-SRBench gives Acc_0.1 with Llama-3.1-8B, GPT-3.5 and GPT-4o-mini.
   - SR-Scientist re-ran LLM-SR with its own backbones at Acc_0.01. Its MatSci 29–80% is lower than LLM-SRBench's 82–88%, partly because of the tighter τ.
   - RESTART (per the secondary source) re-ran it with Qwen3-8B. Its Bio ACC is 70.83, vs 58.33 for Llama in LLM-SRBench.
   - PiT-PO copies the LLM-SRBench Llama rows while running its own method for 2,500 iterations, so the budgets are not matched.
4. **LaSR results diverge strongly.** In LLM-SRBench, LaSR is competitive (MatSci Acc_0.1 64–72%, Chem 28–39%). In SR-Scientist, LaSR is far below both PySR and LLM-SR (MatSci Acc_0.01 13–20%; overall 12–16% vs PySR 29.46%). The two setups differ in backbone, τ and LLM weights (0.005 vs 0.02).
5. **The two PySR rows are not comparable:**
   - LLM-SRBench: "default config, equivalent evaluations", SA/Acc_0.1, apparently a single run. Chem 41.67, Bio 25.0, Phys 29.55, MatSci 68.0, with SA 0/0/4.54/0.
   - SR-Scientist: a custom config (125 iterations, 15×33, maxsize 30, ≤100k candidates), Acc_0.01/0.001 with 5% trimming, mean of 3 runs. Overall 29.46/14.47, and SA 4.65 over all 129 problems.
   - In both papers PySR is competitive numerically: it beats LaSR in SR-Scientist, and is close to LaSR in LLM-SRBench.
6. **Problem counts differ:** LLM-SRBench's text says 128 (physics 43). The released data and SR-Scientist have 129 (physics 44). RESTART uses 93 and omits Chemistry.
7. **Aggregation:**
   - LLM-SRBench values are often not k/N and the run count is unstated.
   - SR-Scientist averages 3 runs.
   - PiT-PO and LLM-SRBench's PySR values look like single runs.
   - SR-Scientist's "Overall" is pooled over the 129 problems (e.g. 29.46 ≈ 38/129).
8. **Final equation selection:**
   - SR-Scientist takes the best train MSE/NMSE.
   - LLM-SR in llm-srbench takes the best train score, with no validation split.
   - PiT-PO and RESTART do not state their selection rule.
   - None of these papers selects on the ID test split.
9. **OOD:** no paper gives a numeric LSR-Synth OOD table except RESTART (secondary source, no Chem). LLM-SRBench and SR-Scientist show OOD only in figures.

## 9. Not covered, possibly relevant

These other 2026 arXiv papers report on LSR-Synth. I did not extract numbers from them.

- *Library Reachability in LSR-Synth* ([2607.28684](https://arxiv.org/abs/2607.28684)): a semantics-free fixed-library baseline.
- A-SR ([2608.04872](https://arxiv.org/abs/2608.04872)).
- Deliberate Evolution ([2606.04360](https://arxiv.org/abs/2606.04360)).
- Programmatic Context Augmentation ([2605.03101](https://arxiv.org/abs/2605.03101)).

## 10. Compute / wall-clock per problem

**Bottom line:** no paper reports a measured wall-clock time per LSR-Synth problem for any baseline. The only LSR-Synth compute figures are SR-Scientist's cost (≈$0.25 per problem) and a batch time (≤5 h for all 129 problems). PiT-PO gives a fixed 25,000 s per-problem budget, but only on the LLM-SR Suite. Its public logs contain real per-run timings for oscillator1.

### 10a. Reported figures (quoted)

| Source | Where | Quote / figure |
|---|---|---|
| LLM-SRBench (2504.10415) | Sec. 3.1 | "we standardize each of the methods to use 1k LLM calls per problem while maintaining their core algorithmic designs and hyperparameter settings." |
| LLM-SRBench | App. D Table 2 (PDF) | LLM-SR: "Time limit T = 30s per program hypothesis, Memory limit M = 2GB", "Batch size b = 4 equation programs per prompt", "e = 4 parallel evaluators", "m = 10 islands". Direct: "Time limit T = 30s per program hypothesis execution". LaSR: "Iterations = 25, Cycles per iteration = 550, Populations = 10, Population size = 33". |
| LLM-SRBench | App. F (PDF) | PySR: "We used PySR's default configuration with the same computational budget (equivalent number of evaluations) as the LLM-based methods". **No wall-clock, GPU-hour or $ figure appears anywhere in the paper, appendix included.** |
| SR-Scientist (2510.11661) | Sec. 4.1 | "For methods without LLMs, we constrain the number of equation candidates for each problem to 100,000. For methods with LLMs, we constrain the LLM calls to 1,000, with at most 1,000 equation candidates from each problem. For SR-Scientist, we set the maximum number of turns per iteration to 25, with a total of 40 iterations." |
| SR-Scientist | App. D.2, Table 8 | "SR-Scientist (GPT-OSS-120B) — Input: $0.05/1M; Output: $0.25/1M — **$0.25** [API cost per problem]"; "SR-Scientist (GPT-OSS-20B) — Input: $0.03/1M; Output: $0.15/1M — **$0.1**". Caption: "This calculation does not consider cached tokens". |
| SR-Scientist | App. D.2 | "we deployed GPT-OSS-120B on a local server with 2 NVIDIA H100s for batch-level inference on 129 problems and recorded the wall-clock times. **The maximum time was no more than 5 hours**". This is batch-parallel over all 129 problems, not a per-problem serial time. |
| SR-Scientist | App. B / C.3 | Temperature 0.7, "maximum completion length per call to 8,192". RL training: "We conduct RL training on 32 NVIDIA H200 GPUs"; elsewhere "4 nodes, each with 8 H200 GPUs" (README); reward "saturates at around 60 steps". No GPU-hours are given. No baseline timings are given. |
| PiT-PO (2602.10576) | Sec. 4.2 | "All experiments were run for 2,500 search iterations." "experiments were conducted on a single NVIDIA RTX 3090 using 4-bit quantized Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, and Llama-3.1-8B-Instruct" |
| PiT-PO | App. C.2, Fig. 9 | "We fix the total runtime of each method to **25,000 seconds (≈ 6.9 hours)** and track the best-so-far NMSE as a function of elapsed time". This applies to LLM-SR vs PiT-PO on the 4 LLM-SR Suite problems. Main text: "While the in-search fine-tuning introduces a computational overhead…" **No per-problem time is given for LLM-SRBench.** |
| LLM-SR (2404.18400) | Sec. 4 / App. | "We run LLM-SR variants for around 2.5K iterations in all experiments." "each iteration samples b=4 equation skeletons per prompt with temperature τ=0.8, optimizes parameters via numpy+BFGS or torch+Adam (with 30 seconds timeout)". Backbone: "Mixtral-8x7B (using 4 NVIDIA RTX 8000 GPUs with 48GB memory each) or GPT-3.5-turbo (via OpenAI API)". Island reset: "Every T_reset iterations (every 4 hrs)". The 4 h reset period implies runs longer than 4 h, but **no total runtime is reported**. Baselines (PySR, DSR, etc.): "over 2 million iterations", "5 replications". |
| RESTART | code `configs/RESTART_qwen3_8b.yaml` | `global_max_sample_num: 1000`, `samples_per_prompt: 4`, `num_islands: 10`, `num_evaluators: 4`, E2E init/residual `e2e_timeout: 60`, Qwen3-8B via vLLM. Paper: not accessible. Secondary summary (papernotes): RESTART "surpass[es] LLMSR with only 25% of the iteration budget". No time figures found. |

### 10b. Measured logs in public repos (primary data, but not the paper setup)

The PiT-PO repo ships two run logs:
- Run directories: `PiT-PO_for_Paddle/train_log/oscillator1_gpu{0,1}_seed{3408,3409}_retry1/`, with files `time_summary.json`, `config.json` and `run_summary.json`.
- Config: **Llama-3.2-3B, bf16, Paddle port**, oscillator1 (LLM-SR Suite), 2,500 samples, 4 per prompt, `evaluate_timeout_seconds: 30`, GRPO every 64 samples.
- Logged times:

| run | total_sample_time (s) | total_evaluate_time (s) | total_grpo_time (s) | avg_sample (s) | avg_evaluate (s) |
|---|---|---|---|---|---|
| seed3408 | 8473.4 | 8706.7 | 459.8 | 4.57 | 4.69 |
| seed3409 | 9951.8 | 8624.5 | 556.8 | 4.80 | 4.16 |

Summing the three components gives ≈17.6k–19.1k s (≈4.9–5.3 h) per problem, if the phases don't overlap. That sum is my arithmetic, not a reported number.

### 10c. Code-config facts (llm-srbench, local `baselines/llm-srbench-src`)

LLM-SR:
- `configs/llmsr_*.yaml` sets `global_max_sample_num: 1000`, `samples_per_prompt: 4`, `num_islands: 10`.
- `methods/llmsr/config.py` defaults to `evaluate_timeout_seconds: 30` and `num_evaluators: 1`; the yaml does not override it, while the paper says e=4.
- In `methods/llmsr/sampler.py`, each sample is a separate `async_draw_single_sample` call, with up to 20 concurrent requests. The 1000 budget therefore equals 1000 LLM completions, i.e. 1000 candidate programs, each BFGS-fitted under the 30 s cap.
- The search also stops early if the best score is exactly 0.

LaSR:
- `configs/lasr_*.yaml` sets `num_iterations: 25`, `num_populations: 10`, `llm_weight: 0.001`, `max_num_samples: 2000`.
- `methods/lasr/searcher.py` sets `ncyclesperiteration=550`, `population_size=33`, `maxsize=30`, and an LLM `readtimeout=360`.
- Note: the yaml `llm_weight` of 0.001 differs from the paper's 0.005.

Direct prompting: `global_max_sample_num: 5`. SGA has no iteration or timeout field in its yaml.

### 10d. My estimates (not reported anywhere)

- **LLM-SR / RESTART at 1k samples:**
  - The evaluation worst case is 1000 × 30 s ≈ 8.3 h serial, or ≈2.1 h with 4 evaluators.
  - The PiT-PO logs show a mean of ≈4–5 s per evaluation and ≈4.6–4.8 s per LLM sample for a local 3B model on one GPU.
  - Applied to 1k samples, that gives ≈2.5 h per problem serial, or less with overlap and API concurrency.
  - Rough range: **≈0.5–3 h per problem**, dominated by BFGS evaluation and LLM latency.
- **PiT-PO / LLM-SR at 2,500 samples:** ≈5 h per problem on one RTX 3090, consistent with the logs above and with the 25,000 s budget used in C.2.
- **SR-Scientist:** about 1k LLM calls with up to 8,192 completion tokens and multi-turn contexts. The reported ≤5 h for all 129 problems on 2 H100s, run in parallel, plus ≈$0.1–0.25 per problem in API cost, are the only anchors. A per-problem serial time cannot be derived from them.
- **PySR baselines:**
  - SR-Scientist's 100k-candidate cap and LLM-SRBench's "equivalent evaluations" are evaluation budgets, not time budgets.
  - A 1k-evaluation-matched PySR run would take seconds to minutes. SR-Scientist's config (125 × 550 × 15 × 33 ≈ 34M mutations, capped at 100k candidates) is unclear in its interpretation; no times are reported.
