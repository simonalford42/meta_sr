# First-pass decision log

Review date: 2026-07-02. Evidence was read-only: Sourcegraph search results, repository metadata, README files, and matched public source/notebook files. “Candidate” means worth a deeper extraction/reproduction pass; it does not yet mean the data, configuration, or published result is reproducible.

## Summary

| Classification | Repositories |
|---|---:|
| `real_science_candidate` | 6 |
| `scientific_synthetic` | 4 |
| `sr_method_or_benchmark` | 11 |
| `incidental_or_educational` | 16 |
| `false_positive` | 1 |
| **Total** | **38** |

The six first-pass real-science candidates plausibly contain more than six benchmark problems. PhysicsRegression describes five real-world physics cases, and AeroSandbox has five distinct matched engineering studies. This means the next pass can be selective without expanding the discovery query immediately.

## Real-science candidates

| Repository | Priority | Decision and evidence |
|---|---|---|
| [Gy-Hu/E-Syn](https://github.com/Gy-Hu/E-Syn) | Medium | DAC engineering research fits circuit-analysis metrics from standard circuit benchmarks with PySR. Keep provisionally; the README points paper reproduction to the `new` branch, so branch/data lineage must be resolved. |
| [Jie0618/PhysicsRegression](https://github.com/Jie0618/PhysicsRegression#applications) | High | README explicitly lists five real-world physics SR cases: sunspot number, equatorial plasma pressure, solar differential rotation, contribution functions, and lunar tides. `PhysicsRegression.py` optionally calls `PySRRegressor.fit(x[i], y[i])` as a warm start. |
| [jil095/tinyRNN](https://github.com/jil095/tinyRNN/blob/HEAD/analyzing_experiments/analyzing_dynamics.py) | Medium | PySR is applied to learned state transitions of small RNNs trained on rat/monkey behavioral datasets. Keep provisionally because the matched file comments out its PySR imports, so the path may be stale or broken. |
| [MilesCranmer/pysr_scaling_laws](https://github.com/MilesCranmer/pysr_scaling_laws) | High | Focused wild application to published Llama 2 scaling data; README shows the source paper and reported discovered laws. Likely the cheapest candidate to extract and reproduce. |
| [NumCosmo/NumCosmo](https://github.com/NumCosmo/NumCosmo/blob/HEAD/notebooks/primordial_perturbations/two_fluids.ipynb) | High | Notebook computes two-fluid primordial-perturbation spectra and runs PySR on log-wavenumber/log-power arrays. It is genuine cosmology computation, but the notebook does not clearly connect the PySR cell to a paper/result. |
| [peterdsharpe/AeroSandbox](https://github.com/peterdsharpe/AeroSandbox/tree/HEAD/studies) | High | Five matched studies derive compact engineering fits for control-surface effectiveness, superellipse perimeter, lifting-line calibration, and critical Mach behavior. These are high-value wild engineering tasks and may yield several benchmark problems. |

## Scientific but synthetic

| Repository | Decision and evidence |
|---|---|
| [divelab/AIRS](https://github.com/divelab/AIRS/tree/HEAD/OpenODE/DIF) | IFL-DIF is an ICML scientific-discovery paper; PySR explains learned invariant dynamics on simulated pendulum, Lotka–Volterra, and epidemic ODE tasks. Useful as a secondary, known-system tier. |
| [keyonvafa/inductive-bias-probes](https://github.com/keyonvafa/inductive-bias-probes/blob/HEAD/inductivebiasprobes/experiments/physics/fit_symbolic_regression.py) | Clear extraction path and paper result, but the solar systems and force data are generated to probe foundation-model world models. |
| [m2lines/L96_demo](https://github.com/m2lines/L96_demo/blob/HEAD/notebooks/symbolic_methods_comparison.ipynb) | Climate-science pedagogy comparing equation discovery on a simulated two-scale Lorenz-96 parameterization. |
| [mariodeflorio/AI-Lorenz](https://github.com/mariodeflorio/AI-Lorenz) | Research code recovers known terms of Lorenz, Sprott, and 6D hyperchaotic systems from noisy/sparse simulated observations. |

## SR method, benchmark, or infrastructure

| Repository | Decision |
|---|---|
| [astroautomata/SymTorch](https://github.com/astroautomata/SymTorch) | PySR-based interpretability infrastructure for neural-network components. |
| [cavalab/srbench](https://github.com/cavalab/srbench) | General SR benchmark; PySR is a competitor. |
| [cool-japan/scirs](https://github.com/cool-japan/scirs/blob/HEAD/scirs2-symbolic/bench-comparison/run_pysr.py) | Benchmarks its Rust SR engine against PySR. |
| [deep-symbolic-mathematics/llm-srbench](https://github.com/deep-symbolic-mathematics/llm-srbench) | Scientific equation-discovery benchmark; PySR occurs inside a method. |
| [fastmachinelearning/hls4ml](https://github.com/fastmachinelearning/hls4ml/blob/HEAD/hls4ml/utils/symbolic_utils.py) | PySR-backed symbolic conversion/optimization utility and tests, not a domain application. |
| [hftsoi/symbolfit](https://github.com/hftsoi/symbolfit) | PySR-based fitting framework motivated by HEP; checked-in runnable datasets are demonstrations/toys. |
| [intell-sci-comput/PSE](https://github.com/intell-sci-comput/PSE) | SR-method repository with PySR baselines. |
| [MilesCranmer/PySR](https://github.com/MilesCranmer/PySR) | PySR implementation itself. |
| [MilesCranmer/pysr_paper](https://github.com/MilesCranmer/pysr_paper) | PySR method paper and empirical benchmark. |
| [sdascoli/odeformer](https://github.com/sdascoli/odeformer/blob/HEAD/odeformer/baselines/pysr_wrapper.py) | ODEFormer method repository; PySR is a baseline. |

## Incidental, educational, or non-central use

| Repository | Decision |
|---|---|
| [AdityaLab/lstprompt](https://github.com/AdityaLab/lstprompt/blob/HEAD/experiments/run_simplicity_bias.py) | Auxiliary forecast simplicity-bias analysis. |
| [deepmodeling/AI4S-agent-tools](https://github.com/deepmodeling/AI4S-agent-tools/blob/HEAD/servers/Symbolic_regression/src/symbolic_regression.py) | Generic agent tool/server wrapper. |
| [deepmodeling/build-your-agent](https://github.com/deepmodeling/build-your-agent/blob/HEAD/agents/SRAgent/Nexusagent_SR/tool/pysr.py) | Generic example-agent wrapper. |
| [ECNU-ICALK/AutoSkill](https://github.com/ECNU-ICALK/AutoSkill/blob/HEAD/SkillBank/ConvSkill/english_gpt4_8_GLM4.7/symbolic-regression-for-constants-using-pysr/SKILL.md) | Generated skill-bank prose, not an experiment. |
| [fastmachinelearning/hls4ml-tutorial](https://github.com/fastmachinelearning/hls4ml-tutorial/blob/HEAD/part8_symbolic_regression.ipynb) | Tutorial notebook. |
| [FrontierCS/Frontier-CS](https://github.com/FrontierCS/Frontier-CS/tree/HEAD/research/problems/symbolic_regression) | Five deliberately synthetic function-recovery tasks. |
| [GDS-Education-Community-of-Practice/DSECOP](https://github.com/GDS-Education-Community-of-Practice/DSECOP/blob/HEAD/Automated_Object_Detection/03_auto_tracking.py) | Educational object-tracking module. |
| [Human-Agent-Society/CORAL](https://github.com/Human-Agent-Society/CORAL/tree/HEAD/examples/frontier_cs_research) | Vendored Frontier-CS synthetic tasks used for agent evaluation. |
| [marcomusy/vedo](https://github.com/marcomusy/vedo/blob/HEAD/examples/extras/pysr_regression.py) | Visualization/example script. |
| [MrTomRod/scoary-2](https://github.com/MrTomRod/scoary-2/blob/HEAD/benchmarking/picking_performance/benchmark_picking.py) | Fits software benchmark performance, not biological relations. |
| [ngruver/llmtime](https://github.com/ngruver/llmtime/blob/HEAD/experiments/run_simplicity_bias.py) | Auxiliary forecast simplicity-bias analysis. |
| [PaddlePaddle/PaddleCFD](https://github.com/PaddlePaddle/PaddleCFD/blob/HEAD/examples/symbolic_gn/README.md) | Illustrative README code explicitly labeled future work; no executable PySR call found. |
| [raj-brown/APMA_2070_ENGN_2912_SPRING_2024](https://github.com/raj-brown/APMA_2070_ENGN_2912_SPRING_2024/blob/HEAD/Lecture_7b_Notebook/pysr.ipynb) | Course lecture notebook. |
| [THUIR/MemoryBench](https://github.com/THUIR/MemoryBench/blob/HEAD/baselines/AutoSkill/SkillBank/ConvSkill/english_gpt4_8_GLM4.7/symbolic-regression-for-constants-using-pysr/SKILL.md) | Vendored generated AutoSkill prose. |
| [thuml/MiniVeo3-Reasoner](https://github.com/thuml/MiniVeo3-Reasoner/tree/HEAD/data/maze/maze-dataset) | Vendored maze-dataset benchmark and generated coverage page. |
| [understanding-search/maze-dataset](https://github.com/understanding-search/maze-dataset/blob/HEAD/maze_dataset/benchmark/sweep_fit.py) | Fits package benchmark/runtime behavior. |

## False positive

| Repository | Decision |
|---|---|
| [osirislab/LeakyPastes-V2](https://github.com/osirislab/LeakyPastes-V2/blob/HEAD/pastes/pastes_20230925120434.csv) | `PySRRegressor` occurs inside an archived Pastebin CSV row; the repository does not use PySR. |

## Next-pass order

1. `MilesCranmer/pysr_scaling_laws` — small, explicit data source, explicit reported result.
2. `peterdsharpe/AeroSandbox` — several compact engineering fits with direct source paths.
3. `Jie0618/PhysicsRegression` — five real-world cases and linked data, but heavier dependencies.
4. `NumCosmo/NumCosmo` — direct X/y construction, though result provenance is unclear.
5. `jil095/tinyRNN` and `Gy-Hu/E-Syn` — retain as candidates but first resolve stale imports/branch mismatch.

After those, the four `scientific_synthetic` repositories are good candidates for a separate benchmark tier with known or controlled ground truth.
