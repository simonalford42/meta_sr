# Authenticated GitHub query pass

Review date: 2026-07-03. This pass used GitHub's authenticated REST code-search endpoint with the exact base query `PySRRegressor`. Authentication came from the existing Git credential helper; the credential was neither displayed nor written to an output file.

No repository was cloned and no repository code was executed. Manual evidence was limited to public repository metadata, README text, and matching source/notebook paths.

## Retrieval result

| Item | Count |
|---|---:|
| Unqualified GitHub `total_count` response | 2,736 |
| Raw records returned by size-partitioned pagination | 2,638 |
| Duplicate public file records removed | 310 |
| Unique public matching files | 2,328 |
| GitHub repositories represented | 637 |
| Private file records written | 0 |
| Fork repositories represented | 0 |
| GitHub code-search requests | 57 |

GitHub's `total_count` was inconsistent. For example, repeated pages of the same 2,048–4,095 byte query reported totals of 438 and 518, while empty-page pagination returned 518 records. Another partition returned 414 records after initially reporting 393. The script therefore never uses `total_count` as a stopping condition: it retrieves pages until GitHub returns an empty page, then deduplicates by repository, path, and blob SHA.

The search was partitioned into ten non-overlapping file-size intervals covering 0–393,215 bytes. Each returned fewer than 1,000 records, avoiding GitHub's per-query retrieval cap. Full query/page observations are in `data/github_search_snapshot.json`.

GitHub documents additional legacy code-search restrictions: only default branches are indexed, archived repositories are excluded, files must be smaller than 384 KiB, and repository activity/size and fork rules affect coverage. See [GitHub's legacy code-search documentation](https://docs.github.com/en/search-github/searching-on-github/searching-code).

## Reconciliation with Sourcegraph

After canonicalizing the renamed `MilesCranmer/PySR` repository to `astroautomata/PySR`:

| Coverage | Repositories |
|---|---:|
| GitHub only | 602 |
| Sourcegraph only | 3 |
| Both | 35 |
| **Union** | **640** |

The three Sourcegraph-only repositories are `AdityaLab/lstprompt`, `keyonvafa/inductive-bias-probes`, and `sdascoli/odeformer`. All remain public and unarchived, demonstrating that neither index alone is complete.

## Screening process

`scripts/triage_github.py` gives every GitHub repository a reproducible prioritization score based on:

- domain keywords in repository metadata and matching paths;
- research/data signals;
- whether matches occur in primary code rather than docs, tests, tutorials, vendors, or generated artifacts;
- negative signals for generic SR methods, benchmarks, educational material, and agent/LLM repositories.

This produced 60 high-priority, 141 medium-priority, and 436 deprioritized rows. The score is only a screening mechanism.

Manual review covered 64 plausible applications. The decisions were:

| Classification | Count |
|---|---:|
| `real_science_candidate` | 45 |
| `scientific_synthetic` | 13 |
| `unclear` | 4 |
| `method_or_incidental` | 2 |

The 13 `scientific_synthetic` repositories remain useful for a separate benchmark tier: they are genuine scientific work, but their PySR targets are known simulated systems, controlled method tests, or learned-model behavior rather than primary measured relations.

## Recommended first 25

These are the high-priority real-science candidates. “Simulation” here can still mean a genuine domain problem—such as cosmological emulation or aerodynamics—not a toy formula benchmark.

| Repository | Data origin | Why prioritize |
|---|---|---|
| `arnablahiry/SymReg-L1-Norm` | Simulation | Cosmological parameter-to-wavelet-statistic mappings with several direct PySR pipelines. |
| `Dr-Yehia/corrosion-rc-beam-optimizer` | Experimental | Checked-in data for 804 experimental reinforced-concrete beam specimens. |
| `eather0056/Catenary-Model-Estimation-and-MPC-Control-for-ROV-Tethered-Systems` | Mixed | Laboratory ROV tether dynamics with experimental and simulated datasets. |
| `eelregit/5par` | Mixed | Published reionization-history relation evaluated on simulation and observational data. |
| `EyringMLClimateGroup/grundner23james_EquationDiscovery_CloudCover` | Mixed | Published cloud-cover equations from DYAMOND and ERA5/ERA5.1 data. |
| `FAIR-UMN/FAIR-UMN-CDMS` | Mixed | SuperCDMS detector interaction-location problem with reduced data included. |
| `generalroboticslab/sym2real` | Experimental | Symbolic dynamics with an explicit real-hardware adaptive-control workflow. |
| `Gotsmy/FBA-Hyb` | Experimental | Experimental *E. coli* fed-batch time series and extensive direct PySR fitting code. |
| `Jie0618/PhysicsRegression` | Observational | Five real-world space-physics cases and linked reproduction data. |
| `jibanCat/lya1d_priya_forecast` | Simulation | Paper-quality symbolic distillation of a Lyman-alpha cosmology emulator with result sidecars. |
| `komimensah/Symbolic-Regression-for-Trait-Environment-Functions` | Observational | Biologically constrained thermal development-rate relations. |
| `rogeriog/MatterVial` | Observational | Published materials-feature interpretation across established materials datasets. |
| `mbejger/pysr_r-as-mlambda` | Simulation | Focused neutron-star radius paper with input data and reported PySR model. |
| `MilesCranmer/pysr_scaling_laws` | Empirical | Small published Llama scaling dataset and explicit discovered laws; low extraction cost. |
| `Pablo-Lemos/orbits` | Observational | Orbital-law rediscovery from NASA ephemeris data. |
| `peterdsharpe/AeroSandbox` | Simulation | Five compact engineering studies whose formulas support a real aircraft-design package. |
| `pragmaticscientist/animal-brain-decoder` | Observational | Comparative animal brain morphology and behavior data with paper reproduction scripts. |
| `qnwang93/SR_SNIa_2025summer` | Observational | Dedicated fits for several observed supernova light curves. |
| `Ragerlab/2026_Chappel_Interpretable_Machine_Learning_to_Understand_Wildfire_Toxicity` | Experimental | Chemical-exposure and transcriptomic-response datasets with a dedicated PySR stage. |
| `SiyuLou/UnsupervisedHierarchicalSymbolicRegression` | Mixed | Published molecular structure-polarity relations with separate solvent/solute fits. |
| `strifinopoulos/Nuclear_DNA` | Observational | Interprets nuclear-mass models over nuclear-observable datasets. |
| `TeresaTonelli/GP-4-Alkalinity` | Observational | Mediterranean Sea alkalinity from spatiotemporal NetCDF/CSV data. |
| `TomTom9595/Cas12a-Complex-Environment` | Experimental | Figure-specific PySR analysis of experimental CRISPR/Cas12a behavior. |
| `wangfelix/symbolic-regression-algorithms-for-modeling-power-grid-frequency-dynamics` | Mixed | Clearly separated empirical and synthetic power-grid dynamics experiments. |
| `ZehaoJin/Ultimate_black_hole_mass_scaling_relations_Symbolic_Regression` | Observational | Roughly 100 direct dynamical black-hole mass measurements and extensive SR notebooks. |

The other 20 real-science candidates are retained in `data/github_manual_review.csv`. They are mostly medium priority because documentation, data availability, or the exact PySR-to-paper connection needs verification.

## Next-pass implications

The next phase should start from the 25 rows above, but select approximately 20–30 **problems**, not necessarily repositories. `PhysicsRegression`, `AeroSandbox`, and several other repositories contain multiple distinct targets. A sensible first extraction order remains:

1. Small, explicit problems: scaling laws, neutron-star radius, orbits.
2. Multi-problem repositories with direct data/code: AeroSandbox, PhysicsRegression, FBA-Hyb.
3. Larger published pipelines: cloud cover, Lyman-alpha, reionization, materials, wildfire toxicity.

Each extracted problem should pin a commit, record the exact `X`, `y`, weights, train/test split, operators, PySR configuration, reported equation, metrics, and the paper's validity criterion.
