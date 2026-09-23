#!/usr/bin/env bash
# Main experiments. Commands are commented out; copy a block into submit_jobs.sh to run it.
# "official:" names the existing run behind the current result. Re-runs write to runs/official-*.
# Evaluations read the artifacts below; point them at runs/official-* to rerun end to end.

HPO_GT=outputs/hpo_pysr_20260824_180547_120309
HPO_GTR2=outputs/hpo_pysr_20260824_190637_506162
HPO_R2=outputs/hpo_pysr_20260824_183759_524347
PYSR_GT=runs/709715
PYSR_GTR2=runs/941339
PYSR_R2=runs/120458
BASICSR_GT=runs/225437
BASICSR_GTR2=runs/229869
BASICSR_R2=runs/150812
MIPS_PYSR=runs/709714
NEURON_PYSR=runs/708907
MIPS_ROOT="$(pwd)/outputs/mips_evolution_51_artifacts"

TRAIN_SPLITS=(--split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt)


# (1) HPO: 500 trials, then pick finalists from the first 300.
# official: $HPO_GT, $HPO_GTR2, $HPO_R2 (reselected from outputs/hpo_pysr_20260727_172105_{644009,644293,644046})
HPO_ARGS=(--n-trials 500 --n-runs 3 --n-parallel 20 --split splits/barely_unsolvable.txt --random-target-noise)
HPO_RESELECT_ARGS=(--n-trials 300 --n-runs 3 --n-runs-final 10 --final-topk 10 --n-parallel 20 --split splits/barely_unsolvable.txt --val-split splits/val.txt --random-target-noise)

# h1=$(sbatch --parsable -J hpo-gt    run.sh hpo_pysr.py --fitness-metric gt    "${HPO_ARGS[@]}" --output-dir runs/official-hpo-gt-500) || exit 1
# h2=$(sbatch --parsable -J hpo-gt-r2 run.sh hpo_pysr.py --fitness-metric gt-r2 "${HPO_ARGS[@]}" --output-dir runs/official-hpo-gt-r2-500) || exit 1
# h3=$(sbatch --parsable -J hpo-r2    run.sh hpo_pysr.py --fitness-metric r2    "${HPO_ARGS[@]}" --output-dir runs/official-hpo-r2-500) || exit 1
# sbatch --dependency=afterok:"$h1" -J hpo300-gt    run.sh hpo_pysr.py --reselect-from runs/official-hpo-gt-500    --fitness-metric gt    "${HPO_RESELECT_ARGS[@]}" --output-dir runs/official-hpo-gt || exit 1
# sbatch --dependency=afterok:"$h2" -J hpo300-gt-r2 run.sh hpo_pysr.py --reselect-from runs/official-hpo-gt-r2-500 --fitness-metric gt-r2 "${HPO_RESELECT_ARGS[@]}" --output-dir runs/official-hpo-gt-r2 || exit 1
# sbatch --dependency=afterok:"$h3" -J hpo300-r2    run.sh hpo_pysr.py --reselect-from runs/official-hpo-r2-500    --fitness-metric r2    "${HPO_RESELECT_ARGS[@]}" --output-dir runs/official-hpo-r2 || exit 1


# (2) PySR++ (evolve_pysr.py). official: $PYSR_GT, $PYSR_GTR2, $PYSR_R2
# TODO: the three objectives used different protocols: GT = 1e6 evals, GT-R2 = 90 s budget, R2 = older topk + smart reeval + random target noise.
PYSR_ARGS=(--operator-type all --population-type task --generations 45 --simplify-cooldown 15 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2)
PYSR_90S=(--val-n-runs 10 --identify-topk 10 --final-eval-runs 10 --max-time-in-seconds 90 --pysr-wall-limit 270 --val-pysr-timeout 90 --val-pysr-wall-limit 270)

# sbatch -J pysr-gt    run.sh evolve_pysr.py --fitness-metric gt    "${PYSR_ARGS[@]}" "${TRAIN_SPLITS[@]}" --output-dir runs/official-pysr-gt
# sbatch -J pysr-gt-r2 run.sh evolve_pysr.py --fitness-metric gt-r2 "${PYSR_ARGS[@]}" "${TRAIN_SPLITS[@]}" "${PYSR_90S[@]}" --output-dir runs/official-pysr-gt-r2
# TODO: 120458 used --reeval smart --max-runs-per-generation 60, both since removed. smart now maps to TTTS-dynamic; budget 45 = 60 minus 5 offspring x 3 runs, not verified.
# sbatch -J pysr-r2    run.sh evolve_pysr.py --fitness-metric r2 --operator-type all --population-type topk --generations 30 --population 10 --offspring 5 --n-runs 3 --max-evals 1000000 --exec-feedback-n 3 --models best --reeval TTTS-dynamic --reeval-budget 45 --random-target-noise "${TRAIN_SPLITS[@]}" --output-dir runs/official-pysr-r2


# (3) BasicSR++ (evolve_fullsr.py). official: $BASICSR_GT, $BASICSR_GTR2, $BASICSR_R2
# GT-R2 is 150815, then two simplify-only continuations (150815-simplify-30-best2, then 229869).
BASICSR_ARGS=(--operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --models best)
BASICSR_SIMPLIFY_ARGS=(--fitness-metric gt-r2 --generations 30 --mutation-mode simplify --population-type complexity --population 10 --offspring 10 --n-runs 3 --val-n-runs 10 --models best2 --max-evals 1000000 --timeout 500 --fullsr-wall-limit 600 --val-fullsr-timeout 1500 --val-fullsr-wall-limit 1800)

# sbatch -J basicsr-gt run.sh evolve_fullsr.py --fitness-metric gt "${BASICSR_ARGS[@]}" "${TRAIN_SPLITS[@]}" --output-dir runs/official-basicsr-gt
# sbatch -J basicsr-r2 run.sh evolve_fullsr.py --fitness-metric r2 "${BASICSR_ARGS[@]}" "${TRAIN_SPLITS[@]}" --output-dir runs/official-basicsr-r2
# b=$(sbatch --parsable -J basicsr-gt-r2 run.sh evolve_fullsr.py --fitness-metric gt-r2 "${BASICSR_ARGS[@]}" "${TRAIN_SPLITS[@]}" --output-dir runs/official-basicsr-gt-r2-stage1) || exit 1
# b=$(sbatch --parsable --dependency=afterok:"$b" -J basicsr-gt-r2-simp1 run.sh evolve_fullsr.py --continue-from runs/official-basicsr-gt-r2-stage1 "${BASICSR_SIMPLIFY_ARGS[@]}" "${TRAIN_SPLITS[@]}" --output-dir runs/official-basicsr-gt-r2-stage2) || exit 1
# sbatch --dependency=afterok:"$b" -J basicsr-gt-r2-simp2 run.sh evolve_fullsr.py --continue-from runs/official-basicsr-gt-r2-stage2 "${BASICSR_SIMPLIFY_ARGS[@]}" "${TRAIN_SPLITS[@]}" --output-dir runs/official-basicsr-gt-r2


# (4) SRBench ground truth + black box, 90 s per fit, 10 seeds, all noise levels.
# official: runs/<method>-srbench_full_9-22_10seed-90s
SRB_DRIVER=(--partition=default_partition --time=48:00:00 --mem=20G)
SRB90_ARGS=(--ground-truth --black-box --max-evals 1000000000 --timeout 90 --black-box-timeout 90 --seed 10000 --n-runs 10 --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --black-box-max-samples 10000 --pysr-wall-limit 300 --fullsr-wall-limit 600 --black-box-wall-limit 1800 --cpus-per-task 1 --partition default_partition --max-concurrent-jobs 100 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5)
# TODO: on 9/22 only base PySR and PySR++ GT got these flags. Decide whether every row should.
SRB90_SNAPSHOT=(--no-early-stop --no-maxsize-warmup --frontier-snapshot-seconds 10)

# s=$(sbatch --parsable "${SRB_DRIVER[@]}" -J srb90-pysr-base run.sh srbench_full_eval.py "${SRB90_ARGS[@]}" "${SRB90_SNAPSHOT[@]}" --results-dir runs/official-srbench90-pysr-base) || exit 1
# s=$(sbatch --parsable --dependency=afterany:"$s" "${SRB_DRIVER[@]}" -J srb90-basicsr-base  run.sh srbench_full_eval.py --fullsr-baseline "${SRB90_ARGS[@]}" --results-dir runs/official-srbench90-basicsr-base) || exit 1
# s=$(sbatch --parsable --dependency=afterany:"$s" "${SRB_DRIVER[@]}" -J srb90-hpo-gt       run.sh srbench_full_eval.py --hpo-results "$HPO_GT"   "${SRB90_ARGS[@]}" --results-dir runs/official-srbench90-hpo-gt) || exit 1
# s=$(sbatch --parsable --dependency=afterany:"$s" "${SRB_DRIVER[@]}" -J srb90-hpo-gt-r2    run.sh srbench_full_eval.py --hpo-results "$HPO_GTR2" "${SRB90_ARGS[@]}" --results-dir runs/official-srbench90-hpo-gt-r2) || exit 1
# s=$(sbatch --parsable --dependency=afterany:"$s" "${SRB_DRIVER[@]}" -J srb90-hpo-r2       run.sh srbench_full_eval.py --hpo-results "$HPO_R2"   "${SRB90_ARGS[@]}" --results-dir runs/official-srbench90-hpo-r2) || exit 1
# s=$(sbatch --parsable --dependency=afterany:"$s" "${SRB_DRIVER[@]}" -J srb90-pysr-gt      run.sh srbench_full_eval.py --evolve-results "$PYSR_GT"   --select-by val "${SRB90_ARGS[@]}" "${SRB90_SNAPSHOT[@]}" --results-dir runs/official-srbench90-pysr-gt) || exit 1
# s=$(sbatch --parsable --dependency=afterany:"$s" "${SRB_DRIVER[@]}" -J srb90-pysr-gt-r2   run.sh srbench_full_eval.py --evolve-results "$PYSR_GTR2" --select-by val "${SRB90_ARGS[@]}" --results-dir runs/official-srbench90-pysr-gt-r2) || exit 1
# s=$(sbatch --parsable --dependency=afterany:"$s" "${SRB_DRIVER[@]}" -J srb90-pysr-r2      run.sh srbench_full_eval.py --evolve-results "$PYSR_R2"   --select-by val "${SRB90_ARGS[@]}" --results-dir runs/official-srbench90-pysr-r2) || exit 1
# s=$(sbatch --parsable --dependency=afterany:"$s" "${SRB_DRIVER[@]}" -J srb90-basicsr-gt    run.sh srbench_full_eval.py --evolve-results "$BASICSR_GT"   --select-by val   "${SRB90_ARGS[@]}" --results-dir runs/official-srbench90-basicsr-gt) || exit 1
# s=$(sbatch --parsable --dependency=afterany:"$s" "${SRB_DRIVER[@]}" -J srb90-basicsr-gt-r2 run.sh srbench_full_eval.py --evolve-results "$BASICSR_GTR2" --select-by train "${SRB90_ARGS[@]}" --results-dir runs/official-srbench90-basicsr-gt-r2) || exit 1
# s=$(sbatch --parsable --dependency=afterany:"$s" "${SRB_DRIVER[@]}" -J srb90-basicsr-r2    run.sh srbench_full_eval.py --evolve-results "$BASICSR_R2"   --select-by val   "${SRB90_ARGS[@]}" --results-dir runs/official-srbench90-basicsr-r2) || exit 1
# TODO: no summary-table command for the 90 s rows yet. inspect_srbench_results.py --official targets the 1e6-eval protocol.


# (5) SRBench ground truth, 15 min per trial: one long run vs a portfolio of 1e6-eval restarts. Base PySR vs PySR++ GT.
# official: runs/srbench_gt_baseline_15m_single, runs/709715/srbench_gt_15m_single, runs/srbench_gt_baseline_15m_portfolio_1e6, runs/709715/srbench_gt_15m_portfolio_1e6
SRB15_ARGS=(--ground-truth --seed 10000 --n-runs 10 --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --pysr-wall-limit 1200 --partition default_partition --max-concurrent-jobs 100 --time-limit 00:30:00 --job-timeout 7200 --cpus-per-task 1 --mem-per-cpu 8G --max-retries 5 --no-cache)
SRB15_SINGLE=(--max-evals 1000000000 --timeout 900)
SRB15_PORTFOLIO=(--portfolio-time-limit 900 --portfolio-restart-max-evals 1000000)

# s=$(sbatch --parsable -J srb15-base-single run.sh srbench_full_eval.py "${SRB15_ARGS[@]}" "${SRB15_SINGLE[@]}" --results-dir runs/official-srbench15-base-single) || exit 1
# s=$(sbatch --parsable --dependency=afterany:"$s" -J srb15-pysr-gt-single    run.sh srbench_full_eval.py --evolve-results "$PYSR_GT" "${SRB15_ARGS[@]}" "${SRB15_SINGLE[@]}"    --results-dir runs/official-srbench15-pysr-gt-single) || exit 1
# s=$(sbatch --parsable --dependency=afterany:"$s" -J srb15-base-portfolio    run.sh srbench_full_eval.py                            "${SRB15_ARGS[@]}" "${SRB15_PORTFOLIO[@]}" --results-dir runs/official-srbench15-base-portfolio) || exit 1
# s=$(sbatch --parsable --dependency=afterany:"$s" -J srb15-pysr-gt-portfolio run.sh srbench_full_eval.py --evolve-results "$PYSR_GT" "${SRB15_ARGS[@]}" "${SRB15_PORTFOLIO[@]}" --results-dir runs/official-srbench15-pysr-gt-portfolio) || exit 1
# TODO: scripts/analyze_portfolio_solve_over_time.py hard-codes the official portfolio dirs, and its group/array stages were never recorded.
# python figures/plot_portfolio_solve_over_time.py


# (6) Domain evolution of PySR. official: $MIPS_PYSR (MIPS), $NEURON_PYSR (NeuronBench: Z-rebound only, uninformative prompts, built-in 6-world eval)
# TODO: document how outputs/mips_transition_tables was built. outputs/mips_refined_six_artifacts comes from scripts/mips_refined_sr_artifacts.py (submit_jobs.sh, 8/27).
# python scripts/prepare_mips_evolution_overlay.py --split splits/mips_sr_targets_plus_refined.txt --output-root "$MIPS_ROOT"
# sbatch --partition=default_partition --time=2-00:00:00 --cpus-per-task=1 --mem=8G -J mips-evolve --export=ALL,MIPS_TRANSITION_ROOT="$MIPS_ROOT" run.sh evolve_pysr.py --domain mips --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 3 --seed 42 --fitness-metric gt --population-type task --reeval none --identify-topk 0 --exec-feedback-n 3 --exec-feedback-prob 0.5 --models best2 --split splits/mips_sr_targets_plus_refined.txt --val-split "" --final-eval-runs 10 --max-samples 1000 --max-evals 1000000 --timeout 500 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 300 --time-limit 00:15:00 --mem-per-cpu 8G --job-timeout 1800 --no-random-target-noise --output-dir runs/official-mips-pysr
# sbatch -J neuron-evolve run.sh evolve_pysr.py --domain neuron --uninformative-prompts --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models medium2 --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_first1.txt --val-split "" --seed 0 --output-dir runs/official-neuron-pysr


# (7) NeuronBench, all six worlds, 5 seeds, 1e6 evals. official: runs/190178 (base), $NEURON_PYSR/neuron_full_eval (evolved, written by --neuron-full-eval in (6))
NEURON_EVAL_ARGS=(--n-runs 5 --seed 10000 --max-evals 1000000 --max-samples 1024 --partition default_partition --max-concurrent-jobs 30 --time-limit 00:15:00 --mem-per-cpu 8G --timeout 500 --pysr-wall-limit 600 --job-timeout 1800)

# sbatch -J neuron-base run.sh neuron_full_eval.py "${NEURON_EVAL_ARGS[@]}" --output-dir runs/official-neuron-base
# sbatch -J neuron-pysr run.sh neuron_full_eval.py --evolve-results "$NEURON_PYSR" --train-split splits/neuron_first1.txt "${NEURON_EVAL_ARGS[@]}" --output-dir runs/official-neuron-pysr-eval
# TODO: figures/plot_neuronbench_uninformative.py hard-codes runs/190178 and runs/708907.
# python figures/plot_neuronbench_uninformative.py


# (8) MIPS, 51 relations: base PySR, PySR++ GT (with the MIPS operator set), MIPS-evolved PySR.
# official, 1e6 evals x 10 seeds: runs/709714/final_eval_baseline_10seed, runs/709715/final_eval_mips_native_10seed, runs/709714/final_eval_summary.json (built-in final eval, same seed)
# official, 1 h x 3 seeds: runs/709714/final_eval_baseline_3seed_1h, runs/709715/final_eval_mips_native_3seed_1h, runs/709714/final_eval_mips_3seed_1h
MIPS_DRIVER=(--partition=default_partition --time=05:00:00 --cpus-per-task=1 --mem=8G --export=ALL,MIPS_TRANSITION_ROOT="$MIPS_ROOT")
MIPS_EVAL_ARGS=(--domain mips --fitness-metric gt --splits splits/mips_sr_targets_plus_refined.txt --n-runs 10 --seed 192 --max-samples 1000 --max-evals 1000000 --timeout 500 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 300 --time-limit 00:15:00 --mem-per-cpu 8G --job-timeout 1800 --no-cache)
MIPS_1H_ARGS=(--domain mips --fitness-metric gt --splits splits/mips_sr_targets_plus_refined.txt --n-runs 3 --seed 192 --max-samples 1000 --wall-clock-only --timeout 3600 --pysr-wall-limit 3900 --partition default_partition --max-concurrent-jobs 153 --time-limit 01:20:00 --mem-per-cpu 8G --job-timeout 14400 --no-cache)

# sbatch "${MIPS_DRIVER[@]}" -J mips-base    run.sh evaluate_new_pysr.py                                                "${MIPS_EVAL_ARGS[@]}" --output-dir runs/official-mips-eval-base
# sbatch "${MIPS_DRIVER[@]}" -J mips-pysr-gt run.sh evaluate_new_pysr.py --evolve-results "$PYSR_GT" --use-domain-defaults "${MIPS_EVAL_ARGS[@]}" --output-dir runs/official-mips-eval-pysr-gt
# sbatch "${MIPS_DRIVER[@]}" -J mips-mips    run.sh evaluate_new_pysr.py --evolve-results "$MIPS_PYSR"                      "${MIPS_EVAL_ARGS[@]}" --output-dir runs/official-mips-eval-mips-pysr
# m=$(sbatch --parsable "${MIPS_DRIVER[@]}" -J mips-1h-base run.sh evaluate_new_pysr.py "${MIPS_1H_ARGS[@]}" --output-dir runs/official-mips-1h-base) || exit 1
# m=$(sbatch --parsable --dependency=afterany:"$m" "${MIPS_DRIVER[@]}" -J mips-1h-mips    run.sh evaluate_new_pysr.py --evolve-results "$MIPS_PYSR"                      "${MIPS_1H_ARGS[@]}" --output-dir runs/official-mips-1h-mips-pysr) || exit 1
# m=$(sbatch --parsable --dependency=afterany:"$m" "${MIPS_DRIVER[@]}" -J mips-1h-pysr-gt run.sh evaluate_new_pysr.py --evolve-results "$PYSR_GT" --use-domain-defaults "${MIPS_1H_ARGS[@]}" --output-dir runs/official-mips-1h-pysr-gt) || exit 1


# (9a) Reevaluation ablations: 20 generations, 1e6 evals. official: runs/373691 ... runs/373695
ABL20_ARGS=(--operator-type all --generations 20 --population 10 --offspring 10 --models best2)

# a=$(sbatch --parsable -J abl20-n1 run.sh evolve_pysr.py "${ABL20_ARGS[@]}" --population-type task --n-runs 1 --reeval none --output-dir runs/official-abl20-n1) || exit 1
# b=$(sbatch --parsable -J abl20-n3 run.sh evolve_pysr.py "${ABL20_ARGS[@]}" --population-type task --n-runs 3 --reeval none --output-dir runs/official-abl20-n3) || exit 1
# a=$(sbatch --parsable --dependency=afterany:"$a" -J abl20-n10     run.sh evolve_pysr.py "${ABL20_ARGS[@]}" --population-type task --n-runs 10 --reeval none                   --output-dir runs/official-abl20-n10) || exit 1
# b=$(sbatch --parsable --dependency=afterany:"$b" -J abl20-pop1to3 run.sh evolve_pysr.py "${ABL20_ARGS[@]}" --population-type task --n-runs 1 --reeval population --n-reevals 3 --output-dir runs/official-abl20-pop1to3) || exit 1
# sbatch -J abl20-ttts run.sh evolve_pysr.py "${ABL20_ARGS[@]}" --population-type topk --n-runs 1 --reeval TTTS --reeval-budget 20 --output-dir runs/official-abl20-ttts
# TODO: these plots hard-code runs 373691-373695.
# python figures/plot_reevaluation_ablations20.py
# python figures/plot_winners_curse_ablations20.py
# python figures/plot_mean_winners_curse_ablations20.py
# python figures/plot_n1_vs_population_reeval.py


# (9b) Reevaluation ablations: 90 s per fit, seeds 1-3. official: run IDs in figures/plot_90s_reevaluation_ablations.py (four seed-2 runs failed; retries submitted 9/23)
ABL90_ARGS=(--operator-type all --population-type topk --generations 15 --population 10 --offspring 10 --models best2 --max-time-in-seconds 90 --pysr-wall-limit 270 --val-pysr-timeout 90 --val-pysr-wall-limit 270 --population-reeval-runs 3)

# a= b= c=
# for seed in 1 2 3; do
#     a=$(sbatch --parsable ${a:+--dependency=afterany:$a} -J n1-s$seed        run.sh evolve_pysr.py "${ABL90_ARGS[@]}" --n-runs 1 --reeval none                    --seed $seed --output-dir runs/official-abl90-n1-s$seed) || exit 1
#     b=$(sbatch --parsable ${b:+--dependency=afterany:$b} -J n1-reeval-s$seed run.sh evolve_pysr.py "${ABL90_ARGS[@]}" --n-runs 1 --reeval population --n-reevals 3  --seed $seed --output-dir runs/official-abl90-n1-reeval-s$seed) || exit 1
#     c=$(sbatch --parsable ${c:+--dependency=afterany:$c} -J n3-s$seed        run.sh evolve_pysr.py "${ABL90_ARGS[@]}" --n-runs 3 --reeval none                    --seed $seed --output-dir runs/official-abl90-n3-s$seed) || exit 1
#     a=$(sbatch --parsable --dependency=afterany:"$a" -J n3-reeval-s$seed     run.sh evolve_pysr.py "${ABL90_ARGS[@]}" --n-runs 3 --reeval population --n-reevals 10 --seed $seed --output-dir runs/official-abl90-n3-reeval-s$seed) || exit 1
#     b=$(sbatch --parsable --dependency=afterany:"$b" -J n1-ttts-s$seed       run.sh evolve_pysr.py "${ABL90_ARGS[@]}" --n-runs 1 --reeval TTTS --reeval-budget 10     --seed $seed --output-dir runs/official-abl90-n1-ttts-s$seed) || exit 1
#     c=$(sbatch --parsable --dependency=afterany:"$c" -J n3-ttts-s$seed       run.sh evolve_pysr.py "${ABL90_ARGS[@]}" --n-runs 3 --reeval TTTS --reeval-budget 30     --seed $seed --output-dir runs/official-abl90-n3-ttts-s$seed) || exit 1
# done
# TODO: the plot hard-codes run IDs and wandb IDs.
# python figures/plot_90s_reevaluation_ablations.py --refresh


# (9c) Synthetic reevaluation: oracle replay of reevaluation policies on two frozen 10-seed runs. official: runs/568245, runs/568246
# TODO: confirm this is the intended "synthetic reeval" experiment. scripts/oracle_replay.py hard-codes runs 568245/568246.
# sbatch -J oracle-n10-s0 run.sh evolve_pysr.py --operator-type all --population-type topk --generations 15 --population 10 --offspring 20 --n-runs 10 --reeval none --models cheap --random-target-noise --identify-topk 0 --val-split "" --seed 0 --output-dir runs/official-oracle-n10-s0
# sbatch -J oracle-n10-s1 run.sh evolve_pysr.py --operator-type all --population-type topk --generations 15 --population 10 --offspring 20 --n-runs 10 --reeval none --models cheap --random-target-noise --identify-topk 0 --val-split "" --seed 1 --output-dir runs/official-oracle-n10-s1
# python scripts/oracle_replay.py
# python scripts/oracle_replay_table.py
# python figures/plot_reeval_fitness_vs_seeds.py


# (10) EmpiricalBench: base PySR vs PySR++ GT, 10 seeds.
# 90 s per fit. official: runs/empiricalbench_baseline_9-15_10seed_90s_snap5, runs/709715-empiricalbench_9-15_10seed_90s_snap5
EMP90_ARGS=(--n-runs 10 --seed 10000 --timeout 90 --frontier-snapshot-seconds 5 --pysr-wall-limit 300 --no-maxsize-warmup --cpus-per-task 1 --time-limit 00:15:00 --job-timeout 3600 --mem-per-cpu 8G --max-concurrent-jobs 45 --no-cache)

# e1=$(sbatch --parsable --partition=default_partition --time=02:00:00 --mem=20G -J emp90-base    run.sh empbench_full_eval.py                            "${EMP90_ARGS[@]}" --output-dir runs/official-emp90-base) || exit 1
# e2=$(sbatch --parsable --partition=default_partition --time=02:00:00 --mem=20G -J emp90-pysr-gt run.sh empbench_full_eval.py --evolve-results "$PYSR_GT" "${EMP90_ARGS[@]}" --output-dir runs/official-emp90-pysr-gt) || exit 1
# sbatch --dependency=afterok:"$e1":"$e2" --partition=default_partition --cpus-per-task=4 --mem=8G --time=08:00:00 -J emp90-solve-times run.sh scripts/summarize_frontier_snapshot_times.py runs/official-emp90-base runs/official-emp90-pysr-gt --workers 4
# TODO: figures/plot_empiricalbench_snapshot_solve_rate.py hard-codes the 9-15 dirs.
# python figures/plot_empiricalbench_snapshot_solve_rate.py

# 60 min portfolio of 90 s restarts, scored by the paid LLM recovery review (--max-cost is in USD).
# official: runs/empiricalbench_baseline_9-16_10seed_60m_portfolio90s_snap5, runs/709715-empiricalbench_9-16_10seed_60m_portfolio90s_snap5, runs/empiricalbench_9-16_portfolio90s_terra_recovery
EMP60_ARGS=(--n-runs 10 --seed 10000 --timeout 3600 --portfolio-time-limit 3600 --portfolio-restart-timeout 90 --frontier-snapshot-seconds 5 --pysr-wall-limit 3900 --no-maxsize-warmup --cpus-per-task 1 --partition default_partition --time-limit 01:15:00 --job-timeout 10800 --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache)

# e1=$(sbatch --parsable --partition=default_partition --time=04:00:00 --mem=20G -J emp60-base    run.sh empbench_full_eval.py                            "${EMP60_ARGS[@]}" --output-dir runs/official-emp60-base) || exit 1
# e2=$(sbatch --parsable --partition=default_partition --time=04:00:00 --mem=20G -J emp60-pysr-gt run.sh empbench_full_eval.py --evolve-results "$PYSR_GT" "${EMP60_ARGS[@]}" --output-dir runs/official-emp60-pysr-gt) || exit 1
# sbatch --dependency=afterok:"$e1":"$e2" --partition=default_partition --time=08:00:00 --mem=8G -J emp60-review run.sh scripts/empbench_portfolio_recovery.py --baseline runs/official-emp60-base --evolved runs/official-emp60-pysr-gt --output-dir runs/official-emp60-recovery --max-cost 15 --run
# TODO: figures/plot_empiricalbench_portfolio_log_seconds.py hard-codes runs/empiricalbench_9-16_portfolio90s_terra_recovery.
# python figures/plot_empiricalbench_portfolio_log_seconds.py
