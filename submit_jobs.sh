#!/usr/bin/env bash

# 8/25 — Full 62-task MIPS raw-checkpoint reproduction.
# Each processed task gets one CPU and a strict 3600-second child-process cap;
# five hidden_dim > 10 tasks are recorded as upstream-protocol skips. The
# dependent aggregation job runs even if array elements time out or fail.
# Uncomment this block together, then run `bash submit_jobs.sh`.
# MIPS_OUTPUT="outputs/mips_reproduction_all"
# /home/sca63/.conda/envs/meta_sr/bin/python scripts/reproduce_mips_all.py prepare --output-dir "$MIPS_OUTPUT"
# mips_array=$(sbatch --parsable --array=0-61%12 --cpus-per-task=1 --mem=8G --time=01:10:00 --partition=default_partition --job-name=mips-all --output="$MIPS_OUTPUT/slurm/%A_%a.out" --error="$MIPS_OUTPUT/slurm/%A_%a.err" run.sh scripts/reproduce_mips_all.py worker --task-index-env --timeout-seconds 3600 --output-dir "$MIPS_OUTPUT")
# mips_aggregate=$(sbatch --parsable --dependency=afterany:"$mips_array" --cpus-per-task=1 --mem=4G --time=00:15:00 --partition=default_partition --job-name=mips-aggregate --output="$MIPS_OUTPUT/slurm/aggregate_%j.out" --error="$MIPS_OUTPUT/slurm/aggregate_%j.err" run.sh scripts/reproduce_mips_all.py aggregate --output-dir "$MIPS_OUTPUT")
# echo "Submitted MIPS array $mips_array and aggregate $mips_aggregate"

# 8/25
# srbench full evaluation for the 300-trial HPO selections
# chain_a=$(sbatch --parsable -J srb-hpo300-gtr2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260824_190637_506162 --ground-truth --black-box --timeout 0)
# chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J srb-hpo300-r2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260824_183759_524347 --ground-truth --black-box --timeout 0)
# chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J srb-hpo300-gt run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260824_180547_120309 --ground-truth --black-box --timeout 0)

# 8/24
# Counterfactual 300-trial HPO selections. Each driver reuses trials 0-299 and
# the baseline from the corresponding 500-trial run, submits only the 10-way ×
# 10-seed finalist comparison, then automatically runs the standard 10-seed
# final evaluation on barely_unsolvable + val.
# sbatch -J hpo300-gt run.sh hpo_pysr.py --reselect-from outputs/hpo_pysr_20260727_172105_644009 --n-trials 300 --n-runs 3 --n-runs-final 10 --final-topk 10 --n-parallel 20 --split splits/barely_unsolvable.txt --val-split splits/val.txt --fitness-metric gt --random-target-noise
# sbatch -J hpo300-r2 run.sh hpo_pysr.py --reselect-from outputs/hpo_pysr_20260727_172105_644046 --n-trials 300 --n-runs 3 --n-runs-final 10 --final-topk 10 --n-parallel 20 --split splits/barely_unsolvable.txt --val-split splits/val.txt --fitness-metric r2 --random-target-noise
# sbatch -J hpo300-gt-r2 run.sh hpo_pysr.py --reselect-from outputs/hpo_pysr_20260727_172105_644293 --n-trials 300 --n-runs 3 --n-runs-final 10 --final-topk 10 --n-parallel 20 --split splits/barely_unsolvable.txt --val-split splits/val.txt --fitness-metric gt-r2 --random-target-noise

# sbatch -J neuron-eval-313196 run.sh neuron_full_eval.py --evolve-results runs/313196 --output-dir runs/313196/neuron_full_eval --n-runs 5 --seed 10000 --max-evals 1000000 --max-samples 1024 --partition default_partition --time-limit 00:15:00 --mem-per-cpu 8G --timeout 500 --pysr-wall-limit 600 --job-timeout 1800 --train-split splits/neuron_first1.txt --held-out-world h_sag --held-out-world na_fatigue --held-out-world ca_rebound --held-out-world d_type --held-out-world textbook_M
# sbatch -J boolean-eval-313197 run.sh boolean_eval.py --evolve-results runs/313197 --output-dir runs/313197/boolean_eval --n-runs 3 --seed 10000 --max-evals 1000000 --partition default_partition --max-concurrent-jobs 100 --time-limit 01:00:00 --mem-per-cpu 8G --timeout 1800 --pysr-wall-limit 2400 --job-timeout 14400 --train-split splits/boolean_train.txt

# 8/20
# chain_a=$(sbatch --parsable -J srb-hpo-gt run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644009 --black-box --timeout 0)
# chain_a=$(sbatch --parsable -J neuron-top2 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models cheap2 --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_first2.txt --val-split "" --seed 0)
# chain_b=$(sbatch --parsable -J neuron-top1 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models cheap2 --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_first1.txt --val-split "" --seed 0)
# chain_c=$(sbatch --parsable -J logic-evolve run.sh evolve_pysr.py --domain boolean --operator-type all --generations 20 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt-acc --reeval population --n-reevals 10 --models cheap2 --population-type topk --identify-topk 0 --exec-feedback-n 0 --split splits/boolean_train.txt --val-split "" --seed 0)

# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J r2-new-bb run.sh evolve_pysr.py --population-type topk --simplify-cooldown 5 --generations 30 --fitness-metric r2 --split splits/bb_train.txt --n-runs 3 --reeval population --n-reevals 10 --models cheap2 --population 10 --offspring 10)
# chain_b=$(sbatch --dependency=afterany:$chain_b --parsable -J r2-new run.sh evolve_pysr.py --population-type topk --simplify-cooldown 5 --generations 30 --fitness-metric r2 --n-runs 3 --reeval population --n-reevals 10 --models cheap2 --population 10 --offspring 10)
# chain_c=$(sbatch --dependency=afterany:$chain_c --parsable -J r2-gt-new run.sh evolve_pysr.py --population-type topk --simplify-cooldown 5 --generations 30 --fitness-metric r2-gt --n-runs 3 --reeval population --n-reevals 10 --models cheap2 --population 10 --offspring 10)
# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J gt-new run.sh evolve_pysr.py --population-type topk --simplify-cooldown 5 --generations 30 --fitness-metric gt --n-runs 3 --reeval population --n-reevals 10 --models cheap2 --population 10 --offspring 10)

# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J base-1e7-srb run.sh srbench_full_eval.py --black-box --ground-truth --max-evals 10000000)
# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J gt-1e7-srb run.sh srbench_full_eval.py --black-box --ground-truth --max-evals 10000000 --evolve-results runs/538190)
# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J hpo-gt-1e7-srb run.sh srbench_full_eval.py --black-box --ground-truth --max-evals 10000000 --hpo-results outputs/hpo_pysr_20260727_172105_644009)

# 8/19 evening
# sbatch -J logic-evolve run.sh evolve_pysr.py --domain boolean --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt-acc --reeval population --n-reevals 10 --models best --population-type topk --identify-topk 0 --exec-feedback-n 0 --split splits/boolean_train.txt --val-split "" --seed 0
# sbatch -J neuron-loocv1 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models best --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_loocv1.txt --val-split "" --seed 0
# sbatch -J neuron-top1 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models best --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_first1.txt --val-split "" --seed 0
# sbatch -J neuron-top2 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models best --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_first2.txt --val-split "" --seed 0
# sbatch --dependency=afterany:290163 -J pysr-base2 run.sh srbench_full_eval.py --ground-truth --black-box --no-cache
# sbatch --dependency=afterany:290211 -J r2-new run.sh evolve_pysr.py --population-type topk --simplify-cooldown 5 --generations 30 --fitness-metric r2 --split splits/bb_train.txt --n-runs 3 --reeval population --n-reevals 10 --models best --population 10 --offspring 10

# 8/19
# sbatch -J simp-538 run.sh evolve_pysr.py --operator-type all --mutation-mode simplify --population-type complexity --generations 10 --population 10 --offspring 10 --n-runs 3 --models best --continue-from runs/538190 --reeval population --n-reevals 10
# srb_prev=$(sbatch --dependency=afterany:252289 --parsable -J neuron-loocv1 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models best --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_loocv1.txt --val-split "" --seed 0)
# bool_prev=$(sbatch --dependency=afterany:$srb_prev --parsable -J logic-evolve run.sh evolve_pysr.py --domain boolean --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt-acc --reeval population --n-reevals 10 --models best --population-type topk --identify-topk 0 --exec-feedback-n 0 --split splits/boolean_train.txt --val-split "" --seed 0)

# try doing R2 evolution

# SRBench full evaluation for the completed FullSR run (ground truth + black box).
# sbatch --parsable -J srb-full-150812 run.sh srbench_full_eval.py --evolve-results runs/150812 --ground-truth --black-box
# sbatch --parsable -J srb-full-150812 run.sh srbench_full_eval.py --evolve-results runs/150815 --ground-truth --black-box

# 8/18
# sbatch -J simp-538 run.sh evolve_pysr.py --operator-type all --mutation-mode simplify --population-type complexity --generations 10 --population 10 --offspring 10 --n-runs 3 --models best --continue-from runs/538190 --reeval population --n-reevals 10

# NeuronBench LOOCV: each evolution job trains on five worlds and automatically
# evaluates its final bundle on all six worlds (5 fresh seeds, 1e6 evals).
# srb_prev=$(sbatch --parsable -J neuron-loocv1 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 10 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models best --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --partition default_partition --max-concurrent-jobs 100 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_loocv1.txt --val-split "" --seed 0)

# Base-PySR control: same all-world 5-seed, 1e6-eval evaluation protocol.
# sbatch --dependency=afterany:"$srb_prev" -J neuron-baseline run.sh neuron_full_eval.py --n-runs 5 --seed 10000 --max-evals 1000000 --max-samples 1024 --partition default_partition --max-concurrent-jobs 100

# LogicBench (Boolean synthesis):
# bool_prev=$(sbatch --dependency=afterany:190177 --parsable -J logic-evolve run.sh evolve_pysr.py --domain boolean --operator-type all --generations 10 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt-acc --reeval population --n-reevals 10 --models best --population-type topk --identify-topk 0 --exec-feedback-n 0 --partition default_partition --max-concurrent-jobs 100 --split splits/boolean_train.txt --val-split "" --seed 0)

# Base-PySR control on the 100 held-out IWLS 2020 problems (3 seeds, 1e6 evals).
# sbatch --dependency=afterany:"$bool_prev" -J logic-baseline run.sh boolean_eval.py --n-runs 3 --seed 10000 --max-evals 1000000 --partition default_partition --max-concurrent-jobs 100

# Two independent chains, so at most two of these run at once. Each job depends
# on the last job of its OWN chain -- chain_a and chain_b never reference each
# other, or they collapse into a single serial chain.
#
# chain_a leads with the three srb-full-* cache top-offs (0 / 104 / 4 runs left,
# so minutes each) to get an HPO grid started early; chain_b carries the two long
# evolve_fullsr runs. The HPO evals are cold 6540-run grids -- --timeout 0 keeps
# them on the same no-soft-timeout protocol as the PySR rows they are compared
# against, and so guarantees full cache misses.
# chain_a=$(sbatch --parsable -J srb-full-225437 run.sh srbench_full_eval.py --evolve-results runs/225437 --ground-truth --black-box)
# chain_b=$(sbatch --parsable -J full-r2 run.sh evolve_fullsr.py --generations 30 --offspring 10 --population 10 --n-runs 3 --models best --fitness-metric r2 --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt)

# chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J srb-full-548741 run.sh srbench_full_eval.py --evolve-results runs/548741 --ground-truth --black-box)
# chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J srb-full-base run.sh srbench_full_eval.py --fullsr-baseline --ground-truth --black-box)
# chain_b=$(sbatch --parsable --dependency=afterany:"$chain_b" -J full-gt-r2 run.sh evolve_fullsr.py --generations 30 --offspring 10 --population 10 --n-runs 3 --models best --fitness-metric gt-r2 --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt)

# chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J srb-hpo-gt run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644009 --ground-truth --black-box --timeout 0)
# chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J srb-hpo-r2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644046 --ground-truth --black-box --timeout 0)
# chain_b=$(sbatch --parsable --dependency=afterany:"$chain_b" -J srb-hpo-gtr2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644293 --ground-truth --black-box --timeout 0)

# 8/17
# evolve_full gt-r2
# sbatch -J full-gt-r2 run.sh evolve_fullsr.py --generations 30 --offspring 10 --population 10 --n-runs 3 --models best --fitness-metric gt-r2 --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt

# srb_prev=$(sbatch --parsable -J srb-full-base run.sh srbench_full_eval.py --fullsr-baseline --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-full-225437 run.sh srbench_full_eval.py --evolve-results runs/225437 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-full-548741 run.sh srbench_full_eval.py --evolve-results runs/548741 --ground-truth --black-box)


# 7/31 — BasicSR baseline on full SRBench (driver submits chunked arrays).
# Superseded by the 8/17 block above: these ran with no soft timeout.
# srb_prev=$(sbatch --parsable -J srb-full-base run.sh srbench_full_eval.py --fullsr-baseline --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-full-225437 run.sh srbench_full_eval.py --evolve-results runs/225437 --ground-truth --black-box)

# 7/29

# # Seed the shared cache with successful black-box artifacts from 7/28. Failed
# # trials stay uncached and are the only black-box trials rerun below.
# python scripts/import_srbench_black_box_cache.py \
#     runs/548743 runs/548744 runs/548745 runs/548746

# srb_prev=$(sbatch --parsable -J srb-gtr2 run.sh srbench_full_eval.py --evolve-results runs/120459 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-r2 run.sh srbench_full_eval.py --evolve-results runs/120458 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-gt run.sh srbench_full_eval.py --evolve-results runs/538190 --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-base run.sh srbench_full_eval.py --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-hpo-gt run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644009 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-hpo-r2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644046 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-hpo-gtr2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644293 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-full-225437 run.sh srbench_full_eval.py --evolve-results runs/225437 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-full-base run.sh srbench_full_eval.py --fullsr-baseline --ground-truth --black-box)

# 1e7 gt evals
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-gt-1e7 run.sh srbench_full_eval.py --evolve-results runs/538190 --ground-truth --max-evals 10000000)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-base-1e7 run.sh srbench_full_eval.py --ground-truth --max-evals 10000000)

# simplify
# sbatch -J simp-538 run.sh evolve_pysr.py --operator-type all --mutation-mode simplify --population-type complexity --generations 10 --population 10 --offspring 10 --n-runs 3 --models best --continue-from runs/538190 --reeval population --n-reevals 10

# 7/28/36

# sbatch -J full-r2 run.sh evolve_fullsr.py --generations 30 --offspring 10 --population 10 --n-runs 3 --models best --fitness-metric r2 --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt
# sbatch -J full-gt-r2 run.sh evolve_fullsr.py --generations 30 --offspring 10 --population 10 --n-runs 3 --models best --fitness-metric gt-r2 --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt

# srb_prev=$(sbatch --parsable -J srb-gtr2 run.sh srbench_full_eval.py --evolve-results runs/120459 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-r2 run.sh srbench_full_eval.py --evolve-results runs/120458 --black-box --ground-truth)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-gt run.sh srbench_full_eval.py --evolve-results runs/538190 --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-base run.sh srbench_full_eval.py --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-hpo-gt run.sh srbench_full_eval.py --evolve-results runs/422103 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-hpo-r2 run.sh srbench_full_eval.py --evolve-results runs/422104 --black-box --ground-truth)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-hpo-gtr2 run.sh srbench_full_eval.py --evolve-results runs/422105 --black-box --ground-truth)

# sbatch -J eval-121270 --partition=default_partition run.sh evaluate_new_pysr.py --splits splits/val.txt splits/barely_unsolvable_val2.txt --n-runs 10 --evolve-results runs/121270
# sbatch -J eval-155134 --partition=default_partition run.sh evaluate_new_pysr.py --splits splits/val.txt splits/barely_unsolvable_val2.txt --n-runs 10 --evolve-results runs/155134
# sbatch -J eval-120458 --partition=default_partition run.sh evaluate_new_pysr.py --splits splits/val.txt splits/barely_unsolvable_val2.txt --n-runs 10 --evolve-results runs/120458
# sbatch -J r2-train run.sh evolve_pysr.py --split splits/train.txt --val-split splits/val.txt --generations 30 --models best --offspring 10 --population 10 --n-runs 3 --fitness-metric r2

# set -euo pipefail

# 7/27/26 resubmitting jobs that failed from 7/23
# sbatch -J hpo-gt run.sh hpo_pysr.py --n-trials 500 --n-runs 3 --split splits/barely_unsolvable.txt --fitness-metric gt --n-parallel 20 --random-target-noise --models best
# sbatch -J hpo-r2 run.sh hpo_pysr.py --n-trials 500 --n-runs 3 --split splits/barely_unsolvable.txt --fitness-metric r2 --n-parallel 20 --random-target-noise --models best
# sbatch -J hpo-gt-r2 run.sh hpo_pysr.py --n-trials 500 --n-runs 3 --split splits/barely_unsolvable.txt --fitness-metric gt-r2 --n-parallel 20 --random-target-noise --models best

# 7/23/26 — queued experiment suite. Three independent dependency chains keep
# at most three driver jobs running at once. Independent jobs use `afterany` so
# a chain continues after an earlier failure. Follow-ups that consume a newly
# created run use `afterok`, because they require the producer's run_data.json.
# Run this file to submit; these commands have not been submitted yet.

# # Chain 1, job 1: evaluate base PySR on the 122 SRBench black-box (BB) tasks.
# chain1=$(sbatch --parsable -J bb-pysr-base run.sh srbench_full_eval.py --black-box)
# # Chain 2, job 1: run 500-trial PySR HPO on GT with three runs per trial.
# chain2=$(sbatch --parsable -J hpo-gt run.sh hpo_pysr.py --n-trials 500 --n-runs 3 --split splits/barely_unsolvable.txt --fitness-metric gt --n-parallel 20)
# # Chain 3, job 1: run 500-trial PySR HPO on R² with three runs per trial.
# chain3=$(sbatch --parsable -J hpo-r2 run.sh hpo_pysr.py --n-trials 500 --n-runs 3 --split splits/barely_unsolvable.txt --fitness-metric r2 --n-parallel 20)

# # Chain 1, job 2: evaluate GT-R² PySR++ (run 120459) on BB.
# chain1=$(sbatch --parsable --dependency=afterany:$chain1 -J bb-pysrpp-gtr2 run.sh srbench_full_eval.py --evolve-results runs/120459 --black-box)
# # Chain 2, job 2: evaluate R² PySR++ (run 120458) on BB.
# chain2=$(sbatch --parsable --dependency=afterany:$chain2 -J bb-pysrpp-r2 run.sh srbench_full_eval.py --evolve-results runs/120458 --black-box)
# # Chain 3, job 2: run 500-trial PySR HPO on GT-R² with three runs per trial.
# chain3=$(sbatch --parsable --dependency=afterany:$chain3 -J hpo-gt-r2 run.sh hpo_pysr.py --n-trials 500 --n-runs 3 --split splits/barely_unsolvable.txt --fitness-metric gt-r2 --n-parallel 20)

# # Chain 1, job 3: evolve FullSR for R² (cheap models, 15 generations).
# fullsr_r2=$(sbatch --parsable --dependency=afterany:$chain1 -J fullsr-r2 run.sh evolve_fullsr.py --operator-type all --generations 15 --population 10 --offspring 10 --n-runs 3 --models cheap --fitness-metric r2 --split splits/train.txt --val-split splits/val.txt)
# # Chain 2, job 3: evaluate GT-R² PySR++ (run 120459) on all GT tasks.
# chain2=$(sbatch --parsable --dependency=afterany:$chain2 -J gt-pysrpp-gtr2 run.sh srbench_full_eval.py --evolve-results runs/120459 --ground-truth)
# # Chain 3, job 3: evolve all PySR++ operators on LogicBench (cheap models, 15 generations).
# chain3=$(sbatch --parsable --dependency=afterany:$chain3 -J logic-pysrpp run.sh evolve_pysr.py --domain boolean --operator-type all --generations 15 --population 10 --offspring 10 --n-runs 3 --models cheap)

# # Chain 1, job 4: evaluate the newly evolved R² FullSR run on BB.
# chain1=$(sbatch --parsable --dependency=afterok:$fullsr_r2 -J bb-fullsr-r2 run.sh srbench_full_eval.py --evolve-results runs/$fullsr_r2 --black-box)
# # Chain 2, job 4: evolve FullSR for GT-R² (cheap models, 15 generations).
# fullsr_gtr2=$(sbatch --parsable --dependency=afterany:$chain2 -J fullsr-gt-r2 run.sh evolve_fullsr.py --operator-type all --generations 15 --population 10 --offspring 10 --n-runs 3 --models cheap --fitness-metric gt-r2 --split splits/train.txt --val-split splits/val.txt)

# # Chain 2, job 5: evaluate the newly evolved GT-R² FullSR run on BB.
# chain2=$(sbatch --parsable --dependency=afterok:$fullsr_gtr2 -J bb-fullsr-gtr2 run.sh srbench_full_eval.py --evolve-results runs/$fullsr_gtr2 --black-box)

# echo "Submitted three chains; tails: chain1=$chain1 chain2=$chain2 chain3=$chain3"

# 7/23/26
# sbatch -J fulleval --partition default_partition run.sh srbench_full_eval.py --evolve-results runs/538190
# sbatch run.sh evolve_fullsr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --models best --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt --random-target-noise
# sbatch -J simplify run.sh evolve_pysr.py --operator-type all --generations 10 --population 10 --offspring 10 --n-runs 3 --models best --mutation-mode simplify --population-type complexity --continue-from runs/538190

# 7/22/26 - minimalSR evolution
# sbatch run.sh evolve_fullsr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --models best --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt

# sbatch run.sh evolve_fullsr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --models best --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt
# 7/21/26 — n1 vs n3 vs reeval modes (best models, 10 offspring/gen, 30 gens).
# Oracle-replay follow-up under the NEW reeval CLI (--reeval + --reeval-budget):
#   n1  = 10 evals/gen, no reeval
#   n3  = 30 evals/gen, no reeval
#   un1 = n1 offspring + '--reeval uniform' B=20 (even split over top-10)
#         = 30/gen, budget-matched to n3 (the literal oracle-replay winner)
#   un3 = n3 offspring + '--reeval uniform' B=40 = 70/gen
# 3 jobs run at once: seed-0s start immediately; each seed-1 chains afterany its
# own seed-0; the un3 pair chains after the first six wind down.
# To launch: run `bash submit_jobs.sh` (this block is the only uncommented one).
# COMMON="--operator-type all --generations 30 --population 10 --offspring 10 --models best --random-target-noise"
# N1="--n-runs 1 --reeval none"
# N3="--n-runs 3 --reeval none"
# UN1="--n-runs 1 --reeval uniform --reeval-budget 20"
# UN3="--n-runs 3 --reeval uniform --reeval-budget 40"
# # seed 0 (3 concurrent)
# jn1s0=$(sbatch --parsable -J n1s0      run.sh evolve_pysr.py $COMMON $N1  --seed 0)
# jn3s0=$(sbatch --parsable -J n3s0      run.sh evolve_pysr.py $COMMON $N3  --seed 0)
# jun1s0=$(sbatch --parsable -J u-n1s0 run.sh evolve_pysr.py $COMMON $UN1 --seed 0)
# # seed 1 (each after its own seed 0)
# jn1s1=$(sbatch --parsable --dependency=afterany:$jn1s0 -J n1s1   run.sh evolve_pysr.py $COMMON $N1  --seed 1)
# jn3s1=$(sbatch --parsable --dependency=afterany:$jn3s0 -J n3s1   run.sh evolve_pysr.py $COMMON $N3  --seed 1)
# jun1s1=$(sbatch --parsable --dependency=afterany:$jun1s0 -J u-n1s1 run.sh evolve_pysr.py $COMMON $UN1 --seed 1)
# # n3 + uniform-B follow-up pair (after the first six wind down)
# jun3s0=$(sbatch --parsable --dependency=afterany:$jn3s1 -J u-n3s0 run.sh evolve_pysr.py $COMMON $UN3 --seed 0)
# jun3s1=$(sbatch --parsable --dependency=afterany:$jun1s1 -J u-n3s1 run.sh evolve_pysr.py $COMMON $UN3 --seed 1)

# echo "Submitted 8 jobs: n1=($jn1s0,$jn1s1) n3=($jn3s0,$jn3s1) un1=($jun1s0,$jun1s1) un3=($jun3s0,$jun3s1)"


# 7/16/26

# n1 vs v3, continue to 30 generations
# cheap ensemble, --random-target-noise). Per-generation eval budget B matched per pair:
#   cond1 none  n1o20 (20 evals/gen)  vs  cond3 smart n1o5  B=20   ← chain A (odd jobs)
#   cond2 none  n3o20 (60 evals/gen)  vs  cond4 smart n3o5  B=60   ← chain B (even jobs)

# At most 2 jobs run at once: jobs 1 & 2 start immediately; job N depends on job N-2
# (afterany), forming two parallel chains. Jobs are ordered seed-major (all 4 conditions
# at seed s, then s+1), which makes chain A = the B=20 pair and chain B = the B=60 pair.
# To launch: uncomment this whole block (it must run together — the $jidN vars chain
# the dependencies), then run `bash submit_jobs.sh`.
# COMMON="--operator-type all --generations 30 --population 10 --models cheap --random-target-noise"
# C1="--offspring 20 --n-runs 1 --reeval none"
# C2="--offspring 20 --n-runs 3 --reeval none"
# #
# # seed 0
# jid1=$(sbatch --parsable                       -J nn_n1o20_s0   run.sh evolve_pysr.py $COMMON $C1 --seed 0 --continue-from runs/89281)
# jid2=$(sbatch --parsable                       -J nn_n3o20_s0   run.sh evolve_pysr.py $COMMON $C2 --seed 0 --continue-from runs/89282)
# # seed 1
# jid5=$(sbatch --parsable --dependency=afterany:$jid1 -J nn_n1o20_s1   run.sh evolve_pysr.py $COMMON $C1 --seed 1 --continue-from runs/825769)
# jid6=$(sbatch --parsable --dependency=afterany:$jid2 -J nn_n3o20_s1   run.sh evolve_pysr.py $COMMON $C2 --seed 1 --continue-from runs/825770)
# # seed 2
# jid9=$(sbatch  --parsable --dependency=afterany:$jid5 -J nn_n1o20_s2   run.sh evolve_pysr.py $COMMON $C1 --seed 2 --continue-from runs/825773)
# jid10=$(sbatch --parsable --dependency=afterany:$jid6 -J nn_n3o20_s2   run.sh evolve_pysr.py $COMMON $C2 --seed 2 --continue-from runs/825774)
# # seed 3
# jid13=$(sbatch --parsable --dependency=afterany:$jid9 -J nn_n1o20_s3   run.sh evolve_pysr.py $COMMON $C1 --seed 3 --continue-from runs/825777)
# jid14=$(sbatch --parsable --dependency=afterany:$jid10 -J nn_n3o20_s3   run.sh evolve_pysr.py $COMMON $C2 --seed 3 --continue-from runs/825778)
# # seed 4
# jid17=$(sbatch --parsable --dependency=afterany:$jid13 -J nn_n1o20_s4   run.sh evolve_pysr.py $COMMON $C1 --seed 4 --continue-from runs/825781)
# jid18=$(sbatch --parsable --dependency=afterany:$jid14 -J nn_n3o20_s4   run.sh evolve_pysr.py $COMMON $C2 --seed 4 --continue-from runs/825782)
# echo "Submitted 10 jobs"

# 7/1/26
# COMMON="--generations 20 --population 10 --models best --random-target-noise"
# C1="--offspring 10 --n-runs 10 --reeval none"
# sbatch -J bn10o10s0 run.sh evolve_pysr.py $COMMON $C1 --seed 0
# sbatch -J bn10o10s1 run.sh evolve_pysr.py $COMMON $C1 --seed 1

# sbatch -J fullsr --partition ellis run.sh evolve_fullsr.py --operator-type all --generations 15 --population 10 --offspring 10 --n-runs 3 --models cheap --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt
# sbatch --dependency=afterany:568245 -J fullsr-full-file --partition ellis run.sh evolve_fullsr.py --operator-type all --generations 15 --population 10 --offspring 10 --n-runs 3 --models cheap --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt --full-file-diff
# sbatch --dependency=afterany:568246 -J max-time run.sh evolve_pysr.py --generations 15 --population 10 --n-runs 3 --max-time-in-seconds 120 --models best --random-target-noise --reeval smart
# sbatch --dependency=afterany:689916 -J gt-r2-train run.sh evolve_pysr.py --generations 15 --population 10 --n-runs 3 --models cheap --reeval smart --random-target-noise --fitness-metric gt-r2 --split splits/train.txt --val-split splits/val.txt

# C2="--offspring 20 --n-runs 3 --reeval none"
# C3="--offspring 5 --n-runs 1 --reeval smart --max-runs-per-generation 20"
# C4="--offspring 5 --n-runs 3 --reeval smart --max-runs-per-generation 60"
# #
# # seed 0
# jid1=$(sbatch --parsable                       -J nn_n1o20_s0   run.sh evolve_pysr.py $COMMON $C1 --seed 0)
# jid2=$(sbatch --parsable                       -J nn_n3o20_s0   run.sh evolve_pysr.py $COMMON $C2 --seed 0)
# jid3=$(sbatch --parsable --dependency=afterany:$jid1 -J sm_n1o5b20_s0 run.sh evolve_pysr.py $COMMON $C3 --seed 0)
# jid4=$(sbatch --parsable --dependency=afterany:$jid2 -J sm_n3o5b60_s0 run.sh evolve_pysr.py $COMMON $C4 --seed 0)
# 6/30/26
# sbatch -J fullsr --partition ellis run.sh evolve_fullsr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --models best --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt
# sbatch --dependency=afterany:397152 -J max-time run.sh evolve_pysr.py --generations 30 --population 10 --n-runs 3 --max-time-in-seconds 120 --models best --random-target-noise --reeval smart
# sbatch --dependency=afterany:492224 -J fullsr --partition ellis run.sh evolve_fullsr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --models best --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt --full-file-diff
# sbatch --dependency=afterany:492225 -J r2-train run.sh evolve_pysr.py --generations 15 --population 10 --n-runs 3 --models best --reeval smart --random-target-noise --fitness-metric r2 --split splits/train.txt --val-split splits/val.txt

# 6/29/26 — planet eval of the gt-r2 bundle (runs/120459).
# sbatch -J r2-gt-planet planet_eval.sh --evolve-results ~/meta_sr/runs/120459

# 6/29/26 — smart-reeval comparison on the BEST model ensemble.
# COMMON="--operator-type all --generations 30 --population 10 --n-runs 1 --models best --random-target-noise"
# NONE="--offspring 20 --reeval none"
# KG="--offspring 5 --reeval smart-KG --max-runs-per-generation 20"
# TT="--offspring 5 --reeval smart-TTTS --max-runs-per-generation 20"
# sbatch --parsable  -J re_ttts_s0 run.sh evolve_pysr.py $COMMON $TT   --seed 0
# sbatch --parsable  -J re_ttts_s1 run.sh evolve_pysr.py $COMMON $TT   --seed 1
# # # seed 0 (chain A)
# # a1=$(sbatch --parsable                            -J re_none_s0 run.sh evolve_pysr.py $COMMON $NONE --seed 0)
# # a2=$(sbatch --parsable                            -J re_kg_s0   run.sh evolve_pysr.py $COMMON $KG   --seed 0)
# # # seed 1 (chain B)
# # b1=$(sbatch --parsable                            -J re_none_s1 run.sh evolve_pysr.py $COMMON $NONE --seed 1)
# # b2=$(sbatch --parsable                            -J re_kg_s1   run.sh evolve_pysr.py $COMMON $KG   --seed 1)


# all noise evolution
# sbatch --dependency=afterany:$c1 -J all-noise --partition ellis run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --reeval none --models best --eval-all-noise-levels

# new fullSR evolution
# sbatch -J fullsr --partition ellis run.sh evolve_fullsr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --models cheap --split splits/train.txt


# 6/26/26
# evolve R^2 or R^2 combined with GT
# sbatch -J r2 run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 5 --n-runs 3 --models best --reeval smart --max-runs-per-generation 60 --random-target-noise --fitness-metric r2
# sbatch -J gt-r2 run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 5 --n-runs 3 --models best --reeval smart --max-runs-per-generation 60 --random-target-noise --fitness-metric gt-r2
# sbatch --dependency=afterany:120458 -J train run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 5 --n-runs 3 --models best --reeval smart --max-runs-per-generation 60 --random-target-noise --split splits/train.txt --val-split splits/val.txt

# 6/26/26 — resubmit the two corrupted seed-0 "no reeval" runs. The originals
# (825765 nn_n1o20_s0 and 825766 nn_n3o20_s0, the first two jobs submitted) died
# at gen 2: code on disk was edited mid-run, so their long-running driver hit
# `PySRTaskResult.__init__() got an unexpected keyword argument 'r2_frontier_score'`
# and every offspring scored -1 thereafter (frozen population). Chained after each
# pair's seed-4 job (825783 = chain A / n1 last, 825784 = chain B / n3 last) so we
# keep at most 2 running. After they finish, point the plot GROUPS at the new ids.
# SUBMITTED 6/26: nn_n1o20_s0 -> job 89281 (after 825783); nn_n3o20_s0 -> job 89282 (after 825784).
# COMMON="--operator-type all --generations 15 --population 10 --models cheap --random-target-noise"
# reA=$(sbatch --parsable --dependency=afterany:825783 -J nn_n1o20_s0_re run.sh evolve_pysr.py $COMMON --offspring 20 --n-runs 1 --reeval none --seed 0)
# reB=$(sbatch --parsable --dependency=afterany:825784 -J nn_n3o20_s0_re run.sh evolve_pysr.py $COMMON --offspring 20 --n-runs 3 --reeval none --seed 0)
# echo "resubmitted: nn_n1o20_s0 -> $reA (after 825783),  nn_n3o20_s0 -> $reB (after 825784)"

# 6/24/26 — smart vs nonsmart budget-matched comparison (5 seeds each, 15 gens,
# cheap ensemble, --random-target-noise). Per-generation eval budget B matched per pair:
#   cond1 none  n1o20 (20 evals/gen)  vs  cond3 smart n1o5  B=20   ← chain A (odd jobs)
#   cond2 none  n3o20 (60 evals/gen)  vs  cond4 smart n3o5  B=60   ← chain B (even jobs)
# At most 2 jobs run at once: jobs 1 & 2 start immediately; job N depends on job N-2
# (afterany), forming two parallel chains. Jobs are ordered seed-major (all 4 conditions
# at seed s, then s+1), which makes chain A = the B=20 pair and chain B = the B=60 pair.
# To launch: uncomment this whole block (it must run together — the $jidN vars chain
# the dependencies), then run `bash submit_jobs.sh`.
# COMMON="--operator-type all --generations 15 --population 10 --models cheap --random-target-noise"
# C1="--offspring 20 --n-runs 1 --reeval none"
# C2="--offspring 20 --n-runs 3 --reeval none"
# C3="--offspring 5 --n-runs 1 --reeval smart --max-runs-per-generation 20"
# C4="--offspring 5 --n-runs 3 --reeval smart --max-runs-per-generation 60"
# #
# # seed 0
# jid1=$(sbatch --parsable                       -J nn_n1o20_s0   run.sh evolve_pysr.py $COMMON $C1 --seed 0)
# jid2=$(sbatch --parsable                       -J nn_n3o20_s0   run.sh evolve_pysr.py $COMMON $C2 --seed 0)
# jid3=$(sbatch --parsable --dependency=afterany:$jid1 -J sm_n1o5b20_s0 run.sh evolve_pysr.py $COMMON $C3 --seed 0)
# jid4=$(sbatch --parsable --dependency=afterany:$jid2 -J sm_n3o5b60_s0 run.sh evolve_pysr.py $COMMON $C4 --seed 0)
# # seed 1
# jid5=$(sbatch --parsable --dependency=afterany:$jid3 -J nn_n1o20_s1   run.sh evolve_pysr.py $COMMON $C1 --seed 1)
# jid6=$(sbatch --parsable --dependency=afterany:$jid4 -J nn_n3o20_s1   run.sh evolve_pysr.py $COMMON $C2 --seed 1)
# jid7=$(sbatch --parsable --dependency=afterany:$jid5 -J sm_n1o5b20_s1 run.sh evolve_pysr.py $COMMON $C3 --seed 1)
# jid8=$(sbatch --parsable --dependency=afterany:$jid6 -J sm_n3o5b60_s1 run.sh evolve_pysr.py $COMMON $C4 --seed 1)
# # seed 2
# jid9=$(sbatch  --parsable --dependency=afterany:$jid7 -J nn_n1o20_s2   run.sh evolve_pysr.py $COMMON $C1 --seed 2)
# jid10=$(sbatch --parsable --dependency=afterany:$jid8 -J nn_n3o20_s2   run.sh evolve_pysr.py $COMMON $C2 --seed 2)
# jid11=$(sbatch --parsable --dependency=afterany:$jid9 -J sm_n1o5b20_s2 run.sh evolve_pysr.py $COMMON $C3 --seed 2)
# jid12=$(sbatch --parsable --dependency=afterany:$jid10 -J sm_n3o5b60_s2 run.sh evolve_pysr.py $COMMON $C4 --seed 2)
# # seed 3
# jid13=$(sbatch --parsable --dependency=afterany:$jid11 -J nn_n1o20_s3   run.sh evolve_pysr.py $COMMON $C1 --seed 3)
# jid14=$(sbatch --parsable --dependency=afterany:$jid12 -J nn_n3o20_s3   run.sh evolve_pysr.py $COMMON $C2 --seed 3)
# jid15=$(sbatch --parsable --dependency=afterany:$jid13 -J sm_n1o5b20_s3 run.sh evolve_pysr.py $COMMON $C3 --seed 3)
# jid16=$(sbatch --parsable --dependency=afterany:$jid14 -J sm_n3o5b60_s3 run.sh evolve_pysr.py $COMMON $C4 --seed 3)
# # seed 4
# jid17=$(sbatch --parsable --dependency=afterany:$jid15 -J nn_n1o20_s4   run.sh evolve_pysr.py $COMMON $C1 --seed 4)
# jid18=$(sbatch --parsable --dependency=afterany:$jid16 -J nn_n3o20_s4   run.sh evolve_pysr.py $COMMON $C2 --seed 4)
# jid19=$(sbatch --parsable --dependency=afterany:$jid17 -J sm_n1o5b20_s4 run.sh evolve_pysr.py $COMMON $C3 --seed 4)
# jid20=$(sbatch --parsable --dependency=afterany:$jid18 -J sm_n3o5b60_s4 run.sh evolve_pysr.py $COMMON $C4 --seed 4)
# echo "Submitted 20 jobs"

# 6/24/26 — fullsr evolution sanity check (does evolving improve over baseline?)
# sbatch -J fullsr --partition ellis run.sh evolve_fullsr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --models cheap --split splits/train.txt

# 6/23/26

# sbatch -J train-split-n3o10smart-best run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models best --reeval smart --max-runs-per-generation 200 --random-target-noise --split splits/train.txt

# sbatch -J test_full --partition default_partition run.sh evolve_fullsr.py --operator-type all --generations 10 --population 10 --offspring 10 --n-runs 3 --models best --split splits/train.txt
# sbatch -J fulleval --partition default_partition run.sh srbench_full_eval.py --evolve-results runs/538190
# sbatch -J planetseval2 --partition default_partition planet_eval.sh --evolve-results runs/538190
# sbatch -J planetsbaseline2 --partition default_partition planet_eval.sh --baseline

# ##### SMART, N3O10 OR N1O10, S0 OR S1
# jid1=$(sbatch --parsable -J n3o10s0smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 3 --models best --reeval smart --max-reruns 100 --random-target-noise --seed 0)
# jid2=$(sbatch --parsable -J n3o10s1smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 3 --models best --reeval smart --max-reruns 100 --random-target-noise --seed 1)
# jid3=$(sbatch --parsable -J n1o10s0smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --models cheap --reeval smart --max-reruns 100 --random-target-noise --seed 0)
# jid4=$(sbatch --parsable -J n1o10s1smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --models cheap --reeval smart --max-reruns 100 --random-target-noise --seed 1)

# ##### NONSMART, N3O10 OR N1O10, S0 OR S1
# sbatch -J n3o10s0 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 3 --models best --reeval none --random-target-noise --seed 0
# sbatch -J n3o10s1 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 3 --models best --reeval none --random-target-noise --seed 1
# sbatch --dependency=afterany:$jid3 -J n1o10s0 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --models cheap --reeval none --random-target-noise --seed 0
# sbatch --dependency=afterany:$jid4 -J n1o10s1 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --models cheap --reeval none --random-target-noise --seed 1

# 6/18
# sbatch -J n3o10smart-cheap run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models best --reeval smart --max-reruns 100 --random-target-noise
# sbatch -J n3o10smart-best run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models cheap --reeval smart --max-reruns 100 --random-target-noise

# 6/17 planet_eqs eval (planet_eval.py submits its own 1-node / 32-core Slurm job)
# sbatch -J planets_baseline planet_eval.sh planet_eval.py --baseline --time-in-hours 8
# sbatch -J planets40318 planet_eval.sh planet_eval.py --evolve-results ~/meta_sr/runs/40318

# sbatch -J fulleval_666285 run.sh srbench_full_eval.py --evolve-results runs/666285 --max-evals 1000000

# 6/4
# sbatch -J n2o10 run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 14 --n-runs 2 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n2o10smart run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 14 --n-runs 2 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100
# sbatch --dependency=afterany:40319 -J n1o10smart_cont run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100 --continue-from runs/40319

# 6/3
# sbatch -J n1o10_cont run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --continue-from runs/4901
# sbatch -J n2o14 run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 14 --n-runs 2 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n1o30smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 30 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100
# sbatch -J n1o10smart run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100


# 6/1

# sbatch -J n1o10_cont run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --continue-from runs/4901
# sbatch --dependency=afterany:4903 -J n1o10smart_cont run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --continue-from runs/4903
# sbatch -J n2o14 run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 14 --n-runs 2 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium

# sbatch -J n1smartre run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100
# sbatch -J n2o14 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 14 --n-runs 2 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n1o10 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n1o30smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 30 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100
# sbatch -J n1o10smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100
# sbatch -J n4o7 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 7 --n-runs 4 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n5o18 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 18 --n-runs 5 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium


# 5/28
# sbatch -J smart-reeval run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100


# 5/26
# sbatch -J n3o30 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 30 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n3o9 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 9 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n1o30 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 30 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n10o9 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 9 --n-runs 10 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J topk_race run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --n-extra-runs 5 --n-runs-max 25 --exec-feedback-n 3 --population-type topk --max-evals 1000000


# 5/14
# sbatch -J topk run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max-evals 1000000 --population-type topk --exec-feedback-n 3
# p evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max-evals 1000000 --population-type topk --exec-feedback-n 3
# sbatch -J topk_race run.sh evolve_pysr.py --operator-type all --generations 25 --population 10 --offspring 10 --n-runs 3 --n-extra-runs 5 --n-runs-max 25 --exec-feedback-n 3 --population-type topk

# 5/8
# sbatch -J full run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max-evals 1000000 --population-type task --exec-feedback-n 3
# sbatch -J race run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --n-extra-runs 5 --n-runs-max 25 --max-evals 1000000 --exec-feedback-n 3 --continue-from runs/982249
# sbatch -J loss run.sh evolve_pysr.py --operator-type loss --generations 5 --population 10 --offspring 10 --n-runs 10 --max-evals 1000000 --population-type task --exec-feedback-n 3

# 5/6
# Simplify-only run: seed initial pop with the best bundle from 399313, then
# only ever apply the simplify meta-mutation (init pop + every generation's
# offspring). 10 gens, 10 offspring/gen, 10 seeds/eval.
# sbatch -J simplify run.sh evolve_pysr.py --operator-type all --generations 10 --population 10 --offspring 10 --n-runs 10 --max-evals 1000000 --baseline runs/399313 --mutation-mode simplify --population-type complexity --continue-from runs/666285
# sbatch -J race-train run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --n-extra-runs 5 --n-runs-max 25 --lambda-target 5 --max-evals 1000000 --exec-feedback-n 3 --continue-from runs/666286 --split splits/train.txt --val-split splits/val.txt

# 5/6
# sbatch -J race2 run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --n-extra-runs 5 --n-runs-max 25 --lambda-target 5 --max-evals 1000000 --exec-feedback-n 3
# sbatch -J race4 run.sh evolve_pysr.py --operator-type all --generations 40 --population 10 --offspring 10 --n-runs 6 --n-extra-runs 10 --n-runs-max 50 --max-evals 1000000 --exec-feedback-n 3 --lambda-target 2 --continue-from runs/666286


# 5/5


# 5/4
# sbatch -J loss run.sh evolve_pysr.py --operator_type loss --generations 5 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --task_diverse_pop --exec_feedback_n 3
# sbatch -J full run.sh evolve_pysr.py --operator_type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --task_diverse_pop --exec_feedback_n 3
# sbatch -J racing --dependency=afterany:399304 run.sh evolve_pysr.py --operator_type all --generations 30 --population 10 --offspring 10 --n-runs 5 --max_evals 1000000 --racing --exec_feedback_n 3
# sbatch -J train run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max-evals 1000000 --task-diverse-pop --exec-feedback-n 3 --split splits/train.txt --val-split splits/val.txt
# sbatch -J train3 run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --max-evals 1000000 --task-diverse-pop --exec-feedback-n 3 --split splits/train.txt --val-split splits/val.txt


# sbatch -J test run.sh evolve_pysr.py --operator_type loss --generations 2 --population 3 --offspring 3 --n-runs 1 --max_evals 10000 --task_diverse_pop --exec_feedback_n 3

# sbatch -J loss run.sh evolve_pysr.py --operator_type loss --generations 10 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --task_diverse_pop --exec_feedback_n 3

# 4/30
# sbatch -J smart2 --mem 20G run.sh evolve_pysr.py --operator_type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --task_diverse_pop --exec_feedback_n 3 --continue_from runs/947961
# sbatch -J smart_no_task --mem 20G run.sh evolve_pysr.py --operator_type all --generations 25 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --exec_feedback_n 3

# SPLITS="splits/barely_unsolvable_val2.txt"
# SEED=1000
# BUNDLE_RESULTS=runs/947961/run_data.json
# JID1A=$(sbatch --parsable -J eval_947961_val2  --mem 20G run.sh evaluate_new_pysr.py --evolve-results $BUNDLE_RESULTS --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 1000000 --timeout 600 --pysr-wall-limit 900 --time-limit 02:00:00)
# JID1B=$(sbatch --parsable -J eval_baseline_val2 --mem 20G run.sh evaluate_new_pysr.py                                  --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 1000000 --timeout 600 --pysr-wall-limit 900 --time-limit 02:00:00)

# 4/29 — Final eval of 947961 bundle vs PySR baseline.
# Splits: train + val + barely_unsolvable (60 datasets).
# 10 seeds, fresh seed base (1000) so we don't reuse cached evolve-run results.
# 4 budget conditions x 2 methods (947961 bundle, baseline) = 8 invocations.
# Conditions run sequentially via --dependency=afterany; the two methods within
# each condition run in parallel.

# BUNDLE_RESULTS=runs/947961/run_data.json
# SPLITS="splits/train.txt splits/val.txt splits/barely_unsolvable.txt"
# SEED=1000

# # (1) 1e6 max_evals — matches evolve-run training budget.
# JID1A=$(sbatch --parsable -J eval_947961_1e6  --mem 20G run.sh evaluate_new_pysr.py --evolve-results $BUNDLE_RESULTS --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 1000000 --timeout 600 --pysr-wall-limit 900 --time-limit 02:00:00)
# JID1B=$(sbatch --parsable -J eval_baseline_1e6 --mem 20G run.sh evaluate_new_pysr.py                                  --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 1000000 --timeout 600 --pysr-wall-limit 900 --time-limit 02:00:00)

# # (2) Wall-clock only (no max_evals; PySR stops on timeout_in_seconds=300s).
# JID2A=$(sbatch --parsable --dependency=afterany:$JID1A:$JID1B -J eval_947961_wc   --mem 20G run.sh evaluate_new_pysr.py --evolve-results $BUNDLE_RESULTS --splits $SPLITS --seed $SEED --n-runs 10 --wall-clock-only --timeout 300 --pysr-wall-limit 600 --time-limit 01:00:00)
# JID2B=$(sbatch --parsable --dependency=afterany:$JID1A:$JID1B -J eval_baseline_wc --mem 20G run.sh evaluate_new_pysr.py                                  --splits $SPLITS --seed $SEED --n-runs 10 --wall-clock-only --timeout 300 --pysr-wall-limit 600 --time-limit 01:00:00)

# # (3) 1e7 max_evals — 10x training budget.
# JID3A=$(sbatch --parsable --dependency=afterany:$JID2A:$JID2B -J eval_947961_1e7  --mem 20G run.sh evaluate_new_pysr.py --evolve-results $BUNDLE_RESULTS --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 10000000 --timeout 3600 --pysr-wall-limit 4200 --time-limit 06:00:00)
# JID3B=$(sbatch --parsable --dependency=afterany:$JID2A:$JID2B -J eval_baseline_1e7 --mem 20G run.sh evaluate_new_pysr.py                                 --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 10000000 --timeout 3600 --pysr-wall-limit 4200 --time-limit 06:00:00)

# # (4) 1e6 max_evals + Gaussian target noise (SRBench standard 0.001).
# JID4A=$(sbatch --parsable --dependency=afterany:$JID3A:$JID3B -J eval_947961_noise  --mem 20G run.sh evaluate_new_pysr.py --evolve-results $BUNDLE_RESULTS --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 1000000 --noise 0.001 --timeout 600 --pysr-wall-limit 900 --time-limit 02:00:00)
# JID4B=$(sbatch --parsable --dependency=afterany:$JID3A:$JID3B -J eval_baseline_noise --mem 20G run.sh evaluate_new_pysr.py                                 --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 1000000 --noise 0.001 --timeout 600 --pysr-wall-limit 900 --time-limit 02:00:00)

# echo "Submitted: (1) $JID1A,$JID1B  (2) $JID2A,$JID2B  (3) $JID3A,$JID3B  (4) $JID4A,$JID4B"


# 4/29 — Bundle HPO of the 947961 best bundle (combined base PySR hparams +
# LLM-extracted operator hparams, max 2/operator). 500 trials, 3 runs/trial,
# up to 20 trials in parallel.
# sbatch -J hpo_947961 --mem 20G run.sh hpo_pysr.py --baseline runs/947961 --n-trials 100 --n-parallel 20 --n-runs 10 --max-op-hparams 2 --split splits/barely_unsolvable.txt --max-evals 1000000 --time-limit 02:00:00


# 4/24
# sbatch -J smart2 --mem 20G run.sh evolve_pysr.py --operator_type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --exec_feedback_n 3 --models best --continue_from runs/947961
# sbatch -J smart_no_task --mem 20G run.sh evolve_pysr.py --operator_type all --generations 25 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt --exec_feedback_n 3 --models best


# 4/23
# sbatch -J smart_test --mem 40G run.sh evolve_pysr.py --operator_type all --generations 10 --population 10 --offspring 20 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --exec_feedback_n 3 --models best --hp_tuning_trials 3 --hpo-n-runs 3


# 4/22
# sbatch -J task --mem 40G run.sh evolve_pysr.py --operator_type all --generations 10 --population 10 --offspring 20 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --exec_feedback_n 3 --hp_tuning_trials 3 --hpo-n-runs 3 --models best

# 4/21
# MiniSR vs. PySR baseline comparison on full SRBench (130 datasets × 3 seeds × 2 engines).
# Driver runs under run.sh on a login/compute node and submits its own sub-arrays via SLURM.
# sbatch -J minisr_vs_pysr run.sh compare_minisr_vs_pysr.py --split splits/srbench_all.txt --n-runs 3 --max-evals 1000000
# sbatch -J task --mem 40G run.sh evolve_pysr.py --operator_type all --generations 5 --population 10 --offspring 10 --n-runs 5 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --exec_feedback_n 3

# 4/20
# sbatch -J exec run.sh evolve_pysr.py --operator_type all --generations 5 --population 5 --offspring 20 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt --exec_feedback_n 3
# sbatch -J racing run.sh evolve_pysr.py --operator_type all --generations 30 --population 5 --offspring 5 --n-runs 2 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing --exec_feedback_n 3
# sbatch -J task --mem 40G run.sh evolve_pysr.py --operator_type all --generations 5 --population 10 --offspring 10 --n-runs 5 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --exec_feedback_n 3
# sbatch -J exec run.sh evolve_pysr.py --operator_type all --generations 10 --population 5 --offspring 20 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt --exec_feedback_n 3
# sbatch -J racing run.sh evolve_pysr.py --operator_type all --generations 60 --population 5 --offspring 5 --n-runs 2 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing --exec_feedback_n 3


# 4/18
# sbatch -J racing run.sh evolve_pysr.py --operator_type all --generations 60 --population 5 --offspring 5 --n-runs 2 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing --hof
# sbatch -J cont_racing run.sh evolve_pysr.py --operator_type all --generations 60 --population 5 --offspring 5 --n-runs 2 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing --continue_from runs/499255

# 4/16/26
# sbatch -J big run.sh evolve_pysr.py --operator_type all --generations 10 --population 5 --offspring 20 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt
# sbatch -J racing run.sh evolve_pysr.py --operator_type all --generations 60 --population 5 --offspring 5 --n-runs 2 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing
# sbatch -J task --mem 40G run.sh evolve_pysr.py --operator_type all --generations 10 --population 10 --offspring 10 --n-runs 5 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --task_aware


# 4/14/26
# sbatch -J racing run.sh evolve_pysr.py --operator_type all --generations 30 --population 5 --offspring 5 --n-runs 1 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing
# sbatch -J task run.sh evolve_pysr.py --operator_type all --generations 10 --population 10 --offspring 10 --n-runs 5 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --task_aware
# sbatch -J topk_10 run.sh evolve_pysr.py --operator_type all --generations 1 --population 40 --offspring 40 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt

# new hpo run
# sbatch -J hpo5 run.sh hpo_pysr.py --n-trials 100 --n-parallel 2 --n-runs 5
# sbatch -J hpo5_bu run.sh hpo_pysr.py --n-trials 100 --n-parallel 2 --split splits/barely_unsolvable.txt --n-runs 5

# top-k run with --n-runs 10
# sbatch -J topk_10 run.sh evolve_pysr.py --operator_type all --generations 1 --population 40 --offspring 40 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt
# sbatch -J racing run.sh evolve_pysr.py --operator_type all --generations 10 --population 10 --offspring 10 --n-runs 1 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing
# sbatch -J task run.sh evolve_pysr.py --operator_type all --generations 10 --population 10 --offspring 10 --n-runs 1 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --task_aware

# 4/10/26 — task-aware mutation/crossover smoke test (3 generations)
# sbatch run.sh evolve_pysr.py --operator_type mutation --generations 3 --task_aware --task_aware_prob 0.5 --task_diverse_pop
# /home/sca63/meta_sr/outputs/openevolve_pysr_selection_20260407_210230
# 4/9/26 — Task-diverse population experiment (selection, with vs without)
# sbatch run.sh evolve_pysr.py --operator_type selection --baseline outputs/hpo_pysr_20260407_152534 --generations 50 --task_diverse_pop
# sbatch run.sh evolve_pysr.py --operator_type selection --baseline outputs/hpo_pysr_20260407_152534 --generations 50

# 4/9/26 — Final evaluations (10 seeds, train+val)

# # 199034: openevolve selection (0.43)
# sbatch run.sh evaluate_new_pysr.py \
#     --openevolve-results outputs/openevolve_pysr_selection_20260408_213220 \
#     --n-runs 10

# # 199036: evolve selection (0.43)
# sbatch run.sh evaluate_new_pysr.py \
#     --evolve-results outputs/evolve_selection_20260408_213226/run_data.json \
#     --n-runs 10

# # 172094: evolve selection (0.43) — out file overwritten, found in outputs/
# sbatch run.sh evaluate_new_pysr.py \
#     --evolve-results outputs/evolve_selection_20260408_144423/run_data.json \
#     --n-runs 10

# # 199033: evolve mutation+selection bundle (0.43)
# sbatch run.sh evaluate_new_pysr.py \
#     --evolve-results outputs/evolve_mutation+survival+selection_20260408_213219/run_data.json \
#     --n-runs 10

# # 199017: hpo 500 trials
# sbatch run.sh evaluate_new_pysr.py \
#     --best-weights outputs/hpo_pysr_20260408_212941/best_params.json \
#     --n-runs 10

# # 171895: evolve mutation+selection bundle (0.43)
# sbatch run.sh evaluate_new_pysr.py \
#     --evolve-results outputs/evolve_mutation+survival+selection_20260408_143835/run_data.json \
#     --n-runs 10

# # 172092: openevolve selection (0.45)
# sbatch run.sh evaluate_new_pysr.py \
#     --openevolve-results outputs/openevolve_pysr_selection_20260408_144423 \
#     --n-runs 10

# # 151084: openevolve mutation
# sbatch run.sh evaluate_new_pysr.py \
#     --openevolve-results outputs/openevolve_pysr_mutation_20260408_000544 \
#     --n-runs 10

# # 144408: openevolve selection (0.48!)
# sbatch run.sh evaluate_new_pysr.py \
#     --openevolve-results outputs/openevolve_pysr_selection_20260407_210230 \
#     --n-runs 10

# # 136931: openevolve selection (0.45)
# sbatch run.sh evaluate_new_pysr.py \
#     --openevolve-results outputs/openevolve_pysr_selection_20260407_155505 \
#     --n-runs 10

# # 136931 baseline (no operator — logs baseline to wandb)
# sbatch run.sh evaluate_new_pysr.py \
#     --n-runs 10

# # 4/9/26
# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type bundle \
#     --iterations 200 \
#     --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh hpo_pysr.py --n-trials 10
# sbatch run.sh evolve_pysr.py --operator_type all --baseline outputs/hpo_pysr_20260407_152534 --generations 3

# 4/8
# sbatch run.sh hpo_pysr.py --n-trials 500
# sbatch run.sh evolve_pysr.py --operator_type all --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type selection \
#     --iterations 200 \
#     --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type bundle \
#     --iterations 200 \
#     --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh evolve_pysr.py --operator_type selection --baseline outputs/hpo_pysr_20260407_152534 --generations 50

# sbatch run.sh run_pysr_srbench.py --dataset feynman_III_15_27 --time_minutes 1
# sbatch run_pysr.sh
# sbatch run_meta_sr.sh
# sbatch run_sr.sh
# sbatch run_meta_sr.sh --no-trace-feedback
# sbatch run_meta_sr.sh

# for target_noise in 0.001 0.01 0.1; do
#   for max_samples in 100000000; do
    # sbatch --time=10:00:00 run_pysr.sh --results_dir results_pysr_${target_noise}_${max_samples} --max_evals ${max_samples} --target_noise ${target_noise}
#   done
# done

# sbatch --time=10:00:00 run_pysr.sh --results_dir results_pysr_1e6 --max_evals 1000000

# sbatch --time=01:00:00 run_pysr.sh --results_dir results_pysr_1e3 --max_evals 1000 --target_noise 0.001
# sbatch --time=01:00:00 run_pysr.sh --results_dir results_pysr_1e4 --max_evals 10000
# sbatch --time=01:00:00 run_pysr.sh --results_dir results_pysr_1e5 --max_evals 100000
# sbatch --time=01:00:00 run_pysr.sh --results_dir results_pysr_1e6 --max_evals 1000000
# sbatch --time=04:00:00 run_pysr.sh --results_dir results_pysr_1e7 --max_evals 10000000
# sbatch --time=08:00:00 run_pysr.sh --results_dir results_pysr_1e8 --max_evals 100000000
# python evolve_pysr.py --operator_type mutation --generations 2 --n-runs 3
# sbatch run.sh hpo_pysr.py --n-trials 500

# sbatch run.sh evolve_pysr.py --operator_type all --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh evolve_pysr.py --operator_type mutation --fitness_metric gt --split splits/train_small.txt --n-runs 1 --max_evals 100000
# sbatch run.sh evolve_basic_sr.py --split splits/train.txt
# # sbatch run.sh evolve_basic_sr.py
# sbatch run.sh evolve_pysr.py --operator_type selection --fitness_metric gt --split splits/train_small.txt
# sbatch run.sh evolve_pysr.py --operator_type survival --fitness_metric gt --split splits/train.txt
# sbatch run.sh hpo_pysr.py --n-trials 500

# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type selection \
#     --iterations 200 \
#     --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type bundle \
#     --iterations 200 \
#     --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh evolve_pysr.py --operator_type selection --baseline outputs/hpo_pysr_20260407_152534 --generations 50

# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type survival \
#     --iterations 200

# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type mutation \
#     --iterations 200


# sbatch run.sh evolve_pysr.py --operator_type survival --fitness_metric gt --split splits/train.txt
