# 5/4
sbatch -J test run.sh evolve_pysr.py --operator_type all --generations 2 --population 3 --offspring 3 --n-runs 1 --max_evals 10000 --task_diverse_pop --exec_feedback_n 3


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
