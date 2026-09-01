#!/usr/bin/env bash

# Shared arguments for each experiment type. Keep these as arrays so Bash
# preserves argument boundaries when they are expanded with "${ARGS[@]}".
HPO_ARGS=(
    --n-trials 500
    --n-runs 3
    --n-parallel 20
    --random-target-noise
)

EVOLVE_ARGS=(
    --random-target-noise
    --n-runs 3
    --n-reevals 10
    --reeval population
    --models best
    --generations 15
    --population 10
    --offspring 10
    --population-type task
)

BARELY_UNSOLVABLE_SPLITS=(
    --split splits/barely_unsolvable.txt
    --val-split splits/barely_unsolvable_val2.txt
)

TRAIN_SPLITS=(
    --split splits/train.txt
    --val-split splits/val.txt
)


# barely_unsolvable.txt

# HPO
# sbatch -J hpo-gt    run.sh hpo_pysr.py --fitness-metric gt    "${HPO_ARGS[@]}" "${BARELY_UNSOLVABLE_SPLITS[@]}"
# sbatch -J hpo-r2    run.sh hpo_pysr.py --fitness-metric r2    "${HPO_ARGS[@]}" "${BARELY_UNSOLVABLE_SPLITS[@]}"
# sbatch -J hpo-gt-r2 run.sh hpo_pysr.py --fitness-metric gt-r2 "${HPO_ARGS[@]}" "${BARELY_UNSOLVABLE_SPLITS[@]}"

# PySR++
# sbatch -J gt    run.sh evolve_pysr.py --fitness-metric gt    "${EVOLVE_ARGS[@]}" "${BARELY_UNSOLVABLE_SPLITS[@]}"
# sbatch -J r2    run.sh evolve_pysr.py --fitness-metric r2    "${EVOLVE_ARGS[@]}" "${BARELY_UNSOLVABLE_SPLITS[@]}"
# sbatch -J gt-r2 run.sh evolve_pysr.py --fitness-metric gt-r2 "${EVOLVE_ARGS[@]}" "${BARELY_UNSOLVABLE_SPLITS[@]}"

# BasicSR++
# sbatch -J full-gt    run.sh evolve_fullsr.py --fitness-metric gt    "${EVOLVE_ARGS[@]}" "${BARELY_UNSOLVABLE_SPLITS[@]}"
# sbatch -J full-r2    run.sh evolve_fullsr.py --fitness-metric r2    "${EVOLVE_ARGS[@]}" "${BARELY_UNSOLVABLE_SPLITS[@]}"
# sbatch -J full-gt-r2 run.sh evolve_fullsr.py --fitness-metric gt-r2 "${EVOLVE_ARGS[@]}" "${BARELY_UNSOLVABLE_SPLITS[@]}"


# train.txt

# HPO
# sbatch -J hpo-gt    run.sh hpo_pysr.py --fitness-metric gt    "${HPO_ARGS[@]}" "${TRAIN_SPLITS[@]}"
# sbatch -J hpo-r2    run.sh hpo_pysr.py --fitness-metric r2    "${HPO_ARGS[@]}" "${TRAIN_SPLITS[@]}"
# sbatch -J hpo-gt-r2 run.sh hpo_pysr.py --fitness-metric gt-r2 "${HPO_ARGS[@]}" "${TRAIN_SPLITS[@]}"

# PySR++
# sbatch -J gt    run.sh evolve_pysr.py --fitness-metric gt    "${EVOLVE_ARGS[@]}" "${TRAIN_SPLITS[@]}"
# sbatch -J r2    run.sh evolve_pysr.py --fitness-metric r2    "${EVOLVE_ARGS[@]}" "${TRAIN_SPLITS[@]}"
# sbatch -J gt-r2 run.sh evolve_pysr.py --fitness-metric gt-r2 "${EVOLVE_ARGS[@]}" "${TRAIN_SPLITS[@]}"

# BasicSR++
# sbatch -J full-gt    run.sh evolve_fullsr.py --fitness-metric gt    "${EVOLVE_ARGS[@]}" "${TRAIN_SPLITS[@]}"
# sbatch -J full-r2    run.sh evolve_fullsr.py --fitness-metric r2    "${EVOLVE_ARGS[@]}" "${TRAIN_SPLITS[@]}"
# sbatch -J full-gt-r2 run.sh evolve_fullsr.py --fitness-metric gt-r2 "${EVOLVE_ARGS[@]}" "${TRAIN_SPLITS[@]}"
