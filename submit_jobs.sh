
# sbatch run.sh run_pysr_srbench.py --dataset feynman_III_15_27 --time_minutes 1
# sbatch run_pysr.sh
# sbatch run_meta_sr.sh
# sbatch run_sr.sh
# sbatch run_meta_sr.sh --no-trace-feedback
# sbatch run_meta_sr.sh

# for target_noise in 0.001 0.01 0.1; do
#   for max_samples in 100000000; do
#     sbatch --time=10:00:00 run_pysr.sh --results_dir results_pysr_${target_noise}_${max_samples} --max_evals ${max_samples} --target_noise ${target_noise}
#   done
# done

# sbatch --time=01:00:00 run_pysr.sh --results_dir results_pysr_1e3 --max_evals 1000 --target_noise 0.001
# sbatch --time=01:00:00 run_pysr.sh --results_dir results_pysr_1e4 --max_evals 10000
# sbatch --time=01:00:00 run_pysr.sh --results_dir results_pysr_1e5 --max_evals 100000
# sbatch --time=01:00:00 run_pysr.sh --results_dir results_pysr_1e6 --max_evals 1000000
# sbatch --time=04:00:00 run_pysr.sh --results_dir results_pysr_1e7 --max_evals 10000000
# sbatch --time=08:00:00 run_pysr.sh --results_dir results_pysr_1e8 --max_evals 100000000
# python evolve_pysr.py --generations 2 --n-runs 3
# sbatch run.sh hpo_pysr.py --n-trials 500
# sbatch run.sh evolve_pysr.py

sbatch run.sh evolve_pysr.py --fitness_metric gt
# sbatch run.sh evolve_basic_sr.py

