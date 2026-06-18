#!/usr/bin/env python3
"""Submit a jobs.json (empbench_run jobs) as a SLURM array, one task per job.

Each task is single-core with its own generous memory allocation, so we are not
constrained by the interactive node's shared RAM. Resume-safe via the worker.
"""
import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONDA_SH = "/home/sca63/mambaforge/etc/profile.d/conda.sh"
CONDA_ENV = "meta_sr"

SBATCH_TMPL = """#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --output={logdir}/task_%a.out
#SBATCH --error={logdir}/task_%a.out
#SBATCH --array={array_spec}
#SBATCH --time={time}
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --requeue

source {conda_sh}
conda activate {conda_env}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JULIA_NUM_THREADS=1

cd "{repo}"
export PYTHONPATH="{repo}:$PYTHONPATH"
unset JULIA_PROJECT
export PYTHON_JULIAPKG_PROJECT="{repo}/.juliapkg_env"
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

echo "array task $SLURM_ARRAY_TASK_ID on $(hostname)"
python -u {repo}/scripts/empbench_slurm_worker.py --jobs "{jobs}" --index $SLURM_ARRAY_TASK_ID
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--jobname", default="empbench")
    ap.add_argument("--mem", default="24G")
    ap.add_argument("--time", default="03:00:00")
    ap.add_argument("--partition", default="default_partition")
    ap.add_argument("--throttle", type=int, default=40, help="max concurrent array tasks")
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    jobs = json.loads(Path(a.jobs).read_text())
    n = len(jobs)
    logdir = a.logdir or str(Path(a.jobs).parent / "slurm_logs")
    Path(logdir).mkdir(parents=True, exist_ok=True)
    array_spec = f"0-{n-1}%{a.throttle}" if n > 1 else "0"

    script = SBATCH_TMPL.format(
        jobname=a.jobname, logdir=logdir, array_spec=array_spec, time=a.time,
        mem=a.mem, partition=a.partition, conda_sh=CONDA_SH, conda_env=CONDA_ENV,
        repo=str(REPO), jobs=str(Path(a.jobs).resolve()))
    script_path = Path(a.jobs).parent / f"{a.jobname}_array.sh"
    script_path.write_text(script)
    print(f"Wrote {script_path}  ({n} tasks, array={array_spec}, mem={a.mem}, time={a.time})")

    if a.dry_run:
        print("--- dry run; sbatch script ---")
        print(script)
        return

    out = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    print("sbatch stdout:", out.stdout.strip())
    if out.stderr.strip():
        print("sbatch stderr:", out.stderr.strip())
    if out.returncode != 0:
        raise SystemExit(f"sbatch failed rc={out.returncode}")


if __name__ == "__main__":
    main()
