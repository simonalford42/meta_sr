#!/usr/bin/env python3
"""
Launch OpenEvolve for PySR custom operator evolution.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


import hashlib
from utils import copy_slurm_log, load_dataset_names_from_split, resolve_run_dir
from wandb_utils import init_wandb, log_wandb_summary, finish_wandb

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SHUTDOWN_GRACE_SECONDS = float(os.environ.get("OE_SHUTDOWN_GRACE_SECONDS", "30"))
TEMP_SIGNAL_DEBUG_SUBDIR = "TEMP_signal_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenEvolve to evolve PySR custom operators",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--operator-type",
        type=str,
        default="mutation",
        choices=["mutation", "selection", "survival", "bundle"],
        help="Which PySR custom operator type to evolve",
    )
    parser.add_argument("--iterations", type=int, default=50, help="OpenEvolve iterations")
    parser.add_argument("--split", type=str, default="splits/train.txt", help="Dataset split file")
    parser.add_argument("--stage2-datasets", type=int, default=5, help="Datasets used in cascade stage 2")
    parser.add_argument("--fitness-metric", type=str, default="gt", choices=["r2", "gt"], help="Final-stage fitness metric")
    parser.add_argument("--n-runs", type=int, default=3, help="Runs per dataset during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument("--data-seed", type=int, default=42, help="Dataset subsampling seed")
    parser.add_argument("--target-noise", type=float, default=0.0, help="Target noise level")
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument("--random-target-noise", action="store_true",
                             help="Assign per-dataset target noise from {0.0, 0.001, 0.01, 0.1} using the seed")
    noise_group.add_argument("--no-random-target-noise", dest="random_target_noise", action="store_false",
                             help="Disable per-dataset target noise and use --target-noise instead")
    parser.set_defaults(random_target_noise=False)
    parser.add_argument("--max-samples", type=int, default=1000, help="Max samples per dataset")
    parser.add_argument("--max-evals", type=int, default=1000000, help="PySR max_evals")
    parser.add_argument("--timeout", type=int, default=300, help="PySR timeout_in_seconds")
    parser.add_argument("--partition", type=str, default="default_partition", help="SLURM partition")
    parser.add_argument("--time-limit", type=str, default="04:00:00", help="SLURM time limit")
    parser.add_argument("--mem-per-cpu", type=str, default="8G", help="SLURM memory per CPU")
    parser.add_argument("--job-timeout", type=float, default=3000.0, help="Wait timeout for SLURM jobs")
    parser.add_argument("--output-dir", type=str, default=None, help="OpenEvolve output directory")
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "openevolve_pysr" / "config.yaml"), help="OpenEvolve config file")
    parser.add_argument("--api-base", type=str, default=None, help="Override API base")
    parser.add_argument("--primary-model", type=str, default=None, help="Override primary model")
    parser.add_argument("--secondary-model", type=str, default=None, help="Override secondary model")
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Initialize from a previous run. Accepts: evolve_pysr output dir or run_data.json, "
             "hpo_pysr output dir or best_params.json, openevolve output dir or best_program.py, "
             "or a raw .jl file. Loads operator code and/or hparams together.",
    )
    return parser.parse_args()


def _generate_initial_program_from_baseline(
    baseline_bundle,
    operator_type: str,
    template_path: Path,
    output_path: Path,
) -> Path:
    """Generate an initial_program.py with baseline operator code filled in.

    Reads the template file for the given operator_type, replaces the
    EVOLVE-BLOCK content with the baseline operator code, and writes the
    result to output_path.
    """
    template = template_path.read_text()

    if operator_type == "bundle":
        # Replace each EVOLVE-BLOCK section with baseline code
        replacements = {
            "CUSTOM_MUTATION_WEIGHT": None,
            "CUSTOM_MUTATION_CODE": None,
            "CUSTOM_SELECTION_CODE": None,
            "CUSTOM_SURVIVAL_CODE": None,
        }

        mut = baseline_bundle.operators.get("mutation")
        if mut is not None:
            replacements["CUSTOM_MUTATION_CODE"] = mut.code
            replacements["CUSTOM_MUTATION_WEIGHT"] = mut.weight if mut.weight is not None else 0.5

        sel = baseline_bundle.operators.get("selection")
        if sel is not None:
            replacements["CUSTOM_SELECTION_CODE"] = sel.code

        surv = baseline_bundle.operators.get("survival")
        if surv is not None:
            replacements["CUSTOM_SURVIVAL_CODE"] = surv.code

        # Replace code variables in the template
        import re
        for var_name, value in replacements.items():
            if value is None:
                continue
            if var_name == "CUSTOM_MUTATION_WEIGHT":
                # Replace the weight assignment
                template = re.sub(
                    r'^CUSTOM_MUTATION_WEIGHT\s*=\s*[\d.]+',
                    f'CUSTOM_MUTATION_WEIGHT = {value}',
                    template,
                    flags=re.MULTILINE,
                )
            else:
                # Replace the code string between EVOLVE-BLOCK markers
                # Match: VAR_NAME = r"""..."""
                pattern = rf'({re.escape(var_name)}\s*=\s*r""")\n.*?(""")'
                replacement_code = value.strip()
                template = re.sub(
                    pattern,
                    rf'\1\n{replacement_code}\n\2',
                    template,
                    flags=re.DOTALL,
                )
    else:
        # Single operator type
        op = None
        for op_candidate in baseline_bundle.operators.values():
            if op_candidate is not None:
                op = op_candidate
                break

        if op is not None:
            import re
            # Replace code string
            code_var = {
                "mutation": "CUSTOM_MUTATION_CODE",
                "selection": "CUSTOM_SELECTION_CODE",
                "survival": "CUSTOM_SURVIVAL_CODE",
            }[operator_type]
            pattern = rf'({re.escape(code_var)}\s*=\s*r""")\n.*?(""")'
            template = re.sub(
                pattern,
                rf'\1\n{op.code.strip()}\n\2',
                template,
                flags=re.DOTALL,
            )
            # Replace weight for mutation
            if operator_type == "mutation" and op.weight is not None:
                template = re.sub(
                    r'^CUSTOM_MUTATION_WEIGHT\s*=\s*[\d.]+',
                    f'CUSTOM_MUTATION_WEIGHT = {op.weight}',
                    template,
                    flags=re.MULTILINE,
                )

    generated = output_path / "initial_program.py"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(template)
    print(f"Generated initial program from baseline: {generated}")
    return generated


def _load_registered_child_job_ids(registry_path: "Path | None") -> list[str]:
    """Read unique child SLURM job IDs from the temporary registry."""
    if registry_path is None or not registry_path.exists():
        return []

    job_ids = []
    seen = set()
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                job_id = str(payload.get("job_id", "")).strip()
                if job_id and job_id not in seen:
                    seen.add(job_id)
                    job_ids.append(job_id)
    except Exception:
        return []

    return job_ids


def _run_debug_command(cmd: list[str], timeout: float = 15.0) -> str:
    """Run a diagnostics command and format the output for the log file."""
    header = "$ " + " ".join(shlex.quote(part) for part in cmd)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return f"{header}\nERROR: {exc}\n"

    output = result.stdout
    if result.stderr:
        output += ("\n" if output and not output.endswith("\n") else "") + result.stderr
    if not output:
        output = f"<no output> (exit={result.returncode})\n"
    elif not output.endswith("\n"):
        output += "\n"
    return f"{header}\n{output}"


def _append_signal_diagnostics(
    signal_debug_file: Path,
    *,
    phase: str,
    signum: int,
    process: subprocess.Popen,
    child_job_registry: "Path | None",
) -> None:
    """Append a snapshot of SLURM and process state for signal debugging.

    TEMP DEBUG: remove this helper and its callers once the unexpected SIGTERM
    issue is understood.
    """
    signal_debug_file.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    sections = [
        "",
        (
            f"=== TEMP SIGNAL DEBUG START {timestamp} phase={phase} signum={signum} "
            f"wrapper_pid={os.getpid()} child_pid={process.pid} ==="
        ),
        "Note: TEMP signal diagnostics in run_openevolve_pysr.py. Remove after the SIGTERM root cause is fixed.",
    ]

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if slurm_job_id:
        sections.append(_run_debug_command(["squeue", "-j", slurm_job_id, "-o", "%i %T %M %L %R", "-h"]))
        sections.append(
            _run_debug_command(
                [
                    "sacct",
                    "-j",
                    slurm_job_id,
                    "--format=JobID,JobName%25,State,ExitCode,DerivedExitCode,Elapsed,Timelimit,NodeList%30,Reason%40",
                ]
            )
        )
        sections.append(_run_debug_command(["scontrol", "show", "job", slurm_job_id]))
    else:
        sections.append("No SLURM_JOB_ID in environment\n")

    sections.append(
        _run_debug_command(
            [
                "bash",
                "-lc",
                (
                    "ps -eo pid,ppid,pgid,sid,stat,etime,cmd | "
                    f"awk 'NR==1 || $1=={os.getpid()} || $1=={process.pid} || $3=={process.pid} {{print}}'"
                ),
            ]
        )
    )

    child_job_ids = _load_registered_child_job_ids(child_job_registry)
    if child_job_ids:
        child_job_arg = ",".join(child_job_ids)
        sections.append(
            _run_debug_command(
                ["squeue", "-j", child_job_arg, "-o", "%i %T %M %L %R", "-h"]
            )
        )
        sections.append(
            _run_debug_command(
                [
                    "sacct",
                    "-j",
                    child_job_arg,
                    "--format=JobID,JobName%25,State,ExitCode,Elapsed,Timelimit,NodeList%30,Reason%40",
                ]
            )
        )
        try:
            registry_text = child_job_registry.read_text(encoding="utf-8")
        except Exception as exc:
            registry_text = f"ERROR reading child job registry: {exc}\n"
        sections.append(
            f"$ tail -n +1 {shlex.quote(str(child_job_registry))}\n{registry_text}"
            if registry_text
            else f"$ tail -n +1 {shlex.quote(str(child_job_registry))}\n<empty>\n"
        )
    elif child_job_registry is not None:
        sections.append(f"No child jobs recorded yet in {child_job_registry}\n")

    sections.append("=== TEMP SIGNAL DEBUG END ===\n")

    with open(signal_debug_file, "a", encoding="utf-8") as f:
        f.write("\n".join(sections))


def _maybe_wrap_with_signal_strace(
    cmd: list[str],
    *,
    signal_debug_dir: Path,
) -> tuple[list[str], "Path | None"]:
    """TEMP DEBUG: launch the child under strace to capture signal sender info.

    Remove this wrapper once the unexpected SIGTERM root cause is identified.
    """
    if not os.environ.get("SLURM_JOB_ID"):
        return cmd, None

    strace_path = shutil.which("strace")
    if not strace_path:
        return cmd, None

    signal_debug_dir.mkdir(parents=True, exist_ok=True)
    trace_prefix = signal_debug_dir / "signal_strace"
    wrapped_cmd = [
        strace_path,
        "-ff",
        "-tt",
        "-e",
        "trace=signal",
        "-o",
        str(trace_prefix),
        *cmd,
    ]
    return wrapped_cmd, trace_prefix


def _run_openevolve_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    signal_debug_file: "Path | None" = None,
    child_job_registry: "Path | None" = None,
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
) -> int:
    """Run OpenEvolve in its own process group with bounded shutdown."""
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        start_new_session=True,
    )

    signal_state = {
        "signum": None,
        "deadline": None,
        "sent_sigkill": False,
    }
    previous_handlers = {}

    def _terminate_process_group(signum: int) -> None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    def _handle_signal(signum, _frame):
        if signal_state["signum"] is None:
            signal_state["signum"] = signum
            if signal_debug_file is not None:
                _append_signal_diagnostics(
                    signal_debug_file,
                    phase="signal_received",
                    signum=signum,
                    process=process,
                    child_job_registry=child_job_registry,
                )
            signal_state["deadline"] = time.monotonic() + shutdown_grace_seconds
            print(
                f"Received signal {signum}; forwarding to OpenEvolve subprocess "
                f"and allowing {shutdown_grace_seconds:.0f}s for cleanup..."
            )
            _terminate_process_group(signum)
            return

        if not signal_state["sent_sigkill"]:
            print("Received repeated shutdown signal; killing OpenEvolve subprocess group immediately...")
            signal_state["sent_sigkill"] = True
            _terminate_process_group(signal.SIGKILL)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_signal)

    try:
        while True:
            try:
                raw_return_code = process.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                deadline = signal_state["deadline"]
                if (
                    deadline is not None
                    and not signal_state["sent_sigkill"]
                    and time.monotonic() >= deadline
                ):
                    if signal_debug_file is not None:
                        _append_signal_diagnostics(
                            signal_debug_file,
                            phase="grace_period_expired",
                            signum=signal_state["signum"],
                            process=process,
                            child_job_registry=child_job_registry,
                        )
                    print(
                        f"OpenEvolve did not exit within {shutdown_grace_seconds:.0f}s; "
                        "sending SIGKILL to the subprocess group."
                    )
                    signal_state["sent_sigkill"] = True
                    _terminate_process_group(signal.SIGKILL)

        return_code = 128 + (-raw_return_code) if raw_return_code < 0 else raw_return_code
        if signal_state["signum"] is not None and return_code == 0:
            return 128 + signal_state["signum"]
        return return_code
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _log_openevolve_evals_to_wandb(output_path: Path) -> None:
    """Parse OpenEvolve log file and log each evaluated program to wandb.

    Emits one wandb row per program with eval_idx, eval_score (=avg_gt),
    eval_running_best, and eval_iteration, so wandb can recreate the
    avg_gt-over-time scatter + running-best plot from plot_openevolve_logs.py.
    """
    import re
    import wandb

    logs_dir = output_path / "logs"
    if not logs_dir.exists():
        return
    log_files = sorted(logs_dir.glob("openevolve_*.log"))
    if not log_files:
        return

    eval_pat = re.compile(
        r"Evaluated program (\S+) in [\d.]+s: combined_score=[\d.]+.*avg_gt=([\d.]+)"
    )
    iter_pat = re.compile(
        r"Iteration (\d+): Program (\S+) \(parent: \S+\) completed"
    )
    best_pat = re.compile(r"New best program (\S+) replaces")

    program_gt: dict[str, float] = {}
    program_order: list[str] = []
    iter_map: dict[str, int] = {}
    new_bests: set[str] = set()

    for log_file in log_files:
        try:
            with open(log_file) as f:
                for line in f:
                    m = eval_pat.search(line)
                    if m:
                        pid = m.group(1)
                        if pid not in program_gt:
                            program_order.append(pid)
                        program_gt[pid] = float(m.group(2))
                        continue
                    m = iter_pat.search(line)
                    if m:
                        iter_map[m.group(2)] = int(m.group(1))
                        continue
                    m = best_pat.search(line)
                    if m:
                        new_bests.add(m.group(1))
        except Exception as e:
            print(f"WARNING: failed to parse {log_file}: {e}")

    running_best = float("-inf")
    for idx, pid in enumerate(program_order, start=1):
        gt = program_gt[pid]
        if gt > running_best:
            running_best = gt
        wandb.log({
            "eval_idx": idx,
            "eval_score": gt,
            "eval_running_best": running_best,
            "eval_iteration": iter_map.get(pid, 0),
            "eval_is_new_best": int(pid in new_bests),
        })


def main() -> int:
    args = parse_args()

    output_dir = resolve_run_dir(args.output_dir, label=f"openevolve_pysr_{args.operator_type}")

    output_path = REPO_ROOT / output_dir
    # TEMP DEBUG: write extra signal diagnostics here while tracking the
    # unexpected SIGTERM issue. Remove after the root cause is understood.
    signal_debug_dir = output_path / TEMP_SIGNAL_DEBUG_SUBDIR
    signal_debug_file = signal_debug_dir / "wrapper_signal_diagnostics.log"
    child_job_registry_env = os.environ.get("META_SR_CHILD_JOB_REGISTRY")
    child_job_registry = Path(child_job_registry_env) if child_job_registry_env else None
    initial_program_map = {
        "mutation": REPO_ROOT / "openevolve_pysr" / "initial_program.py",
        "selection": REPO_ROOT / "openevolve_pysr" / "initial_program_selection.py",
        "survival": REPO_ROOT / "openevolve_pysr" / "initial_program_survival.py",
        "bundle": REPO_ROOT / "openevolve_pysr" / "initial_program_bundle.py",
    }
    initial_program = initial_program_map[args.operator_type]
    evaluator = REPO_ROOT / "openevolve_pysr" / "evaluator.py"

    # Load baseline and generate seeded initial program if specified
    baseline_bundle = None
    if args.baseline:
        from evolve_pysr import load_baseline_bundle
        baseline_bundle = load_baseline_bundle(
            args.baseline,
            operator_type=args.operator_type if args.operator_type != "bundle" else None,
        )
        initial_program = _generate_initial_program_from_baseline(
            baseline_bundle,
            args.operator_type,
            template_path=initial_program_map[args.operator_type],
            output_path=output_path,
        )

    if args.operator_type == "bundle" and args.config == str(REPO_ROOT / "openevolve_pysr" / "config.yaml"):
        args.config = str(REPO_ROOT / "openevolve_pysr" / "config_bundle.yaml")
    runner = REPO_ROOT / "openevolve" / "openevolve-run.py"

    env = os.environ.copy()
    env.update(
        {
            "OE_PYSR_SPLIT": args.split,
            "OE_PYSR_STAGE2_DATASETS": str(args.stage2_datasets),
            "OE_PYSR_FITNESS_METRIC": args.fitness_metric,
            "OE_PYSR_OPERATOR_TYPE": args.operator_type,
            "OE_PYSR_N_RUNS": str(args.n_runs),
            "OE_PYSR_SEED": str(args.seed),
            "OE_PYSR_DATA_SEED": str(args.data_seed),
            "OE_PYSR_TARGET_NOISE": str(args.target_noise),
            "OE_PYSR_RANDOM_TARGET_NOISE": str(args.random_target_noise),
            "OE_PYSR_MAX_SAMPLES": str(args.max_samples),
            "OE_PYSR_MAX_EVALS": str(args.max_evals),
            "OE_PYSR_TIMEOUT_IN_SECONDS": str(args.timeout),
            "OE_PYSR_PARTITION": args.partition,
            "OE_PYSR_TIME_LIMIT": args.time_limit,
            "OE_PYSR_MEM_PER_CPU": args.mem_per_cpu,
            "OE_PYSR_JOB_TIMEOUT": str(args.job_timeout),
            "OE_PYSR_RESULTS_DIR": str(output_path / "pysr_eval"),
            "OE_PYSR_USE_CACHE": "true",
        }
    )

    # Pass baseline hparams to evaluator and save for round-tripping
    if baseline_bundle is not None and baseline_bundle.best_hparams:
        env["OE_PYSR_BASELINE_HPARAMS"] = json.dumps(baseline_bundle.best_hparams)
        # Save hparams to output dir so loading this OE run preserves them
        output_path.mkdir(parents=True, exist_ok=True)
        hparams_file = output_path / "baseline_hparams.json"
        with open(hparams_file, "w") as f:
            json.dump(baseline_bundle.best_hparams, f, indent=2)
        print(f"Passing {len(baseline_bundle.best_hparams)} hparams to evaluator")
        print(f"Saved hparams to {hparams_file}")

    # Auto-detect latest checkpoint for resume (e.g. after SLURM requeue)
    checkpoint_dir = output_path / "checkpoints"
    latest_checkpoint = None
    if checkpoint_dir.exists():
        checkpoints = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
            key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0,
        )
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"Resuming from checkpoint: {latest_checkpoint}")

    cmd = [
        sys.executable,
        str(runner),
        str(initial_program),
        str(evaluator),
        "--config",
        args.config,
        "--output",
        str(output_path),
        "--iterations",
        str(args.iterations),
    ]
    if latest_checkpoint:
        cmd.extend(["--checkpoint", str(latest_checkpoint)])
    if args.api_base:
        cmd.extend(["--api-base", args.api_base])
    if args.primary_model:
        cmd.extend(["--primary-model", args.primary_model])
    if args.secondary_model:
        cmd.extend(["--secondary-model", args.secondary_model])

    traced_cmd, signal_trace_prefix = _maybe_wrap_with_signal_strace(
        cmd,
        signal_debug_dir=signal_debug_dir,
    )

    # Initialize wandb
    wandb_config = {
        "operator_type": args.operator_type,
        "iterations": args.iterations,
        "split": args.split,
        "stage2_datasets": args.stage2_datasets,
        "fitness_metric": args.fitness_metric,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "target_noise": args.target_noise,
        "random_target_noise": args.random_target_noise,
        "max_samples": args.max_samples,
        "max_evals": args.max_evals,
        "timeout": args.timeout,
        "partition": args.partition,
        "time_limit": args.time_limit,
        "mem_per_cpu": args.mem_per_cpu,
        "config": args.config,
        "baseline": args.baseline,
        "api_base": args.api_base,
        "primary_model": args.primary_model,
        "secondary_model": args.secondary_model,
    }
    wandb_run = init_wandb(
        config=wandb_config,
        script_name="run_openevolve_pysr.py",
        output_dir=str(output_path),
        extra_tags=[args.operator_type],
    )

    print("Running:", " ".join(traced_cmd))
    print(f"Operator type: {args.operator_type}")
    print(f"Output dir: {output_path}")
    print(f"Temporary signal diagnostics: {signal_debug_file}")
    if child_job_registry is not None:
        print(f"Temporary child-job registry: {child_job_registry}")
    if signal_trace_prefix is not None:
        print(
            "Temporary signal strace prefix: "
            f"{signal_trace_prefix}.* "
            "(remove this TEMP DEBUG wrapper once the SIGTERM root cause is fixed)"
        )
    return_code = _run_openevolve_subprocess(
        traced_cmd,
        cwd=REPO_ROOT,
        env=env,
        signal_debug_file=signal_debug_file,
        child_job_registry=child_job_registry,
    )

    # Read best score from OpenEvolve output
    extra = {"return_code": return_code}
    best_info_path = output_path / "best" / "best_program_info.json"
    if best_info_path.exists():
        try:
            with open(best_info_path) as f:
                best_info = json.load(f)
            metrics = best_info.get("metrics", {})
            extra["best_score"] = metrics.get("combined_score")
            extra["best_avg_r2"] = metrics.get("avg_r2")
            extra["best_iteration"] = best_info.get("generation", best_info.get("iteration"))
        except Exception:
            pass

    # Log per-iteration best scores from checkpoints
    if wandb_run is not None:
        import wandb
        checkpoints_dir = output_path / "checkpoints"
        if checkpoints_dir.exists():
            for cp_dir in sorted(checkpoints_dir.iterdir()):
                cp_info = cp_dir / "best_program_info.json"
                if cp_info.exists():
                    try:
                        with open(cp_info) as f:
                            cp_data = json.load(f)
                        iteration = cp_data.get("generation", cp_data.get("iteration", 0))
                        cp_metrics = cp_data.get("metrics", {})
                        score = cp_metrics.get("combined_score")
                        if score is not None:
                            wandb.log({"iteration": iteration, "best_score": score})
                    except Exception:
                        pass

        # Log per-program avg_gt by parsing the OpenEvolve log file. Mirrors
        # scripts/plot_openevolve_logs.py so wandb can render the same
        # avg_gt-over-time / running-best plot.
        _log_openevolve_evals_to_wandb(output_path)

    log_wandb_summary(wandb_run, extra_summary=extra)

    # Final evaluation on train + val (10 seeds)
    best_program_dir = output_path / "best"
    if (best_program_dir / "best_program.py").exists():
        try:
            from evaluate_new_pysr import run_final_evaluation

            TARGET_NOISE_LEVELS = [0.0, 0.001, 0.01, 0.1]
            target_noise_map = None
            if args.random_target_noise:
                all_splits = ["splits/train.txt", "splits/val.txt"]
                all_datasets = []
                for sp in all_splits:
                    all_datasets.extend(load_dataset_names_from_split(sp))
                # Use same deterministic hash as evolve_pysr
                def _stable_noise(name, seed, levels):
                    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
                    return levels[int.from_bytes(digest[:4], "little") % len(levels)]
                target_noise_map = {n: _stable_noise(n, args.seed, TARGET_NOISE_LEVELS)
                                    for n in dict.fromkeys(all_datasets)}

            run_final_evaluation(
                output_dir=str(output_path),
                method_source="openevolve",
                method_path=str(output_path),
                partition=args.partition,
                n_runs=10,
                seed=args.seed,
                max_samples=args.max_samples,
                max_evals=args.max_evals,
                timeout=args.timeout,
                time_limit=args.time_limit,
                mem_per_cpu=args.mem_per_cpu,
                job_timeout=args.job_timeout,
                use_cache=True,
                wandb_run=wandb_run,
                target_noise_map=target_noise_map,
            )
        except Exception as e:
            print(f"\nFinal evaluation failed: {e}")

    finish_wandb(wandb_run)
    copy_slurm_log(output_path)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
