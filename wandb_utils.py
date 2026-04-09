"""
Shared wandb logging utilities for meta-sr experiments.

Usage:
    from wandb_utils import init_wandb, log_wandb_summary, finish_wandb

    run = init_wandb(args, script_name="evolve_pysr.py")
    # ... run experiment ...
    log_wandb_summary(run, evaluator=evaluator)
    finish_wandb(run)
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

WANDB_ENTITY = "simon-alford"
WANDB_PROJECT = "meta-sr"


def _get_slurm_info() -> Dict[str, str]:
    """Get SLURM job info from environment."""
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
    }


def _get_command() -> str:
    """Reconstruct the command used to launch this script."""
    return " ".join(sys.argv)


def _get_cpu_snapshot() -> Optional[Dict[str, Any]]:
    """Run log_cpus.sh and parse the output for current CPU usage."""
    script = Path(__file__).resolve().parent / "log_cpus.sh"
    if not script.exists():
        return None
    try:
        result = subprocess.run(
            ["bash", str(script)],
            capture_output=True, text=True, timeout=15,
        )
        line = result.stdout.strip()
        # Format: "Mon 4/7/26 14:30: Using 42 out of 200; 100 idle"
        m = re.search(r"Using (\d+) out of (\d+); (\d+) idle", line)
        if m:
            return {
                "cpus_in_use": int(m.group(1)),
                "cpus_total": int(m.group(2)),
                "cpus_idle": int(m.group(3)),
            }
    except Exception:
        pass
    return None


def init_wandb(
    config: Dict[str, Any],
    script_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    extra_tags: Optional[list] = None,
) -> Any:
    """Initialize a wandb run.

    Args:
        config: Hyperparameters / config dict to log.
        script_name: Name of the python script (e.g. "evolve_pysr.py").
        output_dir: Experiment output directory.
        extra_tags: Optional additional tags.

    Returns:
        The wandb run object, or None if wandb init fails.
    """
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed, skipping logging")
        return None

    slurm_info = _get_slurm_info()
    command = _get_command()

    if script_name is None:
        script_name = Path(sys.argv[0]).name

    # Build tags
    tags = [script_name]
    if extra_tags:
        tags.extend(extra_tags)

    # Build config with metadata
    wandb_config = {
        **config,
        "script_name": script_name,
        "command": command,
        "output_dir": output_dir or "",
        "slurm_job_id": slurm_info["slurm_job_id"],
        "slurm_job_name": slurm_info["slurm_job_name"],
    }

    # Use slurm job name or script name as the run name
    run_name = slurm_info["slurm_job_name"] or script_name.replace(".py", "")

    try:
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            config=wandb_config,
            name=run_name,
            tags=tags,
            reinit=True,
        )
        print(f"wandb run initialized: {run.url}")
        return run
    except Exception as e:
        print(f"WARNING: wandb init failed: {e}")
        return None


def log_cpu_usage(run: Any) -> None:
    """Log a CPU usage snapshot to wandb."""
    if run is None:
        return
    import wandb
    snap = _get_cpu_snapshot()
    if snap:
        wandb.log(snap)


def log_wandb_summary(
    run: Any,
    evaluator: Any = None,
    extra_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Log final summary metrics to the wandb run.

    Args:
        run: The wandb run object (or None).
        evaluator: A PySRSlurmEvaluator with .total_sr_evals / .total_sr_cached.
        extra_summary: Additional key-value pairs to log as summary.
    """
    if run is None:
        return

    import wandb

    summary = {}

    # OpenRouter cost and cache stats
    try:
        from completions import get_usage
        usage = get_usage()
        summary["openrouter_cost"] = usage["total_cost"]
        summary["openrouter_num_calls"] = usage["num_calls"]
        summary["openrouter_num_cached_calls"] = usage["num_cached_calls"]
        total_queries = usage["num_calls"] + usage["num_cached_calls"]
        summary["openrouter_cache_fraction"] = (
            usage["num_cached_calls"] / total_queries if total_queries > 0 else 0.0
        )
        summary["openrouter_prompt_tokens"] = usage["prompt_tokens"]
        summary["openrouter_completion_tokens"] = usage["completion_tokens"]
    except Exception:
        pass

    # SR eval cache stats
    if evaluator is not None:
        total = getattr(evaluator, "total_sr_evals", 0)
        cached = getattr(evaluator, "total_sr_cached", 0)
        summary["sr_evals_total"] = total
        summary["sr_evals_cached"] = cached
        summary["sr_evals_cache_fraction"] = cached / total if total > 0 else 0.0

    # CPU usage snapshot at end
    snap = _get_cpu_snapshot()
    if snap:
        summary["final_cpus_in_use"] = snap["cpus_in_use"]
        summary["final_cpus_total"] = snap["cpus_total"]

    if extra_summary:
        summary.update(extra_summary)

    summary["end_time"] = datetime.now().isoformat()

    for k, v in summary.items():
        run.summary[k] = v


def finish_wandb(run: Any) -> None:
    """Finish the wandb run."""
    if run is None:
        return
    try:
        import wandb
        wandb.finish()
    except Exception as e:
        print(f"WARNING: wandb finish failed: {e}")
