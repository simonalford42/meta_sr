"""Shared budget / timeout resolution for evolve_pysr.py and evolve_fullsr.py.

A PySR/FullSR run can be bounded two ways:
  * eval budget   (--max-evals, default 1e6): the search stops after N evaluations.
  * time budget   (--max-time-in-seconds T):  the search stops after ~T seconds.

In eval mode the stopping criterion is `max_evals`, and `timeout_in_seconds`
(soft) / `wall_limit` (hard SIGALRM) / `job_timeout` (batch watchdog) /
`slurm_time_limit` are *safety rails* sized comfortably above a typical 1e6-eval
run (which takes ~M seconds on the reference node).

In time mode the stopping criterion becomes the wall clock itself: we set
`timeout_in_seconds = T` (PySR/FullSR checks between iterations and stops near
T), drop the eval cap (`max_evals = None`), and *extrapolate* the safety rails
from their eval-mode values by the factor T/M, preserving the eval-mode ratios
between budget-time, hard wall, batch watchdog, and SLURM wall. See
resolve_run_budget() for the exact formula.

M (DEFAULT_SECONDS_PER_1E6_EVALS) is measured: runs/414990 analysis gave base
PySR ~127s and the evolved-population mean ~211s for a full 1e6 evals; 200 is a
representative round value covering the population the rails must protect.
Override per-run with --seconds-per-1e6 to recalibrate for other hardware.
"""
from typing import Optional, Dict, Any

DEFAULT_SECONDS_PER_1E6_EVALS = 200.0

# Eval-mode reference rails. The ratios among these define the safety policy
# that is preserved (scaled by T/M) when switching to a time budget.
_REF_TIMEOUT = 500          # soft timeout_in_seconds
_REF_WALL_LIMIT = 600       # hard per-fit wall (SIGALRM)
_REF_JOB_TIMEOUT = 1800.0   # batch watchdog


def _hms_to_seconds(s: str) -> int:
    parts = [int(p) for p in s.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3], parts[-2], parts[-1]
    return h * 3600 + m * 60 + sec


def _seconds_to_hms(total: int) -> str:
    total = max(60, int(round(total)))
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def resolve_run_budget(
    *,
    max_evals: int,
    max_time_in_seconds: Optional[float],
    timeout: Optional[int],
    wall_limit: Optional[int],
    job_timeout: Optional[float],
    slurm_time_limit: Optional[str],
    seconds_per_1e6: float = DEFAULT_SECONDS_PER_1E6_EVALS,
    default_timeout: int = _REF_TIMEOUT,
    default_wall_limit: int = _REF_WALL_LIMIT,
    default_job_timeout: float = _REF_JOB_TIMEOUT,
    default_slurm_time_limit: str = "00:15:00",
) -> Dict[str, Any]:
    """Resolve the run's budget + timeout settings.

    Any of timeout/wall_limit/job_timeout/slurm_time_limit passed as None means
    "not overridden by the user" and is filled in automatically. An explicitly
    supplied value always wins in either mode.

    Returns a dict with keys: mode ('evals'|'time'), max_evals (int|None),
    timeout_in_seconds, wall_limit, job_timeout, slurm_time_limit, scale,
    seconds_per_1e6, max_time_in_seconds.
    """
    if max_time_in_seconds is None:
        # ---- eval-budget mode: unchanged behaviour ----
        return {
            "mode": "evals",
            "max_evals": max_evals,
            "timeout_in_seconds": timeout if timeout is not None else default_timeout,
            "wall_limit": wall_limit if wall_limit is not None else default_wall_limit,
            "job_timeout": job_timeout if job_timeout is not None else default_job_timeout,
            "slurm_time_limit": slurm_time_limit or default_slurm_time_limit,
            "scale": None,
            "seconds_per_1e6": seconds_per_1e6,
            "max_time_in_seconds": None,
        }

    # ---- time-budget mode ----
    T = float(max_time_in_seconds)
    if T <= 0:
        raise ValueError("--max-time-in-seconds must be positive")
    M = float(seconds_per_1e6)
    scale = T / M

    # The soft timeout IS the budget: the search stops between iterations at ~T.
    soft = int(round(T)) if timeout is None else timeout
    # Hard wall scales by T/M; floored above the budget so the graceful soft stop
    # (and HOF write + the ~16s first-fit JIT compile) always happens first.
    if wall_limit is not None:
        hard = wall_limit
    else:
        hard = max(int(round(default_wall_limit * scale)), int(round(T)) + 30)
    # Batch watchdog scales by T/M; floored to comfortably exceed one hard wall.
    if job_timeout is not None:
        job = job_timeout
    else:
        job = max(default_job_timeout * scale, hard * 1.5)
    # SLURM --time scales by T/M; floored above the hard wall.
    if slurm_time_limit is not None:
        slurm = slurm_time_limit
    else:
        scaled = _hms_to_seconds(default_slurm_time_limit) * scale
        slurm = _seconds_to_hms(max(scaled, hard + 60))

    return {
        "mode": "time",
        "max_evals": None,
        "timeout_in_seconds": soft,
        "wall_limit": hard,
        "job_timeout": job,
        "slurm_time_limit": slurm,
        "scale": scale,
        "seconds_per_1e6": M,
        "max_time_in_seconds": T,
    }


def describe_budget(b: Dict[str, Any]) -> str:
    """One-line human summary for logging."""
    if b["mode"] == "evals":
        return (f"budget=evals max_evals={b['max_evals']:,} "
                f"timeout(soft)={b['timeout_in_seconds']}s wall(hard)={b['wall_limit']}s "
                f"job_timeout={b['job_timeout']:.0f}s slurm_time={b['slurm_time_limit']}")
    return (f"budget=time T={b['max_time_in_seconds']:.0f}s "
            f"(M={b['seconds_per_1e6']:.0f}s, scale T/M={b['scale']:.2f}) -> "
            f"timeout(soft=budget)={b['timeout_in_seconds']}s wall(hard)={b['wall_limit']}s "
            f"job_timeout={b['job_timeout']:.0f}s slurm_time={b['slurm_time_limit']} "
            f"max_evals=None")
