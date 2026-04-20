"""Parameterized evaluator for function_minimization with per-trial evaluator noise.

Env vars:
  NOISE_STD     — stddev of Gaussian noise added to each trial's score before averaging (default 0.0)
  NUM_TRIALS    — number of algorithm trials to average per evaluation (default 10)
  EVAL_LOG_PATH — optional JSONL path for per-evaluation records (includes true vs noisy scores)
  NOISE_SEED    — optional int seed offset for noise RNG (default: derived from pid+time)
"""

import importlib.util
import json
import os
import time
import traceback
import concurrent.futures

import numpy as np

from openevolve.evaluation_result import EvaluationResult


GLOBAL_MIN_X = -1.704
GLOBAL_MIN_Y = 0.678
GLOBAL_MIN_VALUE = -1.519


def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None and v != "" else default


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None and v != "" else default


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=5):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func, *args, **kwargs)
        return fut.result(timeout=timeout_seconds)


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _base_score_from_trial(x, y, value):
    """Deterministic base score given a trial's (x, y, value).

    Same shape as original evaluator's per-trial score (weighted value/distance
    with distance-based multiplier), just computed per-trial instead of once after
    averaging. This is the noise-free quantity we add noise to.
    """
    x_diff = x - GLOBAL_MIN_X
    y_diff = y - GLOBAL_MIN_Y
    distance = float(np.sqrt(x_diff * x_diff + y_diff * y_diff))
    value_score = 1.0 / (1.0 + abs(value - GLOBAL_MIN_VALUE))
    distance_score = 1.0 / (1.0 + distance)
    if distance < 0.5:
        mult = 1.5
    elif distance < 1.5:
        mult = 1.2
    elif distance < 3.0:
        mult = 1.0
    else:
        mult = 0.7
    base = 0.5 * value_score + 0.3 * distance_score + 0.2  # reliability=1 for a successful trial
    return float(base * mult), float(value_score), float(distance_score), distance


def _jsonl_append(path, record):
    if not path:
        return
    try:
        line = json.dumps(record, default=float)
        with open(path, "a") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[evaluator] failed to append log: {e}")


def _load_program(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    return program


def _error_result(err_type, message, suggestion=""):
    return EvaluationResult(
        metrics={
            "value_score": 0.0,
            "distance_score": 0.0,
            "reliability_score": 0.0,
            "combined_score": 0.0,
            "true_combined_score": 0.0,
            "true_value_score": 0.0,
            "true_distance_score": 0.0,
            "num_trials_used": 0.0,
            "noise_std_used": 0.0,
            "error": message,
        },
        artifacts={
            "error_type": err_type,
            "error_message": message,
            "suggestion": suggestion,
        },
    )


def evaluate(program_path):
    noise_std = _env_float("NOISE_STD", 0.0)
    num_trials = _env_int("NUM_TRIALS", 10)
    log_path = os.environ.get("EVAL_LOG_PATH", "")
    noise_seed = _env_int("NOISE_SEED", 0)

    # Fresh RNG per evaluation — mix in program path + time + pid + noise_seed
    rng_seed = (hash((program_path, time.time_ns(), os.getpid(), noise_seed))) & 0xFFFFFFFF
    rng = np.random.default_rng(rng_seed)

    eval_start = time.time()

    try:
        program = _load_program(program_path)
        if not hasattr(program, "run_search"):
            return _error_result(
                "MissingFunction",
                "Program is missing required 'run_search' function",
                "Make sure your program includes a function named 'run_search' that returns (x, y, value) or (x, y)",
            )

        per_trial = []  # list of dicts: {x,y,value,base_score,noisy_score,value_score,distance_score,distance}
        for trial in range(num_trials):
            try:
                result = run_with_timeout(program.run_search, timeout_seconds=5)
                if isinstance(result, tuple):
                    if len(result) == 3:
                        x, y, value = result
                    elif len(result) == 2:
                        x, y = result
                        value = np.sin(x) * np.cos(y) + np.sin(x * y) + (x * x + y * y) / 20.0
                    else:
                        continue
                else:
                    continue
                x = _safe_float(x)
                y = _safe_float(y)
                value = _safe_float(value)
                if any(np.isnan(v) or np.isinf(v) for v in (x, y, value)):
                    continue
                base, vs, ds, dist = _base_score_from_trial(x, y, value)
                eps = float(rng.normal(0.0, noise_std)) if noise_std > 0 else 0.0
                noisy = base + eps
                per_trial.append(
                    {
                        "x": x,
                        "y": y,
                        "value": value,
                        "base_score": base,
                        "noisy_score": noisy,
                        "noise_eps": eps,
                        "value_score": vs,
                        "distance_score": ds,
                        "distance": dist,
                    }
                )
            except TimeoutError:
                continue
            except Exception as e:
                print(f"[evaluator] trial {trial} error: {e}")
                continue

        if not per_trial:
            return _error_result(
                "AllTrialsFailed",
                f"All {num_trials} trials failed",
                "Check for infinite loops, ensure function returns (x,y) or (x,y,value)",
            )

        n = len(per_trial)
        noisy_mean = float(np.mean([t["noisy_score"] for t in per_trial]))
        true_mean = float(np.mean([t["base_score"] for t in per_trial]))
        true_value_score = float(np.mean([t["value_score"] for t in per_trial]))
        true_distance_score = float(np.mean([t["distance_score"] for t in per_trial]))
        avg_distance = float(np.mean([t["distance"] for t in per_trial]))
        reliability = n / float(num_trials)

        metrics = {
            # drives selection (noisy)
            "combined_score": noisy_mean,
            # diagnostics — NOT used as fitness because combined_score is present
            "true_combined_score": true_mean,
            "true_value_score": true_value_score,
            "true_distance_score": true_distance_score,
            "avg_distance": avg_distance,
            "reliability_score": reliability,
            "num_trials_used": float(n),
            "noise_std_used": float(noise_std),
        }

        _jsonl_append(
            log_path,
            {
                "ts": time.time(),
                "pid": os.getpid(),
                "program_path": program_path,
                "program_basename": os.path.basename(program_path),
                "noise_std": noise_std,
                "num_trials": num_trials,
                "n_successful": n,
                "noisy_mean": noisy_mean,
                "true_mean": true_mean,
                "true_value_score": true_value_score,
                "true_distance_score": true_distance_score,
                "avg_distance": avg_distance,
                "eval_seconds": time.time() - eval_start,
                "trials": per_trial,
            },
        )

        return EvaluationResult(
            metrics=metrics,
            artifacts={
                "n_trials": str(n),
                "noise_std": f"{noise_std:.4f}",
                "true_vs_noisy": f"true={true_mean:.4f} noisy={noisy_mean:.4f} gap={noisy_mean - true_mean:+.4f}",
                "avg_distance": f"{avg_distance:.4f}",
            },
        )
    except Exception as e:
        print(f"[evaluator] fatal: {e}")
        print(traceback.format_exc())
        return _error_result(type(e).__name__, str(e), "Check program syntax/imports")


def evaluate_stage1(program_path):
    """Cheap single-trial stage for cascade filtering.

    Uses ONE trial, noise-free — just to reject crashes and obviously bad programs.
    """
    try:
        program = _load_program(program_path)
        if not hasattr(program, "run_search"):
            return _error_result("MissingFunction", "Missing run_search")
        try:
            result = run_with_timeout(program.run_search, timeout_seconds=5)
        except TimeoutError:
            return _error_result("TimeoutError", "Stage 1 timeout")

        if not isinstance(result, tuple) or len(result) not in (2, 3):
            return _error_result("InvalidReturn", f"Bad return shape {type(result)}")
        if len(result) == 3:
            x, y, value = result
        else:
            x, y = result
            value = np.sin(x) * np.cos(y) + np.sin(x * y) + (x * x + y * y) / 20.0
        x, y, value = _safe_float(x), _safe_float(y), _safe_float(value)
        if any(np.isnan(v) or np.isinf(v) for v in (x, y, value)):
            return _error_result("InvalidValues", f"Got x={x} y={y} v={value}")
        base, vs, ds, dist = _base_score_from_trial(x, y, value)
        return EvaluationResult(
            metrics={
                "runs_successfully": 1.0,
                "value_score": vs,
                "distance_score": ds,
                "combined_score": base,
                "true_combined_score": base,
            },
            artifacts={"stage1": f"x={x:.3f} y={y:.3f} v={value:.3f} d={dist:.3f}"},
        )
    except Exception as e:
        return _error_result(type(e).__name__, str(e))


def evaluate_stage2(program_path):
    return evaluate(program_path)
