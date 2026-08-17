#!/usr/bin/env python3
"""Fully observable NeuronBench -> scalar symbolic-regression benchmark.

The original benchmark hides Hodgkin--Huxley gating state and asks a solver to
actively select experiments.  This demo removes both difficulties.  For each
of the six deterministic worlds, PySR observes external current, voltage, and
every channel's open fraction, and learns the membrane vector field dV/dt.

The composite open fractions are exactly phi_c in Eq. (32) of the NeuronBench
paper.  Thus the scalar target remains the genuine current-balance dynamics,
but it is an ordinary, noiseless supervised SR task:

    dV/dt = I_ext - sum_c g_c * phi_c * (V - E_c).

Commands:
  generate-data   Materialize deterministic train/test collocation datasets.
  validate        Check the installed package and generated equations/data.
  run-task        Run one method/world/seed PySR fit.
  run-all-local   Run a task matrix in isolated subprocesses (for smoke tests).
  status          Print completion status for the requested matrix.
  report          Build the Markdown summary and multi-page PDF.

The production design is 2 methods x 6 worlds x 3 seeds x 1e6 max_evals.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RESULTS = ROOT / "runs" / "neuronbench_fully_observable"
DEFAULT_REPORT = ROOT / "reports" / "neuronbench_pysr_fully_observable.pdf"
NEURONBENCH_COMMIT = "c354622458c460b419cab821d482c879f0578377"
PAPER_URL = "https://arxiv.org/abs/2608.09696"
REPO_URL = "https://github.com/murphyk/neuronbench"
EVOLVED_SOURCE = "runs/538190"
WORLDS = (
    "z_rebound",
    "h_sag",
    "na_fatigue",
    "ca_rebound",
    "d_type",
    "textbook_M",
)
METHODS = ("baseline", "evolved_538190")
DEFAULT_SEEDS = (0, 1, 2)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    os.replace(tmp, path)


def parse_csv(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_seeds(value: str) -> Tuple[int, ...]:
    return tuple(int(v) for v in parse_csv(value))


def world_spec(world: str) -> Dict[str, Any]:
    """Read channel constants from the installed, pinned NeuronBench package."""
    if world not in WORLDS:
        raise ValueError(f"Unknown world {world!r}; expected one of {WORLDS}")
    from neuronbench.worlds import WORLDS as NB_WORLDS

    nb = NB_WORLDS[world]
    extra = list(nb["alt"].get("extra", ()))
    feature_names = ["i_ext", "v", "phi_na", "phi_k"]
    terms = [
        {"feature": "phi_na", "g": 120.0, "E": 50.0, "channel": "Na"},
        {"feature": "phi_k", "g": 36.0, "E": -77.0, "channel": "K"},
        {"feature": None, "g": 0.3, "E": -54.4, "channel": "leak"},
    ]
    for channel in extra:
        feature = f"phi_{channel.name.lower()}"
        feature_names.append(feature)
        terms.append(
            {
                "feature": feature,
                "g": float(channel.g),
                "E": float(channel.E),
                "channel": str(channel.name),
            }
        )

    # na_fatigue has no extra current. Its phi_na is m_Na^3*h_Na*s, rather
    # than m_Na^3*h_Na. This distinction is retained in the state definition.
    state_definitions = {
        "phi_na": "m_Na^3 * h_Na * s_Na" if world == "na_fatigue" else "m_Na^3 * h_Na",
        "phi_k": "n_K^4",
    }
    for channel in extra:
        activation = f"m_{channel.name}^{int(channel.mpow)}"
        if channel.hvh is not None:
            activation += f" * h_{channel.name}^{int(channel.hpow)}"
        state_definitions[f"phi_{channel.name.lower()}"] = activation

    pieces = ["i_ext"]
    for term in terms:
        drive = f"(v - ({term['E']:.17g}))"
        if term["feature"] is not None:
            drive = f"{term['feature']}*{drive}"
        pieces.append(f"- ({term['g']:.17g})*{drive}")
    ground_truth = " ".join(pieces)

    # PySR node count for this factored expression: each gated current is 7,
    # leak is 5, plus additions, I leaf, and the outer subtraction chain.
    n_gated = 2 + len(extra)
    n_terms = n_gated + 1
    gt_complexity = 7 * n_gated + 5 + (n_terms - 1) + 1 + 1

    return {
        "world": world,
        "description": nb["desc"],
        "feature_names": feature_names,
        "state_definitions": state_definitions,
        "terms": terms,
        "ground_truth": ground_truth,
        "ground_truth_complexity": gt_complexity,
        "neuronbench_commit": NEURONBENCH_COMMIT,
    }


def evaluate_truth(spec: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    names = spec["feature_names"]
    col = {name: X[:, i] for i, name in enumerate(names)}
    y = np.array(col["i_ext"], copy=True)
    v = col["v"]
    for term in spec["terms"]:
        drive = v - float(term["E"])
        if term["feature"] is not None:
            drive = col[term["feature"]] * drive
        y -= float(term["g"]) * drive
    return y


def sample_states(spec: Dict[str, Any], n: int, seed: int) -> np.ndarray:
    """Space-filling independent states over NeuronBench's operating range."""
    from scipy.stats import qmc

    d = len(spec["feature_names"])
    # Scrambled Sobol points give broad, reproducible coverage and break the
    # strong on-trajectory gate correlations that can hide algebraic errors.
    m = int(math.ceil(math.log2(n)))
    unit = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(m=m)[:n]
    X = np.empty_like(unit)
    X[:, 0] = -40.0 + 60.0 * unit[:, 0]  # full protocol/test current range
    X[:, 1] = -95.0 + 155.0 * unit[:, 1]  # clamped simulator voltage range
    X[:, 2:] = unit[:, 2:]                 # open fractions are in [0, 1]
    return X


def generate_data(results_dir: Path, n_train: int, n_test: int, data_seed: int) -> None:
    data_dir = results_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper": PAPER_URL,
        "neuronbench_repo": REPO_URL,
        "neuronbench_commit": NEURONBENCH_COMMIT,
        "n_train": n_train,
        "n_test": n_test,
        "data_seed": data_seed,
        "sampling": {
            "design": "scrambled Sobol collocation states",
            "i_ext": [-40.0, 20.0],
            "v": [-95.0, 60.0],
            "open_fractions": [0.0, 1.0],
        },
        "worlds": {},
    }
    for world_index, world in enumerate(WORLDS):
        spec = world_spec(world)
        X_train = sample_states(spec, n_train, data_seed + 101 * world_index)
        X_test = sample_states(spec, n_test, data_seed + 10_000 + 101 * world_index)
        y_train = evaluate_truth(spec, X_train)
        y_test = evaluate_truth(spec, X_test)
        np.savez_compressed(
            data_dir / f"{world}.npz",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=np.asarray(spec["feature_names"]),
        )
        manifest["worlds"][world] = spec
        print(
            f"{world:12s}: train={X_train.shape}, test={X_test.shape}, "
            f"RMS(dV/dt)={np.sqrt(np.mean(y_test**2)):.3f}"
        )
    write_json(data_dir / "manifest.json", manifest)
    print(f"Wrote {data_dir / 'manifest.json'}")


def ensure_data(results_dir: Path, n_train: int, n_test: int, data_seed: int) -> None:
    manifest_path = results_dir / "data" / "manifest.json"
    if not manifest_path.exists():
        generate_data(results_dir, n_train, n_test, data_seed)


def load_data(results_dir: Path, world: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    with open(results_dir / "data" / "manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with np.load(results_dir / "data" / f"{world}.npz") as arrays:
        data = {key: np.array(arrays[key]) for key in arrays.files}
    return manifest["worlds"][world], data


def task_dir(results_dir: Path, method: str, world: str, seed: int, max_evals: int) -> Path:
    return results_dir / "fits" / f"evals_{max_evals}" / method / world / f"seed_{seed}"


def task_matrix(methods: Sequence[str], worlds: Sequence[str], seeds: Sequence[int]) -> List[Tuple[str, str, int]]:
    return [(method, world, seed) for method in methods for world in worlds for seed in seeds]


def _move_existing_to_trash(path: Path) -> None:
    if not path.exists():
        return
    trash = Path.home() / "trash"
    trash.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = trash / f"{path.name}_{stamp}_{os.getpid()}"
    path.rename(destination)
    print(f"Moved existing task directory to {destination}")


def _load_evolved_config() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from bundle_loader import load_bundle

    bundle = load_bundle(str(ROOT / EVOLVED_SOURCE), select_by="train")
    config = bundle.to_pysr_config({})
    metadata = {
        "source": EVOLVED_SOURCE,
        "train_score": bundle.score,
        "operators": {
            name: (operator.name if operator is not None else None)
            for name, operator in bundle.operators.items()
        },
    }
    return {
        "mutation_weights": config.mutation_weights,
        "pysr_overrides": config.pysr_kwargs,
        "custom_mutation_code": config.custom_mutation_code,
        "custom_survival_code": config.custom_survival_code,
        "custom_selection_code": config.custom_selection_code,
        "custom_loss_code": config.custom_loss_code,
    }, metadata


def _activate_evolved_operators(config: Dict[str, Any]) -> None:
    from parallel_eval_pysr import (
        _load_dynamic_loss,
        _load_dynamic_mutations,
        _load_dynamic_selection,
        _load_dynamic_survival,
    )

    if config["custom_mutation_code"]:
        _load_dynamic_mutations(config["custom_mutation_code"])
    if config["custom_survival_code"]:
        _load_dynamic_survival(config["custom_survival_code"])
    if config["custom_selection_code"]:
        _load_dynamic_selection(config["custom_selection_code"])
    if config["custom_loss_code"]:
        _load_dynamic_loss(config["custom_loss_code"])


def _expression_string(row: Any) -> str:
    value = row.get("sympy_format") if hasattr(row, "get") else None
    if value is None or str(value) in ("", "nan", "None"):
        value = row.get("equation")
    return str(value)


def _symbolic_exact(expression: str, ground_truth: str, feature_names: Sequence[str]) -> bool:
    try:
        import sympy as sp

        symbols = {name: sp.Symbol(name) for name in feature_names}
        found = sp.sympify(expression, locals=symbols)
        truth = sp.sympify(ground_truth, locals=symbols)
        return bool(sp.simplify(found - truth) == 0)
    except Exception:
        return False


def classify_nrmse(nrmse: float) -> str:
    if nrmse <= 1e-6:
        return "recovered"
    if nrmse <= 1e-3:
        return "near-exact"
    if nrmse <= 5e-2:
        return "close"
    return "miss"


def affine_calibration(prediction: np.ndarray, target: np.ndarray) -> Tuple[float, float, str]:
    """Mirror 538190's OLS/sign-covariance affine shape corrections."""
    p = np.asarray(prediction, dtype=float).reshape(-1)
    y = np.asarray(target, dtype=float).reshape(-1)
    mean_p, mean_y = float(np.mean(p)), float(np.mean(y))
    cp, cy = p - mean_p, y - mean_y
    spp = float(np.dot(cp, cp))
    a_ols = float(np.dot(cp, cy) / spp) if spp > np.finfo(float).eps else 0.0
    b_ols = mean_y - a_ols * mean_p
    mad_p = float(np.sum(np.abs(cp)))
    a_robust = float(np.dot(np.sign(cp), cy) / mad_p) if mad_p > np.finfo(float).eps else 0.0
    b_robust = mean_y - a_robust * mean_p
    mae_ols = float(np.mean(np.abs(a_ols * p + b_ols - y)))
    mae_robust = float(np.mean(np.abs(a_robust * p + b_robust - y)))
    if mae_robust < mae_ols:
        return a_robust, b_robust, "robust_sign_covariance"
    return a_ols, b_ols, "ols"


def run_task(
    results_dir: Path,
    method: str,
    world: str,
    seed: int,
    max_evals: int,
    force: bool,
    n_train: int,
    n_test: int,
    data_seed: int,
) -> None:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected {METHODS}")
    ensure_data(results_dir, n_train, n_test, data_seed)
    spec, data = load_data(results_dir, world)
    destination = task_dir(results_dir, method, world, seed, max_evals)
    result_path = destination / "result.json"
    if result_path.exists() and not force:
        print(f"Already complete: {result_path}")
        return
    if destination.exists():
        if not force:
            raise FileExistsError(f"Incomplete task directory exists: {destination}; use --force")
        _move_existing_to_trash(destination)
    destination.mkdir(parents=True, exist_ok=True)

    from julia_env import configure_juliapkg_project

    os.environ["PYTHON_JULIAPKG_PROJECT"] = str((ROOT / ".juliapkg_env").resolve())
    configure_juliapkg_project(ROOT)

    evolved_config: Dict[str, Any] | None = None
    method_metadata: Dict[str, Any] = {"source": "PySR defaults"}
    if method == "evolved_538190":
        evolved_config, method_metadata = _load_evolved_config()

    # Import only after the checkout-local Julia environment is pinned.
    from pysr import PySRRegressor

    if evolved_config is not None:
        _activate_evolved_operators(evolved_config)

    model_kwargs: Dict[str, Any] = {
        "binary_operators": ["+", "-", "*"],
        "unary_operators": [],
        "maxsize": 35,
        "maxdepth": 16,
        "populations": 15,
        "population_size": 33,
        "niterations": 10_000_000,
        "max_evals": int(max_evals),
        "parallelism": "serial",
        "procs": 0,
        "deterministic": True,
        "batching": False,
        "precision": 64,
        "random_state": int(seed),
        "early_stop_condition": "stop_if(loss, complexity) = loss < 1e-24",
        "verbosity": 1,
        "progress": False,
        "temp_equation_file": False,
        "delete_tempfiles": True,
        "output_directory": str(destination / "pysr_output"),
    }
    if evolved_config is not None:
        # 538190 has no HPO overrides, but preserve this ordering if the saved
        # artifact is enriched later. Search hyperparameters remain identical.
        model_kwargs.update(evolved_config["pysr_overrides"])
        model_kwargs.update(evolved_config["mutation_weights"])

    run_manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "method_metadata": method_metadata,
        "world": world,
        "seed": seed,
        "max_evals": max_evals,
        "model_kwargs": model_kwargs,
        "ground_truth": spec,
        "data_manifest": str(results_dir / "data" / "manifest.json"),
    }
    write_json(destination / "run_manifest.json", run_manifest)

    X_train = np.asarray(data["X_train"], dtype=np.float64)
    y_train_physical = np.asarray(data["y_train"], dtype=np.float64)
    X_test = np.asarray(data["X_test"], dtype=np.float64)
    y_test = np.asarray(data["y_test"], dtype=np.float64)
    feature_names = [str(v) for v in data["feature_names"]]
    # Scaling only the target is an invertible, fixed preprocessing step. It
    # prevents 10^3--10^4 physical derivatives from making constant search
    # needlessly brittle; all reported predictions/equations are converted
    # back to physical dV/dt units.
    target_scale = float(np.sqrt(np.mean(y_train_physical**2)))
    y_train = y_train_physical / target_scale

    print(
        f"Running {method} / {world} / seed {seed}: "
        f"{len(y_train)} train, {max_evals} max evals",
        flush=True,
    )
    start = time.time()
    model = PySRRegressor(**model_kwargs)
    model.fit(X_train, y_train, variable_names=feature_names)
    runtime = time.time() - start

    equations = model.equations_.copy().sort_values(["complexity", "loss"])
    equations.to_csv(destination / "pareto_frontier.csv", index=True, index_label="pysr_index")
    denom = float(np.sqrt(np.mean(y_test**2)))
    frontier: List[Dict[str, Any]] = []
    for index, row in equations.iterrows():
        record: Dict[str, Any] = {
            "pysr_index": int(index),
            "complexity": int(row["complexity"]),
            "train_loss": float(row["loss"]),
            "score": float(row.get("score", float("nan"))),
            "equation": str(row.get("equation")),
            "sympy": _expression_string(row),
        }
        try:
            pred_scaled = np.asarray(model.predict(X_test, index=int(index)), dtype=float).reshape(-1)
            pred = target_scale * pred_scaled
            finite = pred.shape == y_test.shape and np.all(np.isfinite(pred))
            if not finite:
                raise ValueError("non-finite or wrong-shape prediction")
            rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
            record["test_rmse"] = rmse
            record["test_nrmse"] = rmse / max(denom, np.finfo(float).tiny)
            ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
            record["test_r2"] = 1.0 - float(np.sum((pred - y_test) ** 2)) / ss_tot

            train_pred_scaled = np.asarray(
                model.predict(X_train, index=int(index)), dtype=float
            ).reshape(-1)
            a, b, calibration_kind = affine_calibration(train_pred_scaled, y_train)
            calibrated = target_scale * (a * pred_scaled + b)
            calibrated_rmse = float(np.sqrt(np.mean((calibrated - y_test) ** 2)))
            record["affine_calibration"] = {"a": a, "b": b, "kind": calibration_kind}
            record["calibrated_test_nrmse"] = calibrated_rmse / max(
                denom, np.finfo(float).tiny
            )
            record["calibrated_test_r2"] = 1.0 - float(
                np.sum((calibrated - y_test) ** 2)
            ) / ss_tot
        except Exception as exc:
            record["prediction_error"] = str(exc)
            record["test_rmse"] = float("inf")
            record["test_nrmse"] = float("inf")
            record["test_r2"] = float("-inf")
            record["calibrated_test_nrmse"] = float("inf")
            record["calibrated_test_r2"] = float("-inf")
        record["physical_sympy"] = f"({target_scale:.17g})*({record['sympy']})"
        record["symbolic_exact"] = _symbolic_exact(
            record["physical_sympy"], spec["ground_truth"], feature_names
        )
        frontier.append(record)

    finite_frontier = [row for row in frontier if math.isfinite(row["test_nrmse"])]
    if not finite_frontier:
        raise RuntimeError("No finite Pareto-frontier predictions")
    best = min(finite_frontier, key=lambda row: row["test_nrmse"])
    gt_complexity = int(spec["ground_truth_complexity"])
    within_gt = [row for row in finite_frontier if row["complexity"] <= gt_complexity]
    best_within_gt = min(within_gt, key=lambda row: row["test_nrmse"]) if within_gt else None
    result = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "method_metadata": method_metadata,
        "world": world,
        "seed": seed,
        "max_evals": max_evals,
        "runtime_seconds": runtime,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "feature_names": feature_names,
        "ground_truth": spec["ground_truth"],
        "ground_truth_complexity": gt_complexity,
        "target_scale": target_scale,
        "target_preprocessing": "fit dV/dt divided by train RMS; multiply predictions by this scale",
        "test_target_rms": denom,
        "best_frontier": best,
        "best_within_ground_truth_complexity": best_within_gt,
        "assessment": classify_nrmse(float(best["test_nrmse"])),
        "any_symbolic_exact": any(row["symbolic_exact"] for row in frontier),
        "frontier": frontier,
    }
    write_json(result_path, result)
    run_manifest["status"] = "complete"
    run_manifest["runtime_seconds"] = runtime
    write_json(destination / "run_manifest.json", run_manifest)
    print(
        f"Complete in {runtime:.1f}s: best NRMSE={best['test_nrmse']:.3e}, "
        f"complexity={best['complexity']}, assessment={result['assessment']}"
    )


def run_all_local(args: argparse.Namespace) -> None:
    matrix = task_matrix(args.methods, args.worlds, args.seeds)
    for task_index, (method, world, seed) in enumerate(matrix):
        print(f"\n[{task_index + 1}/{len(matrix)}] {method} {world} seed={seed}", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-task",
            "--results-dir", str(args.results_dir),
            "--method", method,
            "--world", world,
            "--seed", str(seed),
            "--max-evals", str(args.max_evals),
            "--n-train", str(args.n_train),
            "--n-test", str(args.n_test),
            "--data-seed", str(args.data_seed),
        ]
        if args.force:
            command.append("--force")
        subprocess.run(command, cwd=ROOT, check=True)


def select_task_by_index(args: argparse.Namespace) -> Tuple[str, str, int]:
    matrix = task_matrix(args.methods, args.worlds, args.seeds)
    index = args.task_index
    if index is None:
        env_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_index is None:
            raise ValueError("Provide --task-index or SLURM_ARRAY_TASK_ID")
        index = int(env_index)
    if not 0 <= index < len(matrix):
        raise IndexError(f"task index {index} outside [0, {len(matrix) - 1}]")
    return matrix[index]


def load_results(
    results_dir: Path,
    methods: Sequence[str],
    worlds: Sequence[str],
    seeds: Sequence[int],
    max_evals: int,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, int]]]:
    found: List[Dict[str, Any]] = []
    missing: List[Tuple[str, str, int]] = []
    for method, world, seed in task_matrix(methods, worlds, seeds):
        path = task_dir(results_dir, method, world, seed, max_evals) / "result.json"
        if not path.exists():
            missing.append((method, world, seed))
            continue
        with open(path, encoding="utf-8") as handle:
            found.append(json.load(handle))
    return found, missing


def print_status(args: argparse.Namespace) -> None:
    found, missing = load_results(
        args.results_dir, args.methods, args.worlds, args.seeds, args.max_evals
    )
    expected = len(found) + len(missing)
    print(f"complete: {len(found)}/{expected}")
    if found:
        for result in sorted(found, key=lambda r: (r["method"], r["world"], r["seed"])):
            best = result["best_frontier"]
            print(
                f"  {result['method']:16s} {result['world']:12s} seed={result['seed']} "
                f"NRMSE={best['test_nrmse']:.3e} c={best['complexity']:2d} "
                f"{result['assessment']}"
            )
    if missing:
        print("missing:")
        for method, world, seed in missing:
            print(f"  {method:16s} {world:12s} seed={seed}")


def _fmt_equation(value: str, width: int = 92) -> str:
    return "\n".join(textwrap.wrap(value.replace("**", "^"), width=width))


def _aggregate(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, List[float]] = {method: [] for method in METHODS}
    calibrated_by_method: Dict[str, List[float]] = {method: [] for method in METHODS}
    solve_counts = {method: 0 for method in METHODS}
    exact_counts = {method: 0 for method in METHODS}
    for result in results:
        method = result["method"]
        nrmse = float(result["best_frontier"]["test_nrmse"])
        by_method.setdefault(method, []).append(nrmse)
        calibrated_by_method.setdefault(method, []).append(
            min(float(row.get("calibrated_test_nrmse", float("inf"))) for row in result["frontier"])
        )
        solve_counts[method] = solve_counts.get(method, 0) + int(nrmse <= 1e-6)
        exact_counts[method] = exact_counts.get(method, 0) + int(result["any_symbolic_exact"])
    aggregate: Dict[str, Any] = {
        "n_results": len(results),
        "solve_threshold_nrmse": 1e-6,
        "methods": {},
    }
    for method, values in by_method.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        calibrated = np.asarray(calibrated_by_method[method], dtype=float)
        aggregate["methods"][method] = {
            "n": len(values),
            "median_nrmse": float(np.median(arr)),
            "geometric_mean_nrmse": float(np.exp(np.mean(np.log(np.maximum(arr, 1e-16))))),
            "median_affine_calibrated_nrmse": float(np.median(calibrated)),
            "recovered": solve_counts[method],
            "symbolic_exact": exact_counts[method],
        }

    pairs: List[Tuple[float, float]] = []
    index = {(r["world"], int(r["seed"]), r["method"]): r for r in results}
    for world in WORLDS:
        for seed in DEFAULT_SEEDS:
            b = index.get((world, seed, "baseline"))
            e = index.get((world, seed, "evolved_538190"))
            if b and e:
                pairs.append(
                    (
                        float(b["best_frontier"]["test_nrmse"]),
                        float(e["best_frontier"]["test_nrmse"]),
                    )
                )
    if pairs:
        ratios = np.asarray([e / max(b, 1e-16) for b, e in pairs])
        aggregate["paired"] = {
            "n": len(pairs),
            "evolved_wins": int(sum(e < b for b, e in pairs)),
            "baseline_wins": int(sum(b < e for b, e in pairs)),
            "ties": int(sum(b == e for b, e in pairs)),
            "median_evolved_over_baseline_nrmse": float(np.median(ratios)),
        }
    return aggregate


def build_report(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    results, missing = load_results(
        args.results_dir, args.methods, args.worlds, args.seeds, args.max_evals
    )
    if missing and not args.allow_incomplete:
        formatted = ", ".join(f"{m}/{w}/{s}" for m, w, s in missing[:8])
        raise RuntimeError(
            f"Cannot build final report: {len(missing)} tasks missing ({formatted}). "
            "Use --allow-incomplete only for a smoke-test report."
        )
    if not results:
        raise RuntimeError("No completed results found")

    output_pdf = args.output_pdf
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_md = output_pdf.with_suffix(".md")
    aggregate = _aggregate(results)
    result_index = {(r["method"], r["world"], int(r["seed"])): r for r in results}
    method_colors = {"baseline": "#2878B5", "evolved_538190": "#E07B39"}
    method_labels = {"baseline": "vanilla PySR", "evolved_538190": "evolved PySR (538190)"}

    with PdfPages(output_pdf) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.94, "PySR on Fully Observable NeuronBench", fontsize=22, weight="bold")
        fig.text(
            0.08,
            0.895,
            f"Six deterministic worlds · 3 seeds · max_evals={args.max_evals:,} · generated {datetime.now().date()}",
            fontsize=11,
        )
        intro = (
            "Reduction. The original task combines active experiment design with hidden-state dynamical "
            "system discovery. Here each scalar SR problem exposes I_ext, membrane voltage V, and every "
            "channel open fraction phi_c. PySR learns the exact current-balance vector field "
            "dV/dt = I_ext - sum_c g_c phi_c(V-E_c). Training and test points are independent scrambled "
            "Sobol collocation states spanning the benchmark's current/voltage range. No observation noise, "
            "numerical differentiation, active learning, or latent-state inference remains.\n\n"
            "Search. Both methods use precisely the same data, seeds, +/−/* function set, 15×33 populations, "
            "max size 35, float64 arithmetic, and evaluation cap. The evolved condition replaces PySR's "
            "mutation, selection, survival, and loss with the best training-score bundle from run 538190. "
            "The target is divided by its training RMS during search and exactly unscaled for all reported "
            "physical-unit errors.\n\n"
            "Recovery rule. A frontier counts as recovered when at least one equation has held-out global "
            "NRMSE <= 1e-6; near-exact is <=1e-3 and close is <=0.05. Exact symbolic equality is reported "
            "separately and is stricter because fitted floating constants rarely simplify identically. "
            "Affine-calibrated NRMSE is a secondary shape metric mirroring 538190's custom loss; it does not "
            "count as full dynamics recovery because incorrect scale/offset remain incorrect dynamics."
        )
        fig.text(0.08, 0.84, intro, fontsize=10.3, va="top", wrap=True, linespacing=1.4)

        y = 0.53
        fig.text(0.08, y, "Aggregate result", fontsize=15, weight="bold")
        y -= 0.04
        headers = ["method", "runs", "recovered", "exact", "median raw", "median calibrated"]
        rows = []
        for method in METHODS:
            values = aggregate["methods"].get(method)
            if not values:
                continue
            rows.append(
                [
                    method_labels[method],
                    values["n"],
                    values["recovered"],
                    values["symbolic_exact"],
                    f"{values['median_nrmse']:.2e}",
                    f"{values['median_affine_calibrated_nrmse']:.2e}",
                ]
            )
        table = plt.table(cellText=rows, colLabels=headers, bbox=[0.08, y - 0.12, 0.84, 0.12], cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        paired = aggregate.get("paired")
        if paired:
            verdict = (
                f"Across {paired['n']} paired runs, evolved wins {paired['evolved_wins']}, baseline wins "
                f"{paired['baseline_wins']}, ties {paired['ties']}; median evolved/baseline NRMSE ratio "
                f"= {paired['median_evolved_over_baseline_nrmse']:.3g} (<1 favors evolved)."
            )
            fig.text(0.08, y - 0.17, verdict, fontsize=10, va="top", wrap=True)
        if missing:
            fig.text(
                0.08, 0.17, f"INCOMPLETE SMOKE REPORT: {len(missing)} requested runs are missing.",
                fontsize=12, color="darkred", weight="bold",
            )
        fig.text(
            0.08, 0.08,
            f"Sources: {PAPER_URL}\n{REPO_URL} @ {NEURONBENCH_COMMIT}\n"
            f"Evolved algorithm: {EVOLVED_SOURCE}",
            fontsize=8.5,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for world in args.worlds:
            available = [r for r in results if r["world"] == world]
            if not available:
                continue
            spec = world_spec(world)
            fig = plt.figure(figsize=(8.5, 11))
            grid = fig.add_gridspec(4, 1, height_ratios=[0.9, 2.35, 2.35, 3.0], hspace=0.38)
            ax_title = fig.add_subplot(grid[0]); ax_title.axis("off")
            ax_title.text(0, 0.95, world, fontsize=20, weight="bold", va="top")
            ax_title.text(0, 0.62, spec["description"], fontsize=10.5, va="top", wrap=True)
            ax_title.text(
                0, 0.23,
                f"Ground truth (reference complexity {spec['ground_truth_complexity']}):\n"
                + _fmt_equation(spec["ground_truth"], 110),
                fontsize=8.2, family="monospace", va="top",
            )

            ax = fig.add_subplot(grid[1])
            for method in args.methods:
                for seed in args.seeds:
                    result = result_index.get((method, world, seed))
                    if result is None:
                        continue
                    front = [row for row in result["frontier"] if math.isfinite(row["test_nrmse"])]
                    front.sort(key=lambda row: row["complexity"])
                    ax.plot(
                        [row["complexity"] for row in front],
                        [max(float(row["test_nrmse"]), 1e-16) for row in front],
                        marker="o", markersize=2.8, linewidth=1.0, alpha=0.72,
                        color=method_colors[method],
                        label=method_labels[method] if seed == args.seeds[0] else None,
                    )
            ax.axvline(spec["ground_truth_complexity"], color="black", linestyle="--", linewidth=1, label="GT complexity")
            ax.axhline(1e-6, color="#2E8B57", linestyle=":", linewidth=1, label="recovery threshold")
            ax.axhline(1e-3, color="gray", linestyle=":", linewidth=0.8)
            ax.set_yscale("log")
            ax.set_xlabel("PySR complexity")
            ax.set_ylabel("held-out vector-field NRMSE (lower is better)")
            ax.set_title("Discovered Pareto frontiers (all seeds)")
            ax.grid(True, which="both", alpha=0.2)
            ax.legend(fontsize=8, ncol=2)

            ax_cal = fig.add_subplot(grid[2])
            for method in args.methods:
                for seed in args.seeds:
                    result = result_index.get((method, world, seed))
                    if result is None:
                        continue
                    front = [
                        row for row in result["frontier"]
                        if math.isfinite(float(row.get("calibrated_test_nrmse", float("inf"))))
                    ]
                    front.sort(key=lambda row: row["complexity"])
                    ax_cal.plot(
                        [row["complexity"] for row in front],
                        [max(float(row["calibrated_test_nrmse"]), 1e-16) for row in front],
                        marker="o", markersize=2.5, linewidth=0.9, alpha=0.72,
                        color=method_colors[method],
                        label=method_labels[method] if seed == args.seeds[0] else None,
                    )
            ax_cal.axvline(spec["ground_truth_complexity"], color="black", linestyle="--", linewidth=1)
            ax_cal.axhline(1e-6, color="#2E8B57", linestyle=":", linewidth=1)
            ax_cal.set_yscale("log")
            ax_cal.set_xlabel("PySR complexity")
            ax_cal.set_ylabel("affine-calibrated NRMSE")
            ax_cal.set_title("Shape recovery after train-fitted affine calibration (secondary)")
            ax_cal.grid(True, which="both", alpha=0.2)
            ax_cal.legend(fontsize=8, ncol=2)

            ax_text = fig.add_subplot(grid[3]); ax_text.axis("off")
            ypos = 1.0
            for method in args.methods:
                ax_text.text(0, ypos, method_labels[method], color=method_colors[method], fontsize=12, weight="bold", va="top")
                ypos -= 0.07
                for seed in args.seeds:
                    result = result_index.get((method, world, seed))
                    if result is None:
                        ax_text.text(0.02, ypos, f"seed {seed}: missing", fontsize=8.5, color="darkred", va="top")
                        ypos -= 0.07
                        continue
                    best = result["best_frontier"]
                    summary = (
                        f"seed {seed}: {result['assessment']}; NRMSE={best['test_nrmse']:.3e}, "
                        f"R²={best['test_r2']:.8f}, complexity={best['complexity']}, "
                        f"best calibrated={min(float(r.get('calibrated_test_nrmse', float('inf'))) for r in result['frontier']):.3e}, "
                        f"exact={result['any_symbolic_exact']}, runtime={result['runtime_seconds']:.0f}s"
                    )
                    ax_text.text(0.02, ypos, summary, fontsize=8.5, va="top")
                    ypos -= 0.05
                    equation = _fmt_equation(best.get("physical_sympy", best["sympy"]), 105)
                    ax_text.text(0.04, ypos, equation, fontsize=7.2, family="monospace", va="top")
                    ypos -= 0.045 * max(1, equation.count("\n") + 1) + 0.035
                ypos -= 0.035
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Final paired comparison page.
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        xs, ys, labels = [], [], []
        for world in args.worlds:
            for seed in args.seeds:
                b = result_index.get(("baseline", world, seed))
                e = result_index.get(("evolved_538190", world, seed))
                if b is None or e is None:
                    continue
                xs.append(max(float(b["best_frontier"]["test_nrmse"]), 1e-16))
                ys.append(max(float(e["best_frontier"]["test_nrmse"]), 1e-16))
                labels.append(f"{world}:{seed}")
        if xs:
            ax.scatter(xs, ys, s=45, color="#6A4C93", alpha=0.8)
            low = min(xs + ys) / 2
            high = max(xs + ys) * 2
            ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
            for x, yv, label in zip(xs, ys, labels):
                ax.annotate(label, (x, yv), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(low, high); ax.set_ylim(low, high)
        ax.set_xlabel("vanilla PySR best-frontier NRMSE")
        ax.set_ylabel("evolved 538190 best-frontier NRMSE")
        ax.set_title("Paired comparison: points below diagonal favor evolved PySR")
        ax.grid(True, which="both", alpha=0.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    md_lines = [
        "# PySR on fully observable NeuronBench",
        "",
        f"PDF: `{output_pdf}`",
        "",
        f"Protocol: {len(args.worlds)} worlds × {len(args.methods)} methods × {len(args.seeds)} seeds, "
        f"max_evals={args.max_evals:,}.",
        "",
        "| method | runs | recovered | exact symbolic | median raw NRMSE | median affine-calibrated NRMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in args.methods:
        values = aggregate["methods"].get(method)
        if values:
            md_lines.append(
                f"| {method_labels[method]} | {values['n']} | {values['recovered']} | "
                f"{values['symbolic_exact']} | {values['median_nrmse']:.3e} | "
                f"{values['median_affine_calibrated_nrmse']:.3e} |"
            )
    if aggregate.get("paired"):
        p = aggregate["paired"]
        md_lines.extend([
            "",
            f"Paired: evolved wins {p['evolved_wins']}/{p['n']}; baseline wins "
            f"{p['baseline_wins']}/{p['n']}; median evolved/baseline NRMSE ratio "
            f"{p['median_evolved_over_baseline_nrmse']:.3g}.",
        ])
    if missing:
        md_lines.extend(["", f"**Incomplete:** {len(missing)} requested task(s) missing."])
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    write_json(output_pdf.with_suffix(".summary.json"), {"aggregate": aggregate, "missing": missing})
    print(f"Wrote {output_pdf}")
    print(f"Wrote {output_md}")


def validate(results_dir: Path, n_train: int, n_test: int, data_seed: int) -> None:
    import importlib.metadata

    import neuronbench
    from neuronbench.worlds import WORLDS as NB_WORLDS

    ensure_data(results_dir, n_train, n_test, data_seed)
    assert tuple(NB_WORLDS) == WORLDS, (tuple(NB_WORLDS), WORLDS)
    package_path = Path(neuronbench.__file__).resolve()
    direct_url_text = importlib.metadata.distribution("neuronbench").read_text("direct_url.json") or ""
    assert NEURONBENCH_COMMIT in direct_url_text, direct_url_text
    for world in WORLDS:
        spec, data = load_data(results_dir, world)
        y = evaluate_truth(spec, np.asarray(data["X_test"]))
        np.testing.assert_allclose(y, data["y_test"], rtol=0, atol=0)
        assert spec["ground_truth_complexity"] in (23, 31)
        assert np.all(np.isfinite(y))
    print(f"PASS: NeuronBench imports from {package_path}")
    print(f"PASS: installed distribution is pinned to {NEURONBENCH_COMMIT}")
    print("PASS: six installed world definitions match the demo registry")
    print("PASS: all generated targets exactly match their saved ground-truth vector fields")


def add_matrix_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--methods", type=parse_csv, default=METHODS)
    parser.add_argument("--worlds", type=parse_csv, default=WORLDS)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--max-evals", type=int, default=1_000_000)


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=16384)
    parser.add_argument("--data-seed", type=int, default=260809696)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-data", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_data_args(p)

    p = sub.add_parser("validate", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_data_args(p)

    p = sub.add_parser("run-task", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_data_args(p)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--world", choices=WORLDS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-evals", type=int, default=1_000_000)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("run-array-task", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_data_args(p); add_matrix_args(p)
    p.add_argument("--task-index", type=int, default=None)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("run-all-local", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_data_args(p); add_matrix_args(p)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("status", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    add_matrix_args(p)

    p = sub.add_parser("report", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    add_matrix_args(p)
    p.add_argument("--output-pdf", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate-data":
        generate_data(args.results_dir, args.n_train, args.n_test, args.data_seed)
    elif args.command == "validate":
        validate(args.results_dir, args.n_train, args.n_test, args.data_seed)
    elif args.command == "run-task":
        run_task(
            args.results_dir, args.method, args.world, args.seed, args.max_evals,
            args.force, args.n_train, args.n_test, args.data_seed,
        )
    elif args.command == "run-array-task":
        method, world, seed = select_task_by_index(args)
        run_task(
            args.results_dir, method, world, seed, args.max_evals,
            args.force, args.n_train, args.n_test, args.data_seed,
        )
    elif args.command == "run-all-local":
        run_all_local(args)
    elif args.command == "status":
        print_status(args)
    elif args.command == "report":
        build_report(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
