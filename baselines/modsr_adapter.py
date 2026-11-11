#!/usr/bin/env python3
"""Run MOD-SR (baselines/mod-sr) on prepared SRBench task inputs.

Runs inside the ``modsr`` conda env.  Loads the model once, then for each task
reads ``inputs/task_NNNNNN.npz`` (X_train, y_train) and writes
``predictions/task_NNNNNN.json`` containing every decoded + BFGS-refined
candidate as a sympy string over x0..x{d-1}.  Scoring (GT match, R², frontier)
happens in the meta_sr env (evaluate_modsr_srbench.py score).

Inference mirrors baselines/mod-sr/infer_modsr.py (the "bypassing DFEX"
pipeline): DDIM-50 sampling of top_k=20 sequences, decode, constant-skeleton
BFGS refinement (tools/const_opt.refine).  The encoder sees <=200 training
points (its training regime); refinement uses all training rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

MODSR_ROOT = Path(__file__).resolve().parent / "mod-sr"


def _write_json_atomic(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def _build_model(args):
    import torch
    from parsers import get_parser
    from symbolicregression.envs import build_env
    from symbolicregression.model.modsr_model import MODSRModel
    from symbolicregression.utils import setup_device

    parser = get_parser()
    parser.add_argument("--encoder_type", type=str, default="e2e")
    parser.add_argument("--e2e_checkpoint", type=str)
    parser.add_argument("--snip_checkpoint", type=str)
    parser.add_argument("--model_path", type=str)
    params = parser.parse_args([
        "--encoder_type", "e2e",
        "--e2e_checkpoint", str(MODSR_ROOT / "weights" / "model1.pt"),
        "--snip_checkpoint", str(MODSR_ROOT / "weights" / "snip-10dmax.pth"),
        "--model_path", str(MODSR_ROOT / "weights" / "modsr_10d.pth"),
    ])
    if args.cpu:
        params.cpu = True
    setup_device(params)
    if args.cpu:
        params.device = "cpu"
    params.max_input_dimension = 10
    params.use_fex_encoder = False
    params.use_repa = True
    params.use_negative_constants = False
    if getattr(params, "batch_size_eval", None) is None:
        params.batch_size_eval = 1
    if getattr(params, "eval_size", None) is None:
        params.eval_size = 100

    env = build_env(params)
    model = MODSRModel(params=params, env=env,
                       checkpoint_path=params.e2e_checkpoint, encoder_type="e2e")
    ckpt = torch.load(params.model_path, map_location=params.device)
    if "generator_state_dict" in ckpt:
        model.generator.load_state_dict(ckpt["generator_state_dict"])
        loaded = "generator_state_dict"
    elif "decoder_state_dict" in ckpt:
        model.generator.load_state_dict(ckpt["decoder_state_dict"])
        loaded = "decoder_state_dict"
    else:
        model.load_state_dict(ckpt)
        loaded = "full"
    print(f"Loaded MOD-SR weights ({loaded}) on {params.device}", flush=True)
    model = model.to(params.device)
    model.eval()
    return model, env, params


_VAR_RE = re.compile(r"\bx_(\d+)\b")


def _tree_to_str(env, tree) -> str:
    expr = env.simplifier.tree_to_sympy_expr(tree)
    return _VAR_RE.sub(r"x\1", str(expr))


def _run_task(model, env, params, X, y, top_k, ddim_steps, n_encoder_points, seed):
    import numpy as np
    import torch
    from infer_modsr import decode_token_sequence
    from tools.const_opt import refine
    from symbolicregression.utils import get_model

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    if len(y) > n_encoder_points:
        enc_idx = rng.choice(len(y), n_encoder_points, replace=False)
    else:
        enc_idx = np.arange(len(y))
    x_enc = X[enc_idx].astype(np.float32)
    y_enc = y[enc_idx].astype(np.float32).reshape(-1, 1)
    raw = get_model(model)
    device = params.device
    samples = {
        "x_to_fit": [torch.from_numpy(x_enc).float().to(device)],
        "y_to_fit": [torch.from_numpy(y_enc).float().to(device)],
    }
    t0 = time.monotonic()
    with torch.no_grad():
        tokens, _ = raw.sample(samples, num_samples=top_k, use_ddim=True,
                               ddim_steps=ddim_steps)
    sample_s = time.monotonic() - t0

    trees = []
    for k in range(top_k):
        tree = decode_token_sequence(env, tokens[k])
        if tree is not None:
            trees.append(tree)
    raw_strs = []
    for tree in trees:
        try:
            raw_strs.append(_tree_to_str(env, tree))
        except Exception:
            pass

    # BFGS-refine constants on all training rows (refine() downsamples to 1024).
    t1 = time.monotonic()
    X_ref = X.astype(np.float32)
    y_ref = y.astype(np.float32).reshape(-1, 1)
    refined = refine(env=env, X=X_ref, y=y_ref, candidates=trees, verbose=False) if trees else []
    refine_s = time.monotonic() - t1

    candidates = []
    seen = set()
    for rank, cand in enumerate(refined):
        tree = cand.get("predicted_tree") if isinstance(cand, dict) else None
        if tree is None:
            continue
        try:
            s = _tree_to_str(env, tree)
        except Exception:
            continue
        if s in seen:
            continue
        seen.add(s)
        candidates.append({"equation": s, "rank": rank,
                           "refinement_type": cand.get("refinement_type")})
    return {
        "n_decoded": len(trees),
        "raw_decoded": raw_strs,
        "candidates": candidates,
        "sample_seconds": sample_s,
        "refine_seconds": refine_s,
        "n_encoder_points": int(len(enc_idx)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-dir", required=True)
    p.add_argument("--task-indices", type=int, nargs="*",
                   help="Explicit task indices; default = shard of all tasks")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--n-encoder-points", type=int, default=200)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    tasks = json.loads((batch_dir / "tasks.json").read_text())
    if args.task_indices:
        indices = args.task_indices
    else:
        indices = [i for i in range(len(tasks)) if i % args.n_shards == args.shard]
    pred_dir = batch_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    todo = [i for i in indices
            if args.force or not (pred_dir / f"task_{i:06d}.json").exists()]
    print(f"{len(todo)}/{len(indices)} tasks to run", flush=True)
    if not todo:
        return

    os.chdir(MODSR_ROOT)
    sys.path.insert(0, str(MODSR_ROOT))
    import numpy as np
    model, env, params = _build_model(args)

    for i in todo:
        task = tasks[i]
        out = {"task_index": i, "dataset_name": task["dataset_name"],
               "target_noise": task["target_noise"], "error": None,
               "top_k": args.top_k, "ddim_steps": args.ddim_steps,
               "device": str(params.device)}
        t0 = time.monotonic()
        try:
            data = np.load(batch_dir / "inputs" / f"task_{i:06d}.npz")
            X, y = data["X_train"], data["y_train"]
            if X.shape[1] > 10:
                raise ValueError(f"MOD-SR supports <=10 inputs, got {X.shape[1]}")
            out.update(_run_task(model, env, params, X, y, args.top_k,
                                 args.ddim_steps, args.n_encoder_points,
                                 seed=int(task["seed"])))
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
            out["traceback"] = traceback.format_exc()
        out["predict_seconds"] = time.monotonic() - t0
        _write_json_atomic(pred_dir / f"task_{i:06d}.json", out)
        print(f"[{i}] {task['dataset_name']} noise={task['target_noise']:g} "
              f"cands={len(out.get('candidates', []))} "
              f"t={out['predict_seconds']:.1f}s err={out['error']}", flush=True)


if __name__ == "__main__":
    main()
