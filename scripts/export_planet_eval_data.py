#!/usr/bin/env python3
"""Export the fixed 24880 f2/mse planet eval cache.

Run from the meta_sr repo with the planet environment active:
    conda activate new_bnn
    python scripts/export_planet_eval_data.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


META_ROOT = Path(__file__).resolve().parents[1]
PLANET_ROOT = META_ROOT.parent / "planet_eqs"
OUTPUT_PATH = META_ROOT / "planet_eval_data.pkl"

NN_VERSION = 24880
TARGET = "f2"
LOSS_FN = "mse"
TIME_IN_HOURS = 8
NITERATIONS = 500000
MAX_SIZE = 30
N = 10000
BATCH_SIZE = 1000


def main() -> None:
    if str(PLANET_ROOT) not in sys.path:
        sys.path.insert(0, str(PLANET_ROOT))
    os.chdir(PLANET_ROOT)

    import torch
    import sr as planet_sr
    import spock_reg_model

    sr_args = argparse.Namespace(
        no_log=True,
        nn_version=NN_VERSION,
        version=0,
        time_in_hours=TIME_IN_HOURS,
        niterations=NITERATIONS,
        max_size=MAX_SIZE,
        seed=0,
        target=TARGET,
        residual=False,
        n=N,
        batch_size=BATCH_SIZE,
        sr_residual=False,
        loss_fn=LOSS_FN,
    )
    sr_config = planet_sr.get_config(sr_args)
    X_train, y_train, variable_names, nn_std_arr = planet_sr.load_inputs_and_targets(sr_config)

    model = spock_reg_model.load(version=NN_VERSION)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.make_dataloaders(train=True, plot=True)

    summaries: List[np.ndarray] = []
    truths: List[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in model._val_dataloader:
            out = model(
                x_batch.to(device),
                noisy_val=False,
                deterministic=True,
                return_intermediates=True,
            )
            summaries.append(out["summary_stats"].detach().cpu().numpy())
            truths.append(y_batch.detach().cpu().numpy())

    X_train_np = np.asarray(X_train, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)
    X_test_np = np.concatenate(summaries, axis=0).astype(np.float32)
    y_test_np = np.concatenate(truths, axis=0).astype(np.float32)

    payload: Dict[str, Any] = {
        "meta": {
            "nn_version": NN_VERSION,
            "target": TARGET,
            "loss_fn": LOSS_FN,
            "n": N,
            "batch_size": BATCH_SIZE,
            "n_train": int(X_train_np.shape[0]),
            "n_test": int(X_test_np.shape[0]),
            "n_features": int(X_train_np.shape[1]),
            "variable_names": list(variable_names),
            "train_shape": list(X_train_np.shape),
            "target_shape": list(y_train_np.shape),
            "test_shape": list(X_test_np.shape),
            "test_truth_shape": list(y_test_np.shape),
        },
        "X_train": X_train_np,
        "y_train": y_train_np,
        "X_test": X_test_np,
        "y_test": y_test_np,
        "variable_names": list(variable_names),
        "nn_std_arr": (
            np.asarray(nn_std_arr, dtype=np.float32)
            if nn_std_arr is not None
            else np.asarray([], dtype=np.float32)
        ),
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
