#!/usr/bin/env python3
"""Smoke test for domains.LSRSynthDomain: data loading, opset, and check_solved
on a short real PySR fit (same call shape as parallel_eval_pysr's worker).

    python scripts/test_lsr_synth_domain.py [lsr:matsci/MATSCI2] [seconds]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from domains import get_domain  # noqa: E402
from lsr_synth import load_problem, metrics, parse_dataset_name  # noqa: E402


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "lsr:matsci/MATSCI2"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20
    dom = get_domain("lsr_synth")
    X_tr, y_tr, X_val, y_val, target = dom.load_train_validation(ds, max_samples=2000, data_seed=7)
    full = load_problem(*parse_dataset_name(ds))
    assert X_tr.shape[0] == 2000 and X_val.shape[0] == 500, (X_tr.shape, X_val.shape)
    assert X_tr.shape[1] == full["train"][0].shape[1]
    # same seed -> same subsample; different seed -> different
    assert np.array_equal(X_tr, dom.load_train_validation(ds, 2000, 7)[0])
    assert not np.array_equal(X_tr, dom.load_train_validation(ds, 2000, 8)[0])
    print("load_train_validation OK", X_tr.shape, X_val.shape, target)

    from parallel_eval_pysr import _import_pysr_regressor
    kw = dom.base_pysr_kwargs()
    print("ops:", kw["binary_operators"], kw["unary_operators"])
    kw.update(timeout_in_seconds=secs, niterations=10_000_000, verbosity=0, progress=False,
              random_state=0, temp_equation_file=False,
              output_directory=os.environ.get("TMPDIR", "/tmp") + "/lsr_domain_test")
    kw.pop("max_evals", None)
    model = _import_pysr_regressor()(**kw)
    model.fit(X_tr, y_tr)
    res = dom.check_solved(
        equations_df=model.equations_, best_df_index=None, target=target, var_names=None,
        predict_fn=lambda idx: model.predict(X_val, index=int(idx)), y_val=y_val,
        predict_on=lambda idx, Xq: model.predict(Xq, index=int(idx)), dataset_name=ds)
    print("check_solved:", res)
    # independent recomputation of the selected equation's score
    i = res["selected_df_index"]
    p_tr = model.predict(full["train"][0], index=i)
    fm, ym = p_tr.mean(), full["train"][1].mean()
    b = np.dot(p_tr - fm, full["train"][1] - ym) / np.dot(p_tr - fm, p_tr - fm)
    m = metrics(ym - b * fm + b * model.predict(X_val, index=i), y_val)
    assert m["acc95_0.01"] == res["acc95_0.01"] and abs(m["nmse"] - res["nmse"]) <= 1e-9 * max(1, m["nmse"])
    print("independent recomputation agrees:", m["nmse"], m["acc95_0.01"])


if __name__ == "__main__":
    main()
