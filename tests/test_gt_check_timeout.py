"""Guards on the ground-truth symbolic check's time budgets.

An unbounded GT check used to hang a task past its SLURM array's stall
watchdog, which cancelled the array and zeroed every sibling task
(runs/93799: 3 tasks, plus the whole retry pass).
"""
import signal
import time

import pandas as pd
import pytest

from evaluation import (
    GT_MATCH_TOTAL_TIMEOUT_S,
    _alarm_scope,
    check_pysr_frontier_symbolic_match,
    check_pysr_symbolic_match,
)

# A frontier row that is slow to simplify but not slow to parse.
SLOW_EXPR = "+".join(f"sin(x0*{j}.13)/cos(x1*{j}.7)" for j in range(1, 12))


def _frontier(n_rows):
    return pd.DataFrame([
        {"complexity": i + 1, "loss": 1.0 / (i + 1), "equation": SLOW_EXPR}
        for i in range(n_rows)
    ])


def test_frontier_sweep_stops_at_total_budget():
    t0 = time.time()
    res = check_pysr_frontier_symbolic_match(
        _frontier(12), best_df_index=0, ground_truth_str="x0*x1",
        var_names=["x0", "x1"], timeout_seconds_per_expression=3,
        total_timeout_seconds=5,
    )
    elapsed = time.time() - t0
    assert elapsed < 12, f"budget ignored: {elapsed:.1f}s"
    assert res["match"] is False
    assert res["budget_exhausted"] is True
    assert res["checked_count"] < 12


def test_budget_is_on_by_default():
    assert GT_MATCH_TOTAL_TIMEOUT_S > 0
    res = check_pysr_frontier_symbolic_match(
        _frontier(1), best_df_index=0, ground_truth_str="x0*x1",
        var_names=["x0", "x1"],
    )
    assert res["budget_exhausted"] is False


def test_real_match_still_found():
    df = pd.DataFrame([
        {"complexity": 3, "loss": 1.0, "equation": SLOW_EXPR},
        {"complexity": 4, "loss": 0.1, "equation": "x0*x1 + x0"},
    ])
    res = check_pysr_frontier_symbolic_match(
        df, best_df_index=1, ground_truth_str="x0*x1 + x0",
        var_names=["x0", "x1"], timeout_seconds_per_expression=3,
    )
    assert res["match"] is True


def test_per_expression_timeout_covers_parsing():
    # Parsing used to run before the alarm was armed, so a pathological
    # expression could hang despite timeout_seconds.
    deep = "sin(" * 400 + "x0" + ")" * 400
    t0 = time.time()
    res = check_pysr_symbolic_match(deep, "x0", var_names=["x0"],
                                    timeout_seconds=2)
    assert time.time() - t0 < 15
    assert res["match"] in (False, True)


def test_alarm_scope_preserves_outer_deadline():
    fired = []

    def _outer(_signum, _frame):
        fired.append(True)

    prev = signal.signal(signal.SIGALRM, _outer)
    signal.alarm(5)
    try:
        with _alarm_scope(1, lambda s, f: None):
            time.sleep(1.1)
        remaining = signal.alarm(0)
        assert remaining > 0, "inner scope cancelled the outer wall alarm"
        assert remaining <= 4
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)
    assert not fired


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
