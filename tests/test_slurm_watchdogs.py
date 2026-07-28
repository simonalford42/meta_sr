from types import SimpleNamespace

import pytest

import slurm_eval


def test_pending_time_is_credited_to_both_watchdogs():
    start, progress, previous = slurm_eval._credit_pending_watchdog_time(
        ["PENDING"], now=25.0, previous_poll_time=10.0,
        start_time=0.0, last_progress_time=5.0,
    )

    assert start == 15.0
    assert progress == 20.0
    assert previous == 25.0


def test_completed_jobs_do_not_prevent_pending_credit():
    start, progress, _ = slurm_eval._credit_pending_watchdog_time(
        ["COMPLETED", "PENDING"], now=20.0, previous_poll_time=10.0,
        start_time=0.0, last_progress_time=3.0,
    )

    assert start == 10.0
    assert progress == 13.0


@pytest.mark.parametrize("statuses", [
    ["RUNNING"],
    ["PENDING", "RUNNING"],
    ["UNKNOWN"],
    ["COMPLETED"],
])
def test_active_or_uncertain_time_is_not_credited(statuses):
    start, progress, previous = slurm_eval._credit_pending_watchdog_time(
        statuses, now=20.0, previous_poll_time=10.0,
        start_time=0.0, last_progress_time=3.0,
    )

    assert start == 0.0
    assert progress == 3.0
    assert previous == 20.0


def test_array_status_prefers_running_over_pending(monkeypatch):
    result = SimpleNamespace(
        returncode=0,
        stdout="PENDING\nRUNNING\n",
        stderr="",
    )
    monkeypatch.setattr(slurm_eval.subprocess, "run", lambda *args, **kwargs: result)
    evaluator = SimpleNamespace(_get_slurm_env=lambda: {})

    status = slurm_eval.BaseSlurmEvaluator._get_job_status(evaluator, "123")

    assert status == "RUNNING"
