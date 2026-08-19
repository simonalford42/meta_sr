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


def test_terminal_result_visibility_rechecks_before_sleep(monkeypatch):
    counts = iter([1])
    monkeypatch.setattr(
        slurm_eval.time,
        "sleep",
        lambda *_: pytest.fail("result was visible on the immediate recount"),
    )

    completed = slurm_eval._wait_for_terminal_result_visibility(
        lambda: next(counts), expected=1,
    )

    assert completed == 1


def test_terminal_result_visibility_waits_through_filesystem_lag(monkeypatch):
    counts = iter([0, 0, 1])
    clock = iter([0.0, 0.0, 0.5])
    sleeps = []
    monkeypatch.setattr(slurm_eval.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(slurm_eval.time, "sleep", sleeps.append)

    completed = slurm_eval._wait_for_terminal_result_visibility(
        lambda: next(counts), expected=1, grace_seconds=1.0,
    )

    assert completed == 1
    assert sleeps == [1.0, 0.5]


def test_terminal_poll_race_recounts_new_result(tmp_path, capsys):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    def poll_terminal(*_args):
        # The worker result becomes visible while the SLURM status query is in
        # flight, after the waiter's first result count for this iteration.
        (results_dir / "task_000000.json").write_text("{}")
        return True, ["COMPLETED"]

    evaluator = SimpleNamespace(
        stall_timeout=None,
        job_timeout=None,
        _poll_jobs_terminal=poll_terminal,
    )

    completed = slurm_eval.BaseSlurmEvaluator._wait_for_job(
        evaluator,
        "123",
        1,
        tmp_path,
        stall_timeout=None,
        job_timeout=None,
    )

    assert completed is True
    assert "results found" not in capsys.readouterr().out
