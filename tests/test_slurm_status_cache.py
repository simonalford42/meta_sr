"""Exercise controller rate limiting without invoking any Slurm commands."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import slurm_eval
from parallel_eval_pysr import PySRSlurmEvaluator


@pytest.fixture
def polling(monkeypatch):
    cache = slurm_eval._SlurmStatusCache()
    clock = [0.0]
    calls = []
    responses = []

    def run(command, **kwargs):
        calls.append(command)
        assert responses, f"Unexpected command: {command}"
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, tuple):
            code, output = response
        else:
            code, output = 0, response
        return SimpleNamespace(returncode=code, stdout=output)

    monkeypatch.setattr(slurm_eval, '_SLURM_STATUS_CACHE', cache)
    monkeypatch.setattr(slurm_eval, '_ACTIVE_JOB_IDS', set())
    monkeypatch.setattr(slurm_eval.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(slurm_eval.subprocess, 'run', run)
    return SimpleNamespace(cache=cache, clock=clock, calls=calls, responses=responses)


def test_threads_share_one_batch_per_minute(polling):
    ids = [str(i) for i in range(100, 144)]
    slurm_eval._ACTIVE_JOB_IDS.update(ids)
    polling.responses.append(''.join(f'{jid}|RUNNING\n' for jid in ids))
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda jid: polling.cache.get([jid], {}), ids))
    assert all(result[jid] == ('RUNNING', 0) for jid, result in zip(ids, results))
    assert len(polling.calls) == 1
    assert polling.calls[0][2] == ','.join(ids)
    polling.clock[0] = 59.9
    polling.cache.get(ids, {})
    assert len(polling.calls) == 1
    polling.clock[0] = 60
    polling.responses.append(''.join(f'{jid}|RUNNING\n' for jid in ids))
    polling.cache.get(ids, {})
    assert len(polling.calls) == 2


def test_new_jobs_wait_for_next_refresh(polling):
    polling.responses.append('100|PENDING\n')
    assert polling.cache.get(['100'], {})['100'] == ('PENDING', 0)
    polling.clock[0] = 10
    assert polling.cache.get(['101'], {})['101'] == ('CHECKING', 0)
    assert len(polling.calls) == 1
    polling.clock[0] = 60
    polling.responses.append('100|RUNNING\n101|PENDING\n')
    assert polling.cache.get(['101'], {})['101'] == ('PENDING', 0)
    assert polling.calls[-1][2] == '100,101'


def test_accounting_is_batched_and_terminal_jobs_are_not_repolled(polling):
    polling.responses.extend([
        '100|PENDING\n100|RUNNING\n',
        '101_0|COMPLETED\n101_1|CANCELLED by 123\n102_[0-9]|PENDING\n',
    ])
    states = polling.cache.get(['100', '101', '102'], {})
    assert states == {'100': ('RUNNING', 0), '101': ('COMPLETED', 0), '102': ('PENDING', 0)}
    assert polling.calls[1][0] == 'sacct'
    assert polling.calls[1][2] == '101,102'
    polling.clock[0] = 60
    polling.responses.append('100|RUNNING\n102|RUNNING\n')
    polling.cache.get(['100', '101', '102'], {})
    assert polling.calls[-1][2] == '100,102'
    assert len(polling.calls) == 3


def test_unknown_requires_three_fresh_queries_not_cache_reads(polling):
    evaluator = SimpleNamespace(_get_slurm_env=lambda: {}, UNKNOWN_TERMINAL_POLLS=3)
    streaks = {}
    for minute in range(3):
        polling.clock[0] = minute * 60
        polling.responses.extend(['', ''])
        for _ in range(12):
            terminal, states = slurm_eval.BaseSlurmEvaluator._poll_jobs_terminal(
                evaluator, ['100'], streaks,
            )
            assert terminal is (minute == 2)
            assert states == ['UNKNOWN']
            assert streaks == {'100': minute + 1}
        assert len(polling.calls) == (minute + 1) * 2
    polling.clock[0] = 180
    polling.responses.append('100|RUNNING\n')
    terminal, _ = slurm_eval.BaseSlurmEvaluator._poll_jobs_terminal(evaluator, ['100'], streaks)
    assert not terminal
    assert streaks == {'100': 0}


@pytest.mark.parametrize('failure', [
    (1, '100|COMPLETED\n'),
    slurm_eval.subprocess.TimeoutExpired('squeue', 60),
    OSError('unavailable'),
])
def test_failed_queries_are_rate_limited_and_not_terminal(polling, failure):
    polling.responses.extend([failure, failure])
    evaluator = SimpleNamespace(_get_slurm_env=lambda: {}, UNKNOWN_TERMINAL_POLLS=3)
    for _ in range(10):
        state = slurm_eval.BaseSlurmEvaluator._get_job_status(evaluator, '100')
        assert state == 'CHECKING'
    assert len(polling.calls) == 2


def test_untracked_arrays_are_removed_from_future_batches(polling):
    polling.responses.append('100|RUNNING\n101|RUNNING\n')
    polling.cache.get(['100', '101'], {})
    slurm_eval._untrack_job('100')
    polling.clock[0] = 60
    polling.responses.append('101|RUNNING\n')
    polling.cache.get(['101'], {})
    assert polling.calls[-1][2] == '101'


def test_file_completion_advances_before_next_controller_refresh(polling, monkeypatch, tmp_path):
    results = tmp_path / 'results'
    results.mkdir()
    polling.responses.append('100|RUNNING\n')
    evaluator = SimpleNamespace(
        stall_timeout=None, job_timeout=None,
        _get_slurm_env=lambda: {}, UNKNOWN_TERMINAL_POLLS=3,
    )
    evaluator._poll_jobs_terminal = lambda *args: slurm_eval.BaseSlurmEvaluator._poll_jobs_terminal(
        evaluator, *args,
    )
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        polling.clock[0] += seconds
        (results / 'task_000000.json').write_text('{}')

    monkeypatch.setattr(slurm_eval.time, 'sleep', sleep)
    monkeypatch.setattr(slurm_eval.time, 'time', lambda: polling.clock[0])
    assert slurm_eval.BaseSlurmEvaluator._wait_for_job(evaluator, '100', 1, tmp_path)
    assert sleeps == [10]
    assert len(polling.calls) == 1
    assert '100' not in polling.cache.monitored


@pytest.mark.parametrize('waiter', ['single', 'base_multi', 'base_retry', 'pysr_multi', 'pysr_batches', 'pysr_retry'])
def test_complete_files_skip_status_queries(polling, tmp_path, waiter):
    results = tmp_path / 'results'
    results.mkdir()
    (results / 'task_000000.json').write_text('{}')
    evaluator = SimpleNamespace(stall_timeout=None, job_timeout=None)
    if waiter == 'single':
        slurm_eval.BaseSlurmEvaluator._wait_for_job(evaluator, '100', 1, tmp_path)
    elif waiter == 'base_multi':
        slurm_eval.BaseSlurmEvaluator._wait_for_jobs(evaluator, ['100', '101'], 1, tmp_path)
    elif waiter == 'base_retry':
        slurm_eval.BaseSlurmEvaluator._wait_for_retry_jobs(evaluator, ['100'], 1, tmp_path, [0])
    elif waiter == 'pysr_multi':
        PySRSlurmEvaluator._wait_for_jobs(evaluator, ['100', '101'], 1, tmp_path)
    elif waiter == 'pysr_batches':
        PySRSlurmEvaluator._wait_for_jobs_multi_batch(evaluator, ['100'], 1, [tmp_path])
    else:
        PySRSlurmEvaluator._wait_for_retry_jobs_multi_batch(evaluator, ['100'], [(tmp_path, 0)])
    assert polling.calls == []
