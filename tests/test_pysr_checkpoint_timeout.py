from types import SimpleNamespace

import pandas as pd

import run_pysr_srbench


class _FakeModel:
    def __init__(self, clock, fit_durations):
        self._clock = clock
        self._fit_durations = iter(fit_durations)
        self.fit_timeouts = []
        self.output_directory = None
        self.warm_start = False
        self.equations_ = pd.DataFrame(
            [{"complexity": 1, "loss": 1.0, "equation": "x0"}]
        )

    def fit(self, *_args, **_kwargs):
        self.fit_timeouts.append(self.timeout_in_seconds)
        self._clock.now += next(self._fit_durations)


def test_eval_checkpoints_share_one_timeout(monkeypatch, tmp_path):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(run_pysr_srbench.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(run_pysr_srbench.time, "time", lambda: clock.now)
    model = _FakeModel(clock, [60.0, 200.0, 1.0])

    run_pysr_srbench.run_pysr_with_hof_checkpoints(
        [[0.0]],
        [0.0],
        feature_names=["x0"],
        dataset_name="deadline_test",
        results_dir=str(tmp_path),
        milestones=[333_333, 666_667, 1_000_000],
        model=model,
        hof_path=str(tmp_path / "hof.csv"),
        total_timeout_in_seconds=500,
    )

    assert model.fit_timeouts == [500, 440, 240]


def test_checkpoint_loop_returns_last_frontier_when_budget_is_gone(
    monkeypatch, tmp_path
):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(run_pysr_srbench.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(run_pysr_srbench.time, "time", lambda: clock.now)
    model = _FakeModel(clock, [101.0])

    returned = run_pysr_srbench.run_pysr_with_hof_checkpoints(
        [[0.0]],
        [0.0],
        feature_names=["x0"],
        dataset_name="deadline_test",
        results_dir=str(tmp_path),
        milestones=[1, 2, 3],
        model=model,
        hof_path=str(tmp_path / "hof.csv"),
        total_timeout_in_seconds=100,
    )

    assert returned is model
    assert model.fit_timeouts == [100]
