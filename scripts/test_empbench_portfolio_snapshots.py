"""Offline checks for first-restart snapshots; no Julia fits or SLURM submissions."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("eval_cap", [None, 1000000])
def test_first_restart_only_and_warmup_exclusion(tmp_path, monkeypatch, eval_cap):
    import numpy as np
    import pandas as pd
    import domains
    import parallel_eval_pysr as worker
    import run_pysr_srbench as runner

    class Domain:
        def sympy_mappings(self):
            return {}
        def load_train_validation(self, *args, **kwargs):
            x = np.arange(10.).reshape(-1, 1)
            return x, x[:, 0], x, x[:, 0], 'x0'
        def check_solved(self, **kwargs):
            return {'match': False}
        def pareto_metrics(self, **kwargs):
            return [{'equation': 'x0', 'complexity': 1, 'loss': 0., 'pysr_index': 0, 'r2': 1.}]

    class Model:
        def __init__(self, **kwargs):
            assert '_frontier_snapshot_seconds' not in kwargs
            self.__dict__.update(kwargs)
            self.equations_ = pd.DataFrame([{'equation': 'x0', 'complexity': 1, 'loss': 0.}])
        def get_best(self):
            return self.equations_.iloc[0]
        def predict(self, x, index=None):
            return x[:, 0]

    calls = []
    trace = [{'elapsed_seconds': 5.1, 'scheduled_seconds': 5., 'status': 'ok',
              'equations': [{'equation': 'x0', 'complexity': 1, 'loss': 0.}]}]
    def fake_fit(*args, **kwargs):
        calls.append(kwargs['frontier_snapshot_seconds'])
        if kwargs['frontier_snapshot_seconds'] is not None:
            Path(kwargs['hof_path'] + '.snapshots.jsonl').write_text(json.dumps(trace[0]) + '\n')
        return kwargs['model']

    monkeypatch.setattr(domains, 'get_domain', lambda _: Domain())
    monkeypatch.setattr(worker, '_import_pysr_regressor', lambda: Model)
    monkeypatch.setattr(worker, '_load_dynamic_loss', lambda _: None)
    monkeypatch.setattr(runner, 'run_pysr_with_hof_checkpoints', fake_fit)
    monkeypatch.setattr(worker, '_get_pysr_num_evaluations', lambda _: 100)
    spec = worker.PySRTaskSpec(config_id=0, dataset_name='fake', pysr_kwargs={},
        mutation_weights={}, seed=10000, data_seed=42, custom_loss_code='mock',
        frontier_snapshot_seconds=5, retain_pareto_frontier=True, portfolio_time_limit_seconds=3600,
        portfolio_restart_timeout_seconds=90 if eval_cap is None else None,
        portfolio_restart_max_evals=eval_cap, portfolio_restart_count=2,
        hof_csv_paths=[str(tmp_path/'hof.csv')])
    result = worker._evaluate_pysr_task(spec, use_cache=False)
    assert calls == [None, 5, None]  # Warm-up, first restart, second restart.
    assert result.error is None
    portfolio = result.portfolio
    assert portfolio['warmup_excluded_from_budget'] is True
    assert len(portfolio['restarts']) == 2
    assert portfolio['restarts'][0]['execution_trace'] == trace
    assert portfolio['restarts'][1]['execution_trace'] is None
    assert all(r['pareto_frontier'] for r in portfolio['restarts'])
    assert portfolio['search_runtime_seconds'] == sum(r['search_runtime_seconds'] for r in portfolio['restarts'])


def test_empirical_cli_accepts_time_restart_limit():
    from empbench_full_eval import build_parser
    args = build_parser().parse_args(['--portfolio-time-limit', '3600',
        '--portfolio-restart-timeout', '90', '--frontier-snapshot-seconds', '5',
        '--cpus-per-task', '1'])
    assert args.portfolio_restart_timeout == 90
    assert args.portfolio_restart_max_evals is None
