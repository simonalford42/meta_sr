"""Small synthetic checks of the historical portfolio analysis bookkeeping."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    'portfolio_curve', ROOT / 'scripts/analyze_portfolio_solve_over_time.py')
curve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curve)


def test_first_recovery_cache_gate_and_saved_positive(tmp_path, monkeypatch):
    import evaluation
    import utils
    monkeypatch.setattr(evaluation, 'get_dataset_var_names', lambda _: ['x0'])
    monkeypatch.setattr(utils, 'get_dataset_gt_formula', lambda _: 'x0')
    checked = []

    def check(eq, *args, **kwargs):
        checked.append(eq)
        return {'match': eq == 'match', 'error': None}

    monkeypatch.setattr(evaluation, 'check_pysr_symbolic_match', check)
    for sub in ('cache', 'datasets'):
        (tmp_path / sub).mkdir()

    def row(eq, r2=1.0):
        return {'equation': eq, 'complexity': 1, 'loss': 1.0,
                'r2': r2, 'solved': False}

    raw = {'gt_match_score': 0, 'pareto_frontier': [], 'portfolio': {
        'total_search_budget_seconds': 900, 'search_runtime_seconds': 902,
        'restarts': [
            {'restart_index': 0, 'search_runtime_seconds': 100,
             'pareto_frontier': [row('low', .4), row('miss')]},
            {'restart_index': 1, 'search_runtime_seconds': 200,
             'pareto_frontier': [row('match')]},
            {'restart_index': 2, 'search_runtime_seconds': 602,
             'pareto_frontier': [row('never_check')]},
        ]}}
    path = tmp_path / 'trial.json'
    path.write_text(json.dumps(raw))
    specs = [{'method': 'Base PySR', 'dataset': 'fake', 'seed': i,
              'noise': 0, 'path': str(path)} for i in [10000, 10001]]
    result = curve.analyze_dataset('fake', specs, tmp_path)
    assert checked == ['miss', 'match']
    assert [r['first_solve_seconds'] for r in result['records']] == [300, 300]
    assert result['counters']['later_restarts_skipped'] == 2
    assert result['counters']['r2_gate_skips'] == 2
    assert all(not r['final_merged_solved'] for r in result['records'])
    # False saved flags did not skip checking. Positive saved flags ARE reused.
    raw['gt_match_score'] = 1
    raw['gt_matched_equation'] = 'known'
    raw['portfolio']['restarts'] = [{
        'restart_index': 0, 'search_runtime_seconds': 902,
        'pareto_frontier': [row('known')]}]
    path.write_text(json.dumps(raw))
    result = curve.analyze_dataset('fake', specs, tmp_path)
    assert checked == ['miss', 'match']
    assert result['records'][0]['first_solve_seconds'] == 902
    assert result['records'][0]['first_solve_budget_seconds'] == 900


def test_missing_and_unresolved(tmp_path, monkeypatch):
    import evaluation
    import utils
    monkeypatch.setattr(evaluation, 'get_dataset_var_names', lambda _: ['x0'])
    monkeypatch.setattr(utils, 'get_dataset_gt_formula', lambda _: 'x0')
    monkeypatch.setattr(evaluation, 'check_pysr_symbolic_match',
                        lambda *a, **kw: {'match': False, 'error': 'timeout'})
    for sub in ('cache', 'datasets'):
        (tmp_path / sub).mkdir()
    path = tmp_path / 'trial.json'
    path.write_text(json.dumps({'portfolio': {
        'search_runtime_seconds': 900, 'total_search_budget_seconds': 900,
        'restarts': [{'restart_index': 0, 'search_runtime_seconds': 900,
                      'pareto_frontier': [{'equation': 'hard', 'complexity': 1, 'r2': 1}]}]}}))
    specs = [{'method': 'Base PySR', 'dataset': 'fake', 'seed': i, 'noise': 0,
              'path': str(p)} for i, p in enumerate([path, tmp_path / 'missing.json'])]
    result = curve.analyze_dataset('fake', specs, tmp_path)
    assert result['records'][0]['unresolved_checks_before_solve'] == 1
    assert result['records'][0]['first_solve_seconds'] is None
    assert result['records'][1]['status'] == 'missing'


def test_rounding_equivalent_equations_share_definite_checks(tmp_path, monkeypatch):
    import evaluation
    import utils
    monkeypatch.setattr(evaluation, 'get_dataset_var_names', lambda _: ['x0'])
    monkeypatch.setattr(utils, 'get_dataset_gt_formula', lambda _: 'x0')
    checked = []

    def check(eq, *a, **kw):
        checked.append(eq)
        return {'match': False, 'error': None}

    monkeypatch.setattr(evaluation, 'check_pysr_symbolic_match', check)
    for sub in ('cache', 'datasets'):
        (tmp_path / sub).mkdir()
    path = tmp_path / 'trial.json'
    path.write_text(json.dumps({'portfolio': {
        'search_runtime_seconds': 900, 'total_search_budget_seconds': 900,
        'restarts': [{'restart_index': 0, 'search_runtime_seconds': 900,
                      'pareto_frontier': [
                          {'equation': eq, 'complexity': 5, 'r2': 1}
                          for eq in ['1.00001*x0 + 2.0001', '1.00002*x0 + 2.0002']]}]}}))
    specs = [{'method': 'Base PySR', 'dataset': 'fake', 'seed': 10000,
              'noise': 0, 'path': str(path)}]
    result = curve.analyze_dataset('fake', specs, tmp_path)
    assert len(checked) == 1
    assert result['counters']['rounded_cache_hits'] == 1
