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


def test_group_shards_preserve_records_and_share_seed_cache_read_only(tmp_path, monkeypatch):
    import evaluation
    import utils
    monkeypatch.setattr(evaluation, 'get_dataset_var_names', lambda _: ['x0'])
    monkeypatch.setattr(utils, 'get_dataset_gt_formula', lambda _: 'x0')

    def must_not_check(*a, **kw):
        raise AssertionError('The seed cache already proves this match')

    monkeypatch.setattr(evaluation, 'check_pysr_symbolic_match', must_not_check)
    for directory in ('cache', 'datasets', 'group_shards/cache', 'group_shards/datasets'):
        (tmp_path / directory).mkdir(parents=True)
    seed = tmp_path / 'cache/fake.json'
    seed.write_text(json.dumps({'x0': {'match': True, 'source': 'sympy'}}))
    seed_bytes = seed.read_bytes()
    raw = tmp_path / 'trial.json'
    raw.write_text(json.dumps({'portfolio': {
        'search_runtime_seconds': 30, 'total_search_budget_seconds': 900,
        'restarts': [{'restart_index': 0, 'search_runtime_seconds': 30,
                      'pareto_frontier': [{'equation': 'x0', 'complexity': 1, 'r2': 1}]}]}}))
    specs = [{'method': method, 'dataset': 'fake', 'seed': 10000,
              'noise': 0, 'path': str(raw)} for method in ['Base PySR', '709715']]
    curve.prepare_groups({'fake': specs}, tmp_path)
    plan = json.loads((tmp_path / 'group_plan.json').read_text())
    assert len(plan) == 2
    for i, item in enumerate(plan):
        curve.analyze_dataset('fake', item['specs'], tmp_path / 'group_shards',
                              item['cache_key'], item['seed_cache'])
        curve.collect_groups(tmp_path)
        if i == 0:
            assert not (tmp_path / 'datasets/fake.json').exists()
    final = json.loads((tmp_path / 'datasets/fake.json').read_text())
    assert final['signature'] == curve.input_signature(specs)
    assert len(final['records']) == 2
    assert {r['method'] for r in final['records']} == {'Base PySR', '709715'}
    assert all(r['first_solve_seconds'] == 30 for r in final['records'])
    assert seed.read_bytes() == seed_bytes
    curve.prepare_groups({'fake': specs}, tmp_path)
    assert json.loads((tmp_path / 'group_plan.json').read_text()) == []


def test_memoized_checker_preserves_symbolic_rules():
    import evaluation
    curve.configure_symbolic_caches()
    equations = ['1.00001*x0', 'x0 + 2', 'x0**2', '0']
    for _ in range(2):
        matches = [evaluation.check_pysr_symbolic_match(
            eq, 'x0', var_names=['x0'], timeout_seconds=3)['match'] for eq in equations]
        assert matches == [True, True, False, False]


def test_sibling_cache_prefers_resolved_and_positive(tmp_path):
    path = tmp_path / 'sibling.json'
    entries = {'positive': {'match': True}, 'resolved': {'match': False},
               'unresolved': {'match': False, 'error': 'timeout'},
               'retain_positive': {'match': False}}
    path.write_text(json.dumps(entries))
    cache = {'positive': {'match': False},
             'resolved': {'match': False, 'error': 'timeout'},
             'retain_positive': {'match': True}}
    curve.merge_definite_caches(cache, [path, tmp_path / 'absent.json'])
    assert cache == {'positive': {'match': True}, 'resolved': {'match': False},
                     'retain_positive': {'match': True}}
    assert json.loads(path.read_text()) == entries
