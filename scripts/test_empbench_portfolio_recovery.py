"""Offline tests of snapshot assembly and cached Terra binary-search planning."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('empcurve', Path(__file__).with_name('empbench_portfolio_recovery.py'))
curve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curve)


def row(equation, loss=1):
    return {'equation': equation, 'loss': loss, 'complexity': 1}


def test_timeline_preserves_fine_snapshots_restart_endpoints_and_missing_reads():
    result = {'frontier': [row('later', .5)], 'portfolio': {
        'warmup_excluded_from_budget': True,
        'restarts': [
            {'search_runtime_seconds': 91, 'pareto_frontier': [row('first')],
             'execution_trace': [
                 {'scheduled_seconds': 5, 'elapsed_seconds': 5.1, 'status': 'unavailable'},
                 {'scheduled_seconds': 10, 'elapsed_seconds': 10.2, 'status': 'ok', 'equations': [row('early')]},
                 {'scheduled_seconds': 90, 'elapsed_seconds': 90.1, 'status': 'ok', 'equations': [row('first')]},
                 {'final': True, 'scheduled_seconds': None, 'elapsed_seconds': 90.8, 'status': 'ok', 'equations': [row('first')]},
             ]},
            {'search_runtime_seconds': 92, 'pareto_frontier': [row('later', .5)]},
        ]}}
    snapshots, missing = curve.make_snapshots(result)
    assert [s['seconds'] for s in snapshots] == [10.2, 90.1, 91, 183]
    assert snapshots[-1]['frontier'][0]['equation'] == 'later'
    assert {'scheduled_seconds': 5, 'status': 'unavailable'} in missing
    assert {'scheduled_seconds': 15, 'status': 'missing'} in missing
    result['frontier'] = [row('wrong')]
    with pytest.raises(ValueError, match='differs'):
        curve.make_snapshots(result)


def prepare_state(tmp_path, monkeypatch, cache):
    monkeypatch.setattr(curve, 'OUT', tmp_path)
    monkeypatch.setattr(curve, 'initialize', lambda: None)
    monkeypatch.setattr(curve, 'RUNS', {'Baseline': None})
    snapshots = [{'seconds': i*5+.1, 'kind': 'first_restart_snapshot',
                  'frontier': curve.compact([row(str(i))])} for i in range(1, 19)]
    trial = {'id': 0, 'method': 'Baseline', 'dataset': 'empirical_bode', 'seed': 10000,
             'n_snapshots': len(snapshots), 'budget_seconds': 3600, 'history': []}
    curve.write(tmp_path/'snapshots/000.json', snapshots)
    curve.write(tmp_path/'state.json', {'trials': [trial], 'cache': cache(snapshots), 'round': 0, 'cost_usd': 0})
    return snapshots


def test_dry_run_plans_final_review_including_bode_without_api(tmp_path, monkeypatch):
    snapshots = prepare_state(tmp_path, monkeypatch, lambda _: {})
    monkeypatch.setattr(curve, 'OpenRouterHTTPClient', lambda *_: pytest.fail('Unexpected network access'))
    curve.step(dry_run=True)
    items = json.loads((tmp_path/'rounds/00/items.json').read_text())
    assert len(items) == 1
    item = next(iter(items.values()))
    assert item['dataset'] == 'empirical_bode'
    assert item['frontier'] == snapshots[-1]['frontier']
    assert not (tmp_path/'first_recovery.json').exists()


def test_cached_binary_search_finds_five_second_boundary(tmp_path, monkeypatch):
    def cache(snapshots):
        return {curve.digest(['empirical_bode', s['frontier']]):
                {'classification': 'exact' if i >= 4 else 'miss', 'source': 'llm'}
                for i, s in enumerate(snapshots)}
    prepare_state(tmp_path, monkeypatch, cache)
    curve.step(dry_run=True)
    record = json.loads((tmp_path/'first_recovery.json').read_text())[0]
    assert record['first_solve_seconds'] == 25.1
    assert record['last_negative_seconds'] == 20.1
    assert len(record['history']) <= 5


def test_final_negative_skips_midpoints_and_pending_dry_run_never_polls(tmp_path, monkeypatch):
    prepare_state(tmp_path, monkeypatch, lambda s: {curve.digest(['empirical_bode', s[-1]['frontier']]): {'classification': 'miss'}})
    curve.step(dry_run=True)
    record = json.loads((tmp_path/'first_recovery.json').read_text())[0]
    assert record['first_solve_seconds'] is None and record['history'] == []
    curve.write(tmp_path/'rounds/00/batch.json', {'id': 'pending'})
    monkeypatch.setattr(curve, 'OpenRouterHTTPClient', lambda *_: pytest.fail('Unexpected network access'))
    curve.step(dry_run=True)


def test_initialize_covers_all_180_trials_and_detects_source_changes(tmp_path, monkeypatch):
    datasets = [name for name in curve.TARGETS if name.startswith('empirical_')]
    assert len(datasets) == 9
    paths = {'Baseline': tmp_path/'base', '709715': tmp_path/'evolved'}
    monkeypatch.setattr(curve, 'RUNS', paths)
    monkeypatch.setattr(curve, 'OUT', tmp_path/'reviews')
    for path in paths.values():
        records = []
        for dataset in datasets:
            for seed in range(10000, 10010):
                records.append({'dataset': dataset, 'seed': seed, 'frontier': [row('x0')],
                    'portfolio': {'warmup_excluded_from_budget': True, 'total_search_budget_seconds': 3600,
                        'restarts': [{'search_runtime_seconds': 91, 'pareto_frontier': [row('x0')],
                            'execution_trace': [{'scheduled_seconds': 5, 'elapsed_seconds': 5.1,
                                'status': 'ok', 'equations': [row('x0')]}]}]}})
        curve.write(path/'empbench_results.json', {'protocol': {'datasets': datasets}, 'runs': records})
    curve.initialize()
    state = json.loads((curve.OUT/'state.json').read_text())
    assert len(state['trials']) == 180
    assert {t['dataset'] for t in state['trials']} == set(datasets)
    curve.initialize()  # Resume the same inputs.
    path = paths['Baseline']/'empbench_results.json'
    data = json.loads(path.read_text())
    data['changed'] = True
    curve.write(path, data)
    with pytest.raises(ValueError, match='changed'):
        curve.initialize()


def test_cost_guard_prevents_submission(tmp_path, monkeypatch):
    prepare_state(tmp_path, monkeypatch, lambda _: {})
    monkeypatch.setattr(curve, 'MAX_COST', 0)
    monkeypatch.setattr(curve, 'OpenRouterHTTPClient', lambda *_: pytest.fail('Unexpected submission'))
    with pytest.raises(AssertionError, match='budget'):
        curve.step(dry_run=False)


def test_empirical_family_label_normalization_preserves_evidence():
    item = {'dataset': 'empirical_leavitt', 'frontier': curve.compact([row('log(x0)')])}
    raw = {'classification': 'phenomenological_match', 'matching_equation': 'log(x0)',
           'best_frontier_indices': [0]}
    normalized = curve.normalize_review(raw, item)
    assert normalized['classification'] == 'exact'
    assert normalized['original_classification'] == raw['classification']
    assert raw['classification'] == 'phenomenological_match'
    assert curve.normalize_review({**raw, 'classification': 'near'}, item)['classification'] == 'near'
    with pytest.raises(ValueError, match='Unexpected'):
        curve.normalize_review(raw, {**item, 'dataset': 'empirical_planck'})
    with pytest.raises(ValueError, match='saved equation'):
        curve.normalize_review({**raw, 'matching_equation': 'other'}, item)


def test_saved_completed_batch_resumes_offline_and_charges_once(tmp_path, monkeypatch):
    snapshots = prepare_state(tmp_path, monkeypatch, lambda _: {})
    key = curve.digest(['empirical_bode', snapshots[-1]['frontier']])
    item = {'dataset': 'empirical_bode', 'frontier': snapshots[-1]['frontier']}
    review = {'classification': 'phenomenological_match', 'matching_equation': '18',
              'best_frontier_indices': [0], 'explanation': 'Accepted family.'}
    curve.write(tmp_path/'rounds/00/items.json', {key: item})
    curve.write(tmp_path/'rounds/00/batch.json', {'id': 'already-paid'})
    curve.write(tmp_path/'rounds/00/responses.json', {'status': 'completed', 'results': [
        {'custom_id': key, 'response': {'status_code': 200, 'body': {
            'choices': [{'message': {'content': json.dumps(review)}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}}}]})
    monkeypatch.setattr(curve, 'OpenRouterHTTPClient', lambda *_: pytest.fail('Unexpected network access'))
    curve.step(dry_run=True)
    state = json.loads((tmp_path/'state.json').read_text())
    assert state['round'] == 1 and state['cost_usd'] > 0
    assert state['cache'][key]['original_classification'] == 'phenomenological_match'
    curve.step(dry_run=True)
    assert json.loads((tmp_path/'state.json').read_text())['cost_usd'] == state['cost_usd']
