"""Check fine-shard assembly and reuse without expensive symbolic evaluation."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_portfolio_seed_shards as seeds


def test_seed_collection_and_completed_parent_reuse(tmp_path, monkeypatch):
    source = tmp_path / 'source.json'
    source.write_text('{}')
    specs = [{'method': 'Base PySR', 'dataset': 'fake', 'noise': 0.0,
              'seed': i, 'path': str(source)} for i in (10000, 10001)]
    parent = {'dataset': 'fake', 'cache_key': 'group_0000', 'specs': specs}
    plan = [{'parent_index': 0, 'parent': parent, 'specs': [s],
             'cache_key': f'seed_{i:04d}'} for i, s in enumerate(specs)]
    (tmp_path / 'seed_plan.json').write_text(json.dumps(plan))
    group_dir = tmp_path / 'group_shards/datasets'
    seed_dir = tmp_path / 'seed_shards/datasets'
    group_dir.mkdir(parents=True)
    seed_dir.mkdir(parents=True)
    for i, item in enumerate(plan):
        if i == 1:
            assert seeds.collect_seed_groups(tmp_path) == (0, 1)
            assert not (group_dir / 'group_0000.json').exists()
        record = {k: v for k, v in item['specs'][0].items() if k != 'path'}
        payload = {'signature': seeds.input_signature(item['specs']),
                   'records': [record], 'counters': {'new_checks': 2}}
        (seed_dir / f'seed_{i:04d}.json').write_text(json.dumps(payload))
    assert seeds.collect_seed_groups(tmp_path) == (1, 1)
    result = json.loads((group_dir / 'group_0000.json').read_text())
    assert [r['seed'] for r in result['records']] == [10000, 10001]
    assert result['counters']['new_checks'] == 4
    assert result['signature'] == seeds.input_signature(specs)
    monkeypatch.setattr(seeds, 'analyze_dataset', lambda *a, **k: pytest.fail('rechecked completed parent'))
    seeds.run_seed(tmp_path, 0)
    child = json.loads((seed_dir / 'seed_0000.json').read_text())
    assert child['records'] == [result['records'][0]]
    assert child['counters'] == {'completed_parent_reused': 1}


def test_extension_preserves_live_worker_indices(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text('{}')
    groups = []
    for i in range(2):
        groups.append({'dataset': f'task{i}', 'cache_key': f'group_{i:04d}',
                       'specs': [{'method': 'Base PySR', 'dataset': f'task{i}',
                                  'noise': 0., 'seed': n, 'path': str(source)}
                                 for n in (10000, 10001)]})
    old = [{'parent_index': 0, 'parent': groups[0], 'specs': [spec],
            'cache_key': f'seed_{i:04d}'} for i, spec in enumerate(groups[0]['specs'])]
    (tmp_path / 'group_plan.json').write_text(json.dumps(groups))
    (tmp_path / 'seed_plan.json').write_text(json.dumps(old))
    seeds.prepare(tmp_path, extend=True, all_remaining=True)
    new = json.loads((tmp_path / 'seed_plan.json').read_text())
    assert new[:len(old)] == old
    assert [p['cache_key'] for p in new] == [f'seed_{i:04d}' for i in range(4)]
    assert [p['parent_index'] for p in new] == [0, 0, 1, 1]
    seeds.prepare(tmp_path, extend=True, all_remaining=True)
    assert json.loads((tmp_path / 'seed_plan.json').read_text()) == new
