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
