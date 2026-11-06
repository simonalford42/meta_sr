"""Check generation selection and result indexing without submitting jobs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from eval_neuron_generations import prepare
spec = importlib.util.spec_from_file_location('plot_generations', ROOT / 'figures/plot_neuron_generations.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
configs, manifest = prepare(ROOT / 'runs/708907/run_data.json')
assert len(configs) == 15
assert all(c.pysr_kwargs['max_evals'] == 1000000 for c in configs)
with tempfile.TemporaryDirectory(prefix='neuron-generation-check-') as directory:
    run = Path(directory)
    results = run / 'results'
    results.mkdir()
    (run / 'manifest.json').write_text(json.dumps(manifest))
    (run / 'batch.json').write_text(json.dumps({'batch_dir': str(run)}))
    for config in range(15):
        for world_index, world in enumerate(manifest['worlds']):
            for seed in range(5):
                index = config * 30 + world_index * 5 + seed
                row = dict(config_id=config, dataset_name=world, run_index=seed,
                           error=None, pareto_frontier=[{'test_nrmse': 1e-8 if seed < 3 else 1e-3}])
                (results / f'task_{index:06d}.json').write_text(json.dumps(row))
    rows = plot.summarize(run)
    assert len(rows) == 30
    assert all(r['solved'] == (3 if r['split'] == 'Train' else 15) for r in rows)
    assert all(r['complete'] == r['expected'] for r in rows)
    (results / 'task_000000.json').rename(run / 'pending.json')
    bad = json.loads((results / 'task_000001.json').read_text())
    bad['error'] = 'test worker failure'
    (results / 'task_000001.json').write_text(json.dumps(bad))
    first = plot.summarize(run)[0]
    assert (first['complete'], first['pending'], first['errors'], first['solved']) == (3, 1, 1, 1)
print('PASS: 15 recorded winners, 450 result mappings, solve threshold, pending/error accounting')
