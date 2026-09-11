"""Tests of passive frontier capture without warm-starting or restarting PySR."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from frontier_snapshots import read_frontier
from run_pysr_srbench import run_pysr_with_hof_checkpoints
from parallel_eval_pysr import _load_execution_trace, PySRTaskSpec, _spec_expects_execution_trace, _build_cache_identity


def publish(root,text):
    path=root/'run'/'hall_of_fame.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text)
    Path(str(path)+'.bak').write_text(text)
    return path


def test_reader_rejects_partial_native_writes(tmp_path):
    text='Complexity,Loss,Equation\n1,2.0,"x0"\n3,0.1,"x0 * x0"\n'
    path=publish(tmp_path,text)
    assert len(read_frontier(tmp_path,time.time())['equations'])==2
    path.write_text(text[:25])
    assert read_frontier(tmp_path,time.time())['status']=='unavailable'
    publish(tmp_path,text.replace('3,0.1','3,3.0'))
    assert read_frontier(tmp_path,time.time())['status']=='unavailable'


def test_single_fit_periodic_and_final_snapshots(tmp_path):
    class Model:
        output_directory=str(tmp_path/'native')
        warm_start=False
        max_evals=10**9
        calls=0
        def fit(self,*args,**kwargs):
            self.calls+=1
            for i in range(8):
                publish(Path(self.output_directory),f'Complexity,Loss,Equation\n1,{2-i/10},"x0"\n')
                time.sleep(.1)
    model=Model()
    path=str(tmp_path/'trace.csv')
    run_pysr_with_hof_checkpoints(None,None,['x0'],'fake',str(tmp_path),[],model,
                                  hof_path=path,frontier_snapshot_seconds=.15)
    assert model.calls==1 and model.warm_start is False and model.max_evals==10**9
    trace=_load_execution_trace([path+'.snapshots.jsonl'])
    assert len(trace)>=3 and trace[-1]['final']
    assert all(r['status']=='ok' for r in trace)
    assert all(r['elapsed_seconds']>=r['scheduled_seconds'] for r in trace if not r['final'])
    assert trace[-1]['equations'][0]['loss']==1.3
    assert all(r['milestone_kind']=='fit_wall_time' for r in trace)


def test_snapshot_cache_identity_and_spec_roundtrip():
    spec=PySRTaskSpec(config_id=0,dataset_name='fake',pysr_kwargs={},mutation_weights={},seed=1,data_seed=1)
    before=_build_cache_identity(spec)
    assert not _spec_expects_execution_trace(spec)
    spec.frontier_snapshot_seconds=60
    assert _spec_expects_execution_trace(spec)
    assert _build_cache_identity(spec)!=before
    import dataclasses
    restored=PySRTaskSpec(**json.loads(json.dumps(dataclasses.asdict(spec))))
    assert restored.frontier_snapshot_seconds==60


def test_aggregate_retains_snapshot_trace(tmp_path):
    from srbench_results_io import build_keyed_results
    batch=tmp_path/'slurm_pysr/eval_0000'
    (batch/'results').mkdir(parents=True)
    (batch/'tasks.json').write_text(json.dumps([{'dataset_name':'first_principles_hubble','seed':10000}]))
    trace=[{'elapsed_seconds':60.1,'scheduled_seconds':60,'milestone_kind':'fit_wall_time',
            'status':'ok','equations':[{'equation':'x0','complexity':1,'loss':1}]}]
    (batch/'results/task_000000.json').write_text(json.dumps({'execution_trace':trace,'gt_match_score':0}))
    results=build_keyed_results(tmp_path,{'batches':[{'batch_dir':'slurm_pysr/eval_0000'}]})
    assert results['first_principles_hubble|10000|0']['execution_trace']==trace
