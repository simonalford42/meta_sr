"""Offline checks: replacement, timing, reviews and command dependencies."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pytest
import splice_srbench2_first_restart as splice
import review_srbench2_spliced_portfolio as review


def row():
    def frontier(eq, loss): return [dict(equation=eq,loss=loss,complexity=1)]
    a=dict(seed=123, search_runtime_seconds=10., error=None, pareto_frontier=frontier('old',2))
    b=dict(seed=456, search_runtime_seconds=20., error=None, pareto_frontier=frontier('later',1))
    return dict(dataset='first_principles_hubble',seed=10000,run_index=0,noise=0,config_id=0,present=True,error=None,
                portfolio=dict(restart_max_evals=1000000,restart_timeout_seconds=None,warmup_excluded_from_budget=True,
                               warmup_seconds=90,restarts=[a,b]))


def fresh(old):
    n=copy.deepcopy(old); n['portfolio']['restarts']=n['portfolio']['restarts'][:1]
    f=n['portfolio']['restarts'][0]; f['search_runtime_seconds']=16.
    f['pareto_frontier']=[dict(equation='new',loss=.5,complexity=1)]
    f['execution_trace']=[dict(scheduled_seconds=5,elapsed_seconds=5.2,status='ok',equations=f['pareto_frontier']),
                          dict(scheduled_seconds=10,elapsed_seconds=10.1,status='unavailable',equations=[]),
                          dict(elapsed_seconds=16,final=True,status='ok',equations=f['pareto_frontier'])]
    return n


def test_replace_removes_old_frontier_and_shifts_clock():
    a=row(); original=copy.deepcopy(a); result=splice.replace(a,fresh(a))
    assert a==original
    assert result['pareto_frontier'][0]['equation']=='new'
    obs,missing=splice.snapshots(result)
    assert [o['seconds'] for o in obs]==[5.2,16.,36.]
    assert {m['scheduled_seconds'] for m in missing}=={10,15}
    assert result['solved'] is None
    assert result['portfolio']['restarts'][1]==a['portfolio']['restarts'][1]


def test_wrong_restart_seed_rejected():
    a=row(); n=fresh(a); n['portfolio']['restarts'][0]['seed']=789
    with pytest.raises(ValueError,match='seed'): splice.replace(a,n)


def test_bode_normalization_preserves_raw_label():
    item={'dataset':'first_principles_bode','frontier':[dict(equation='exp(x0)')]}
    r=review.normalize(dict(classification='phenomenological_match',matching_equation='exp(x0)',best_frontier_indices=[0]),item)
    assert r['classification']=='exact'
    assert r['original_classification']=='phenomenological_match'


def test_initialize_and_prepare_without_network(tmp_path,monkeypatch):
    engine=review.engine
    runs={}
    for method in ['Baseline','709715']:
        path=tmp_path/method; path.mkdir(); runs[method]=path
        rows={}
        for ds in splice.DATASETS:
            for seed in range(10000,10010):
                a=row(); a.update(dataset=ds,seed=seed)
                rows[f'{ds}|{seed}|0']=splice.replace(a,fresh(a))
        (path/'srbench_full_results.json').write_text(json.dumps({'results':rows}))
    monkeypatch.setattr(engine,'RUNS',runs);monkeypatch.setattr(engine,'OUT',tmp_path/'reviews')
    monkeypatch.setattr(engine,'initialize',review.initialize)
    engine.step(dry_run=True)
    state=json.loads((engine.OUT/'state.json').read_text())
    assert len(state['trials'])==180
    assert not (engine.OUT/'rounds/00/batch.json').exists()
    assert (engine.OUT/'rounds/00/payload.json').exists()


def test_submission_dependency_chain_without_slurm(tmp_path):
    import os
    import subprocess
    mock=tmp_path/'sbatch'
    log=tmp_path/'calls.jsonl'
    mock.write_text('#!'+sys.executable+'\nimport json,os,sys\nfrom pathlib import Path\np=Path(os.environ["MOCK_SBATCH_LOG"])\nn=len(p.read_text().splitlines()) if p.exists() else 0\nwith p.open("a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nprint(100+n)\n')
    mock.chmod(0o755)
    root=Path(__file__).resolve().parents[1]
    subprocess.run(['bash',str(root/'submit_jobs.sh')],check=True,capture_output=True,
                   env={**os.environ,'PATH':str(tmp_path)+os.pathsep+os.environ['PATH'],'MOCK_SBATCH_LOG':str(log)})
    calls=[json.loads(s) for s in log.read_text().splitlines()]
    assert len(calls)==6
    assert '--dependency=afterok:100' in calls[1]
    assert '--dependency=afterok:100' in calls[2]
    assert '--dependency=afterok:101' in calls[3]
    assert '--dependency=afterok:102:103' in calls[4]
    assert '--dependency=afterok:104' in calls[5]


def test_srbench2_cli_accepts_first_restart_snapshots(monkeypatch):
    import srbench_full_eval
    from srbench2_full_eval import _with_srbench2_defaults
    class Parsed(Exception): pass
    def stop(args):
        assert args.portfolio_restart_max_evals==1000000
        assert args.portfolio_restart_count==1
        assert args.frontier_snapshot_seconds==5
        assert args.timeout==3600
        raise Parsed()
    monkeypatch.setattr(srbench_full_eval,'load_evaluation_datasets',stop)
    args=_with_srbench2_defaults(['--ground-truth','--portfolio-time-limit','3600',
        '--portfolio-restart-max-evals','1000000','--portfolio-restart-count','1',
        '--frontier-snapshot-seconds','5','--no-maxsize-warmup','--cpus-per-task','1'])
    with pytest.raises(Parsed): srbench_full_eval.main(args,force_srbench_2025=True)


def test_complete_splice_checks_worker_configuration(tmp_path):
    manifest=dict(srbench_edition=2025,ground_truth_protocol='srbench2_exact_recovery',
                  mode='baseline',backend='pysr',max_samples=1000,cpus_per_task=1,
                  baseline_l1_loss=True,maxsize_warmup=False,seeds=list(range(10000,10010)),
                  noise_levels=[0],method_meta={},batches=[{'batch_dir':'batch'}])
    original={};replacement={};tasks=[]
    for ds in splice.DATASETS:
        for seed in range(10000,10010):
            r=row();r.update(dataset=ds,seed=seed);key=f'{ds}|{seed}|0'
            original[key]=r;replacement[key]=fresh(r)
            tasks.append(dict(dataset_name=ds,seed=seed,pysr_kwargs={},data_seed=42))
    for name, rows in [('old',original),('new',replacement)]:
        d=tmp_path/name;(d/'batch').mkdir(parents=True)
        (d/'manifest.json').write_text(json.dumps(manifest))
        (d/'srbench_full_results.json').write_text(json.dumps({'results':rows}))
        (d/'batch/tasks.json').write_text(json.dumps(tasks))
    splice.splice(tmp_path/'old',tmp_path/'new',tmp_path/'out')
    assert len(json.loads((tmp_path/'out/srbench_full_results.json').read_text())['results'])==90
    tasks[0]['data_seed']=99
    (tmp_path/'new/batch/tasks.json').write_text(json.dumps(tasks))
    with pytest.raises(ValueError,match='Worker configuration differs'):
        splice.splice(tmp_path/'old',tmp_path/'new',tmp_path/'out2')
