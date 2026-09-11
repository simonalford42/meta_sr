"""Check binary-search termination, cache reuse, and restart-time reporting."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('curve',Path(__file__).with_name('srbench2_portfolio_recovery.py'))
curve=importlib.util.module_from_spec(spec)
spec.loader.exec_module(curve)


def test_cached_search_finds_first_positive_and_skips_final_negatives(tmp_path,monkeypatch):
    monkeypatch.setattr(curve,'OUT',tmp_path)
    snapshots=[{'seconds':n*10,'frontier':[{'frontier_index':0,'complexity':1,'equation':str(n)}]}
               for n in range(1,18)]
    cache={curve.digest(['task',s['frontier']]):{'classification':'exact' if i>=5 else 'miss'}
           for i,s in enumerate(snapshots)}
    trials=[{'id':i,'dataset':'task','seed':10000+i,'low':0,'high':17,'n_restarts':17,'final_positive':i!=2,'history':[]}
            for i in range(3)]
    for t in trials:curve.write(tmp_path/'snapshots'/f"{t['id']:03d}.json",snapshots)
    curve.write(tmp_path/'state.json',{'trials':trials,'round':0,'cache':cache,'cost_usd':0})
    rendered=[]
    monkeypatch.setattr(curve,'render',lambda state:rendered.append(state))
    curve.step(dry_run=True)
    assert len(rendered)==1
    for t in rendered[0]['trials'][:2]:
        assert (t['low'],t['high'])==(5,6)
        assert len(t['history'])<=5
    assert not rendered[0]['trials'][2]['history']


def test_incremental_merge_agrees_with_full_merge_and_can_lose_exact():
    frontiers=[
        [{'complexity':3,'loss':1,'equation':'exact'}],
        [{'complexity':3,'loss':.5,'equation':'approximation'}],
        [{'complexity':1,'loss':2,'equation':'constant'},
         {'complexity':3,'loss':.5,'equation':'approximation'}]]
    current=[]
    for i,frontier in enumerate(frontiers):
        current=curve.merge_frontiers([current,frontier])
        assert curve.compact(current)==curve.compact(curve.merge_frontiers(frontiers[:i+1]))
    assert 'exact' not in [r['equation'] for r in current]


def test_audit_correction_rebuilds_stale_search_boundary(tmp_path,monkeypatch):
    monkeypatch.setattr(curve,'OUT',tmp_path)
    snapshots=[{'seconds':n,'frontier':[{'frontier_index':0,'complexity':1,'equation':str(n)}]}
               for n in range(1,18)]
    cache={curve.digest(['task',s['frontier']]):{'classification':'exact' if i>=5 else 'miss'}
           for i,s in enumerate(snapshots)}
    trial={'id':0,'dataset':'task','seed':10000,'low':5,'high':6,'n_restarts':17,
           'final_positive':True,'history':[]}
    curve.write(tmp_path/'snapshots/000.json',snapshots)
    curve.write(tmp_path/'state.json',{'trials':[trial],'round':0,'cache':cache,'cost_usd':0})
    key=curve.digest(['task',snapshots[5]['frontier']])
    curve.write(tmp_path/'review_overrides.json',{key:{'classification':'near'}})
    rendered=[]
    monkeypatch.setattr(curve,'render',lambda state:rendered.append(state))
    curve.step(dry_run=True)
    assert (rendered[0]['trials'][0]['low'],rendered[0]['trials'][0]['high'])==(6,7)
