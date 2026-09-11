#!/usr/bin/env python3
"""Validate completed approximate recovery records against snapshots and sources."""
import csv
import json
from collections import Counter
from srbench2_portfolio_recovery import ROOT, OUT, RUNS, digest, write


def main():
    state=json.loads((OUT/'state.json').read_text())
    records=json.loads((OUT/'first_recovery.json').read_text())
    assert len(records)==200
    assert len({(r['method'],r['dataset'],r['seed']) for r in records})==200
    totals=Counter()
    for path,signature in state['fingerprints'].items():
        source=ROOT/path
        if source.is_dir():source=source/'srbench_full_results.json'
        assert digest(json.loads(source.read_text()))==signature, path
    for r in records:
        snapshots=json.loads((OUT/'snapshots'/f"{r['id']:03d}.json").read_text())
        assert len(snapshots)==r['n_restarts']
        assert all(a['seconds']<=b['seconds'] for a,b in zip(snapshots,snapshots[1:]))
        if not r['final_positive']:
            assert r['first_solve_seconds'] is None and not r['history']
            continue
        assert r['high']==r['low']+1
        high=snapshots[r['high']-1]
        checked=state['cache'][digest([r['dataset'],high['frontier']])]
        assert checked['classification']=='exact'
        assert checked['matching_equation'] in [e['equation'] for e in high['frontier']]
        if r['low']:
            low=snapshots[r['low']-1]
            assert state['cache'][digest([r['dataset'],low['frontier']])]['classification']!='exact'
        assert r['first_solve_seconds']==high['seconds']
        assert r['first_solve_budget_seconds']==min(high['seconds'],3600)
        totals[r['method']]+=1
    assert totals=={'Baseline':82,'709715':74}
    billed=sum(sum(r['cost_usd'] for r in json.loads(p.read_text()).values())
               for p in (OUT/'rounds').glob('*/reviews.json'))
    assert abs(billed-state['cost_usd'])<1e-9
    table=list(csv.DictReader((ROOT/'figures/srbench2_portfolio_solve_over_time/solve_rate.csv').open()))
    assert len(table)==122
    for row in table:
        count=sum(r['method']==row['method'] and r['first_solve_budget_seconds'] is not None
                  and r['first_solve_budget_seconds']<=60*int(row['minutes']) for r in records)
        assert count==int(row['solved_trials'])
        assert float(row['mean_tasks_solved'])==count/10
    result={'trials':200,'final_positives':dict(totals),'new_review_cost_usd':billed,
            'rounds':state['round'],'validated':'sources, unique trials, adjacent binary boundaries, equation membership, timing, costs, CSV',
            'limitation':'Monotonicity is assumed, not verified; temporary recoveries can be missed.'}
    write(OUT/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
