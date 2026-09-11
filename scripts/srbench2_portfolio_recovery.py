#!/usr/bin/env python3
"""Approximate first recovery by binary searching cumulative native-loss frontiers.

Run --step repeatedly; each invocation submits or polls one checkpointed batch.
No Slurm or SR searches. Final negatives are deliberately not searched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frontier_aggregation import merge_frontiers
from manual_solve_check import (build_request, estimate_cost_upper, OpenRouterHTTPClient,
    write_json_atomic as write, calculate_cost, _extract_output_text, _validate_review)

OUT = ROOT / 'reports/srbench2_portfolio_solve_over_time'
MODEL = 'openai/gpt-5.6-terra'
MAX_TOKENS = 3000
RUNS = {
    'Baseline': ('srb_base_port', 'runs/srbench2_9-8_gt_baseline_1core_l1_portfolio_1m_no_warmup'),
    '709715': ('srb_evo_port', 'runs/709715/srbench2_9-8_ground_truth_1core_portfolio_1m_no_warmup'),
}
EXTRA = '''\nAudit clarification: fitted constants are free ONLY in positions allowed by the reference.
Fixed powers and relative coefficients must match exactly; never round them to theoretical values.
For Rydberg, C-log(1/x0^2 - q/x1^2) is NOT exact unless q is exactly 1.
For ideal gas/Newton, fixed coefficients of logarithmic terms cannot vary.
An affine family can have zero intercept. This is structural recovery, not a raw-R2 threshold.
For supernova, require a reciprocal sum of two exponentials with opposite-sign rates;
a logarithmic spike, polynomial, or rational approximation is not exact.
'''


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def compact(rows):
    return [dict(frontier_index=i, complexity=r['complexity'], equation=r['equation'])
            for i, r in enumerate(rows)]


def initialize():
    if (OUT/'state.json').exists():
        return
    audit_path = ROOT/'analysis/benchmark_positive_audit_2026-09-08.json'
    audit = json.loads(audit_path.read_text())
    records = {(r['setup'],r['dataset'],r['seed']):r for r in audit['records']}
    trials = []
    cache = {}
    fingerprints = {str(audit_path.relative_to(ROOT)):digest(audit)}
    for method,(setup,path) in RUNS.items():
        raw = json.loads((ROOT/path/'srbench_full_results.json').read_text())
        fingerprints[path] = digest(raw)
        for key, result in sorted(raw['results'].items()):
            dataset,seed,noise = key.split('|')
            if dataset.endswith(('absorption','bode')):
                continue
            assert not result.get('error')
            review = records[setup,dataset,int(seed)]
            cumulative, snapshots, elapsed = [], [], 0.0
            for restart in result['portfolio']['restarts']:
                assert restart.get('search_runtime_seconds') is not None
                elapsed += max(0,float(restart['search_runtime_seconds']))
                if not restart.get('error'):
                    cumulative = merge_frontiers([cumulative, restart.get('pareto_frontier') or []])
                snapshots.append({'seconds':elapsed,'frontier':compact(cumulative)})
            assert compact(result['pareto_frontier']) == snapshots[-1]['frontier'], (method,key,'final frontier differs')
            positive = review['audited']=='exact'
            if positive:
                assert review['selected_equation'] in [r['equation'] for r in cumulative]
            trial = {'id':len(trials),'method':method,'dataset':dataset,'seed':int(seed),
                     'final_classification':review['audited'],'final_positive':positive,
                     'low':0,'high':len(snapshots),'n_restarts':len(snapshots),'history':[],
                     'budget_seconds':result['portfolio']['total_search_budget_seconds']}
            trials.append(trial)
            write(OUT/'snapshots'/f"{trial['id']:03d}.json",snapshots)
            cache[digest([dataset,snapshots[-1]['frontier']])] = {
                'classification':review['audited'],'matching_equation':review['selected_equation'],
                'explanation':review['reason'],'source':'audited_final','cost_usd':0}
    assert len(trials)==200
    write(OUT/'state.json',{'trials':trials,'round':0,'cost_usd':0,'cache':cache,
                           'fingerprints':fingerprints,'model':MODEL})
    print('Initialized 200 trials; final positives:',sum(t['final_positive'] for t in trials),flush=True)


def apply(t, midpoint, review, key):
    positive = review['classification']=='exact'
    t['history'].append({'restart_count':midpoint,'positive':positive,'cache_key':key})
    t['high' if positive else 'low'] = midpoint


def step(dry_run=False):
    initialize()
    state = json.loads((OUT/'state.json').read_text())
    round_dir = OUT/'rounds'/f"{state['round']:02d}"
    batch_path=round_dir/'batch.json'
    if batch_path.exists():
        saved=json.loads(batch_path.read_text())
        client=OpenRouterHTTPClient(os.environ['OPENROUTER_API_KEY'])
        batch=client.json_request('GET',f"/beta/batches/{saved['id']}")
        print('Round',state['round'],batch.get('status'),batch.get('request_counts'),flush=True)
        if batch.get('status')!='completed':
            if batch.get('status') in ('failed','expired','cancelled'): raise RuntimeError(batch['status'])
            return
        write(round_dir/'responses.json',batch)
        items=json.loads((round_dir/'items.json').read_text())
        envelopes={r['custom_id']:r for r in batch['results']}
        reviewed={}
        cost=0
        for key,item in items.items():
            envelope=envelopes[key]
            response=envelope.get('response') or {}
            body=response.get('body') or {}
            assert not envelope.get('error') and response.get('status_code',200)==200, key
            review=json.loads(_extract_output_text(body))
            _validate_review(review,len(item['frontier']))
            assert review['classification'] not in ('error','not_applicable','phenomenological_match'), key
            if review['classification']=='exact':
                assert review['matching_equation'] in [r['equation'] for r in item['frontier']]
                assert review['best_frontier_indices']
                assert any(item['frontier'][i]['equation']==review['matching_equation'] for i in review['best_frontier_indices'])
            review.update(usage=body.get('usage',{}),cost_usd=calculate_cost(body.get('usage',{}),MODEL),source='llm')
            reviewed[key]=review
            cost+=review['cost_usd']
        write(round_dir/'reviews.json',reviewed)
        state['cache'].update(reviewed)
        state['cost_usd']+=cost
        state['round']+=1
        print(f"Round cost ${cost:.4f}; cumulative ${state['cost_usd']:.4f}",flush=True)
        write(OUT/'state.json',state)
        round_dir=OUT/'rounds'/f"{state['round']:02d}"
    items={}
    for t in state['trials']:
        if not t['final_positive']: continue
        snapshots=json.loads((OUT/'snapshots'/f"{t['id']:03d}.json").read_text())
        while t['high']-t['low']>1:
            midpoint=(t['high']+t['low'])//2
            frontier=snapshots[midpoint-1]['frontier']
            key=digest([t['dataset'],frontier])
            if key in state['cache']:
                apply(t,midpoint,state['cache'][key],key)
                continue
            items[key]={'custom_id':key,'source_hash':key,'dataset':t['dataset'],
                        'seed':t['seed'],'noise':0,'frontier':frontier}
            break
    write(OUT/'state.json',state)
    if not items:
        render(state)
        print('COMPLETE',flush=True)
        return
    requests=[build_request(item,MODEL,'medium',MAX_TOKENS) for item in items.values()]
    for req in requests: req['body']['messages'][0]['content']+=EXTRA
    estimate=estimate_cost_upper(requests,MODEL,MAX_TOKENS)
    payload={'endpoint':'/v1/chat/completions','model':MODEL,'requests':requests}
    write(round_dir/'items.json',items)
    write(round_dir/'payload.json',payload)
    write(round_dir/'estimate.json',estimate)
    print('Prepared round',state['round'],len(items),'requests;',estimate,flush=True)
    if dry_run: return
    assert state['cost_usd']+estimate['maximum_cost_usd']<=15, 'Would exceed $15 budget'
    client=OpenRouterHTTPClient(os.environ['OPENROUTER_API_KEY'])
    batch=client.json_request('POST','/beta/batches',payload)
    write(round_dir/'batch.json',{'id':batch['id'],'status':batch.get('status')})
    print('Submitted',batch['id'],flush=True)


def render(state):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[]
    for t in state['trials']:
        r=dict(t)
        r['first_solve_seconds']=None
        r['first_solve_budget_seconds']=None
        if t['final_positive']:
            assert t['high']-t['low']==1
            snapshots=json.loads((OUT/'snapshots'/f"{t['id']:03d}.json").read_text())
            s=snapshots[t['high']-1]
            r['first_solve_seconds']=s['seconds']
            r['first_solve_budget_seconds']=min(s['seconds'],t['budget_seconds'])
            key=digest([t['dataset'],s['frontier']])
            assert state['cache'][key]['classification']=='exact'
            r['recovery_review']=state['cache'][key]
        records.append(r)
    write(OUT/'first_recovery.json',records)
    figdir=ROOT/'figures/srbench2_portfolio_solve_over_time'
    figdir.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,5.5))
    table=[]
    for method,color in [('Baseline','#3264ad'),('709715','#d45b24')]:
        rows=[r for r in records if r['method']==method]
        times=sorted(r['first_solve_budget_seconds']/60 for r in rows if r['final_positive'])
        xs=[0]+times+[60]; ys=[0]+[i/10 for i in range(1,len(times)+1)]+[len(times)/10]
        ax.step(xs,ys,where='post',label=f'{method} ({len(times)/10:.1f}/10 at 60 min)',color=color,lw=2)
        for minute in range(61):
            count=sum(t<=minute for t in times)
            table.append({'method':method,'minutes':minute,'solved_trials':count,'total_trials':100,'mean_tasks_solved':count/10})
    ax.set(xlim=(0,60),ylim=(0,10),xlabel='Cumulative search time (minutes)',ylabel='Mean tasks solved out of 10',title='SRBench2 · one-hour portfolios · 10 seeds')
    ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.text(.5,.025,'Approximate binary search of cumulative native-loss frontiers; final negatives excluded.\nRestart-end timing; warm-up excluded; exact reference-family recovery only.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.065,1,1))
    for ext in ['png','pdf']:fig.savefig(figdir/f'solve_rate.{ext}',dpi=180)
    with (figdir/'solve_rate.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# SRBench2 approximate portfolio recovery over time','',
        'Binary search assumes cumulative-frontier recovery persists. Final-negative trials are treated as never solved; temporary recoveries may be missed. This is not exhaustive ever-recovered scoring.','',
        'Native training loss selects each cumulative complexity–loss frontier. Audited final labels initialize the search; midpoint reviews use the same exact-family rubric with explicit fixed-coefficient constraints. No raw-R² gate or calibration-based score is used. Absorption and Bode are excluded.','',
        'Timing uses cumulative recorded search seconds at restart completion, excluding warm-up/scoring. Final overshoot is mapped to 3600 seconds.','',
        f"New LLM usage-based cost at stored batch rates: ${state['cost_usd']:.4f}. Model: {MODEL}, medium reasoning. All 200 final frontiers were reconstructed and matched against the saved aggregate before reusing labels.",'',
        '| Task | Baseline exact / 10 | 709715 exact / 10 |','|---|---:|---:|']
    for ds in sorted({r['dataset'] for r in records}):
        counts=[sum(r['final_positive'] for r in records if r['dataset']==ds and r['method']==m) for m in RUNS]
        lines.append(f"| {ds} | {counts[0]} | {counts[1]} |")
    lines+=['','![Recovery curve](../../figures/srbench2_portfolio_solve_over_time/solve_rate.png)','',
            'Reproduce/resume: `python scripts/srbench2_portfolio_recovery.py --step`. Requests, responses, frontier snapshots, decisions, and source fingerprints are retained in this directory.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--run',action='store_true',help='Resume all rounds, polling every 30 seconds')
    args=parser.parse_args()
    if args.run:
        while not (OUT/'first_recovery.json').exists():
            try:
                step()
            except RuntimeError as exc:
                if 'failed (404)' not in str(exc):
                    raise
                print('Batch not visible yet; retrying.',flush=True)
            if not (OUT/'first_recovery.json').exists():
                time.sleep(30)
    else:
        step(args.dry_run)
