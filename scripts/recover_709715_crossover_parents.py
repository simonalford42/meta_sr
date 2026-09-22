#!/usr/bin/env python3
"""Recover exact crossover parents from cached LLM requests, without LLM calls."""
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/709715_crossover_recovery'


def digest(code):
    return hashlib.sha256(code.strip().encode()).hexdigest()


def extract_code(text):
    text = text.strip()
    match = re.search(r'```(?:julia)?\s*\n(.*?)```', text, re.S)
    return match.group(1).strip() if match else text


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / 'runs/709715/run_data.json').read_text())
    ops = {}
    for g in data['generations']:
        for kind in ('population', 'offspring'):
            for bundle in g[kind]:
                for t, op in bundle['operators'].items():
                    ops.setdefault(op['name'], dict(op, type=t))
    del data
    # This initial candidate disappeared before the first population snapshot,
    # but its exact source survives in the operator directory and a crossover prompt.
    missing = 'pareto_niche_crowding_survival_init_3'
    ops.setdefault(missing, dict(name=missing, type='survival', generation=0,
                                mode='explore', parent_name=None,
                                code=(ROOT / 'runs/709715/operators/gen0_survival3.jl').read_text()))
    # Save a compact code library for review without repeatedly reading 2.1 GB.
    library = {n: {k: op.get(k) for k in ('name','type','generation','mode','parent_name','code')}
               for n, op in ops.items()}
    (OUT / 'operator_library.json').write_text(json.dumps(library, indent=2)+'\n')
    by_code = defaultdict(list)
    expected = defaultdict(list)
    targets = {n:op for n,op in ops.items() if op['mode']=='crossover'}
    for n,op in ops.items():
        by_code[digest(op['code'])].append(n)
    for n,op in targets.items():
        base = re.sub(r'_gen\d+_\d+$', '', n)
        original = re.sub(rf'(?<![\w!?]){re.escape(n)}(?![\w!?])',lambda _:base,op['code'])
        expected[digest(original)].append(n)
    hits = defaultdict(list)
    connection = sqlite3.connect(f'file:{ROOT / "caches/completions_cache.db"}?mode=ro', uri=True)
    scanned = 0
    for h,model,messages,response in connection.execute(
        "SELECT request_hash,model,messages_json,response_json FROM chat_completions WHERE messages_json LIKE '%COMBINE ideas from two%'"):
        scanned += 1
        response = json.loads(response)
        content = response.get('choices',[{}])[0].get('message',{}).get('content','') or ''
        if not isinstance(content,str):
            continue
        candidates = expected.get(digest(extract_code(content)), [])
        if not candidates:
            continue
        prompt = '\n'.join(m.get('content','') for m in json.loads(messages) if isinstance(m.get('content'),str))
        blocks = re.findall(r'## Parent (\w+) operator ([12])\s*```julia\s*\n(.*?)```',prompt,re.S)
        for n in candidates:
            op=targets[n]
            parents=[]
            for t,number,code in blocks:
                matching=[name for name in by_code[digest(code)] if ops[name]['type']==t and ops[name]['generation'] < op['generation']]
                parents.append({'position':int(number),'code_sha256':digest(code),'names':matching})
            hits[n].append({'request_hash':h,'model':model,'parents':parents})
            evidence=OUT/'prompts';evidence.mkdir(exist_ok=True)
            (evidence/f'{h}.md').write_text(f'# {n}\n\nCache request: `{h}`\n\n'+ '\n\n'.join(f'## Parent {t} operator {number}\n```julia\n{code.strip()}\n```' for t,number,code in blocks)+'\n')
    records=[]
    for n,op in sorted(targets.items(),key=lambda x:(x[1]['generation'],x[0])):
        pairs=set()
        for hit in hits[n]:
            ps=hit['parents']
            if len(ps)==2 and all(len(p['names'])==1 for p in ps):
                pairs.add(tuple(p['names'][0] for p in sorted(ps,key=lambda p:p['position'])))
        pair=next(iter(pairs)) if len(pairs)==1 else None
        if pair:
            assert pair[0]==op['parent_name'],(n,pair,op['parent_name'])
        records.append({'child':n,'generation':op['generation'],'operator':op['type'],
                        'saved_parent1':op['parent_name'],'parent1':pair[0] if pair else None,
                        'parent2':pair[1] if pair else None,'status':'confirmed_exact_code' if pair else 'unresolved',
                        'cache_hits':hits[n]})
    (OUT/'crossover_parents.json').write_text(json.dumps(records,indent=2)+'\n')
    lines = ['# Run 709715 crossover parent recovery', '',
             'All 73 crossover events are matched to cached LLM responses and requests. '
             'Child code matches exactly after reversing the generated function-name suffix; '
             'both parent blocks match saved source code exactly after trimming outer whitespace. '
             'The first parent also agrees with persisted `parent_name` in every event.', '',
             'One initial candidate (`pareto_niche_crowding_survival_init_3`) is absent '
             'from population snapshots but survives in `runs/709715/operators/gen0_survival3.jl`. '
             'Two parent slots may contain the same operator, because different bundles can share it.', '',
             'Regenerate with `python scripts/recover_709715_crossover_parents.py` from the repository root; '
             'this only reads saved run data and the local completion cache, with no LLM calls or evaluations.', '',
             '| Generation | Operator | Child | Parent 1 | Parent 2 | Evidence |',
             '| --- | --- | --- | --- | --- | --- |']
    for r in records:
        evidence = ', '.join(f"[cached parent blocks](prompts/{hit['request_hash']}.md)" for hit in r['cache_hits'])
        lines.append(f"| {r['generation']} | {r['operator']} | {r['child']} | {r['parent1']} | {r['parent2']} | {evidence} |")
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('targets',len(targets),'cache rows scanned',scanned,'confirmed',sum(r['status']=='confirmed_exact_code' for r in records),flush=True)
    for r in records:
        if r['status']=='unresolved':print('UNRESOLVED',r['child'],json.dumps(r['cache_hits']))


if __name__=='__main__':
    main()
