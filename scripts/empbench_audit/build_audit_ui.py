#!/usr/bin/env python3
"""Build a self-contained HTML page for human audit of the LLM recovery judge.

Reads a finished binary-search recovery review (scripts/review_srbench2_spliced_portfolio.py)
and emits one HTML file with, for every trial of one method, the frontier the
judge's decision refers to at (a) the end of the search and (b) an early
checkpoint (default 30 s), next to the judge's verdict. No API calls, no jobs.

    python scripts/empbench_audit/build_audit_ui.py

The early-checkpoint verdict is the one the solve-over-time figure plots:
recovered iff the binary-searched first recovery is <= the checkpoint. That
frontier itself was often not shown to the judge (binary search assumes
recovery persists); each item records whether its verdict is direct or implied.
"""
import argparse
import datetime
import hashlib
import json
import multiprocessing
import os
import random
import signal
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import review_srbench2_spliced_portfolio as adapter  # noqa: E402  (sets engine.EXTRA on import)
from manual_solve_check import developer_prompt  # noqa: E402
from splice_srbench2_first_restart import snapshots as source_snapshots  # noqa: E402

engine = adapter.engine

# Feature order of the SRBench2 runs (pmlb/datasets/first_principles_*/metadata.yaml).
# `target`/`lhs` use the page's expression syntax so they render like the frontier.
DATASETS = {
    'first_principles_hubble': dict(
        name="Hubble's law", lhs='v', target='c*x0',
        vars=[('D', 'distance of the galaxy (Mpc)')], positive=True),
    'first_principles_kepler': dict(
        name="Kepler's third law", lhs='P', target='c*x0^(3/2)',
        vars=[('a', 'semi-major axis (AU)')], positive=True),
    'first_principles_newton': dict(
        name="Newton's law of gravitation", lhs='log(F)', target='c0 + log(x1) + log(x2) - 2*log(x0)',
        vars=[('r', 'distance (m)'), ('m1', 'mass of first body (kg)'), ('m2', 'mass of second body (kg)')],
        positive=True),
    'first_principles_ideal_gas': dict(
        name='Ideal gas law', lhs='log(P)', target='c0 + log(x0) + log(x1) - log(x2)',
        vars=[('n', 'number density'), ('T', 'temperature'), ('V', 'volume')], positive=True),
    'first_principles_planck': dict(
        name="Planck's law", lhs='log(B)', target='log(c0*x0^3/(exp(c1*x0/x1) - 1))',
        vars=[('nu', 'frequency (Hz)'), ('T', 'temperature')], positive=True),
    'first_principles_rydberg': dict(
        name='Rydberg formula', lhs='log(lambda)', target='log(c/(1/x0^2 - 1/x1^2))',
        vars=[('n1', 'principal quantum number'), ('n2', 'principal quantum number after the transition')],
        positive=True),
    'first_principles_schechter': dict(
        name='Schechter function', lhs='log(phi)', target='c0 + alpha*log(x0) - x0/c1',
        vars=[('L', 'luminosity')], positive=True),
    'first_principles_leavitt': dict(
        name="Leavitt's law", lhs='M', target='c0 + c1*x0',
        vars=[('log10P', 'log10 of the period in days (already a logarithm)')], positive=True),
    # Bode's planet index is -1000 for Mercury, so no positivity assumption.
    'first_principles_bode': dict(
        name="Bode's law", lhs='a', target='c0 + c1*exp(c2*x0)',
        vars=[('n', 'planet index (Mercury is -1000)')], positive=False),
}

# Cheat sheets: rewrites of the target a frontier row may use (`same`), and look-alikes that are
# different functions (`differ`). check_cheats() verifies both lists numerically at build time.
DATASETS['first_principles_planck']['cheats'] = dict(
    same=[
        ('c + 3*log(x0) - log(exp(c1*x0/x1) - 1)', 'the log of the quotient, split up'),
        ('c + 3*log(x0) - c1*x0/x1 - log(1 - exp(-c1*x0/x1))',
         'exp factored out of the denominator: the Wien terms plus a correction term that must be there'),
        ('log(c0*x0^3*exp(-c1*x0/x1)/(1 - exp(-c1*x0/x1)))', 'the same thing inside one log'),
        ('c - log((exp(c1*x0/x1) - 1)/x0^3)', 'quotient inverted inside the log'),
    ],
    differ=[
        ('c + 3*log(x0) - c1*x0/x1', 'Wien approximation: the term −log(1 − exp(−c1·ν/T)) is missing. Near, not recovered.'),
        ('c + 2*log(x0) + log(x1)', 'Rayleigh-Jeans limit. Near, not recovered.'),
        ('c + 3*log(x0) - log(k*exp(c1*x0/x1) - 1)',
         'the exp and the 1 must have equal weight; k other than 1 (or a constant added inside the exp) is a different function'),
        ('c + q*log(x0) - log(exp(c1*x0/x1) - 1)', 'the power of ν must be exactly 3'),
    ],
    note='Spellings that change nothing: cube(x), x*square(x) or x*x*x for the cube; 3*log(x) as log(x) + log(square(x)); '
         'c1·ν/T as ν/(T·k) or (ν/T)/k; −log(u) as log(1/u); any constant factor inside a log is the same as an added constant outside.')
DATASETS['first_principles_rydberg']['cheats'] = dict(
    same=[
        ('c - log(1/x0^2 - 1/x1^2)', 'the constant pulled out of the log'),
        ('c + 2*log(x0) - log(1 - (x0/x1)^2)', '1/n₁² factored out'),
        ('c + 2*log(x1) - log((x1/x0)^2 - 1)', '1/n₂² factored out'),
        ('c + 2*log(x0) + 2*log(x1) - log(x1^2 - x0^2)', 'common denominator'),
        ('c + 2*log(x0) + 2*log(x1) - log(x1 - x0) - log(x1 + x0)', 'difference of squares factored'),
        ('log(c*x0^2*x1^2/(x1^2 - x0^2))', 'common denominator, inside one log'),
        ('-log((k/x0)^2 - (k/x1)^2)', 'the constant inside both squares, with the same k in both'),
    ],
    differ=[
        ('c - log(1/x0^2 - q/x1^2)', 'the two terms must have equal weight: q = 1.0004 is not a recovery'),
        ('c + 2*log(x0) - log(q - (x0/x1)^2)', 'the same failure in factored form: the constant must be exactly 1'),
        ('c + log(1/x0^2 - 1/x1^2)', 'wrong sign on the log: this is log(1/λ)'),
        ('c - log(1/x0^p - 1/x1^p)', 'both powers must be exactly 2'),
    ],
    note='Spellings that change nothing: square(1/n) or 1/square(n) or 1/(n*n) for 1/n²; −log(u) as log(1/u); '
         'any constant factor inside a log is the same as an added constant outside. n₂ > n₁, so 1/n₁² − 1/n₂² is positive.')

SIMPLIFY_TIMEOUT = 10
LOCALS = {'square': lambda x: x**2, 'cube': lambda x: x**3, 'neg': lambda x: -x, 'inv': lambda x: 1/x}


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def _intify(expr):
    """Floats that are exactly integers print as integers; the page shows exactness."""
    return expr.xreplace({f: sp.Integer(int(f)) for f in expr.atoms(sp.Float) if f == int(f)})


def simplify_one(job):
    """Reading aid only: a sympy-simplified, log-expanded form, or None if it is no shorter."""
    equation, positive = job
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(SIMPLIFY_TIMEOUT)
    try:
        symbols = {f'x{i}': sp.Symbol(f'x{i}', real=True, positive=positive or None) for i in range(6)}
        expr = _intify(sp.sympify(equation, locals={**LOCALS, **symbols}))
        for transform in (sp.simplify, sp.expand):
            out = _intify(sp.expand_log(transform(expr), force=True))
            # force=True can pull a sign out of a log and leave I*pi behind; such forms mislead.
            if out.has(sp.I, sp.zoo, sp.nan, sp.oo, -sp.oo):
                continue
            text = sp.sstr(out)
            if len(text) <= 1.15 * len(equation) + 5:
                return equation, positive, text
        return equation, positive, None
    except Exception:  # Timeout, parse failures: simply no aid for this row.
        return equation, positive, None
    finally:
        signal.alarm(0)


def check_cheats(key):
    """Each `same` form must differ from the target by a constant; each `differ` form must not."""
    import math
    meta, rng = DATASETS[key], random.Random(0)
    constants = dict(c=0.7, c0=1.3, c1=0.9, k=1.6, q=1.07, p=2.3)
    points = []
    for _ in range(40):  # x1 > x0 > 0 keeps every log argument above positive
        x0 = rng.uniform(1, 3)
        points.append(dict(x0=x0, x1=x0 + rng.uniform(0.5, 3)))

    def spread(form):
        scope = dict(log=math.log, exp=math.exp, sqrt=math.sqrt, __builtins__={})
        gaps = [eval(form.replace('^', '**'), scope, {**constants, **p}) -
                eval(meta['target'].replace('^', '**'), scope, {**constants, **p}) for p in points]
        return max(gaps) - min(gaps)

    for form, _ in meta['cheats']['same']:
        if spread(form) > 1e-9:
            raise SystemExit(f'{key}: not equivalent to the target: {form}')
    for form, _ in meta['cheats']['differ']:
        if spread(form) < 1e-3:
            raise SystemExit(f'{key}: listed as different but equals the target up to a constant: {form}')


def plot_time(snapshot):
    """Clock used by the figure: scheduled time for 5 s captures, actual time for restart ends."""
    return snapshot['scheduled_seconds'] if snapshot['kind'] == 'first_restart_snapshot' else snapshot['seconds']


def plotted_seconds(record):
    """Same rule as figures/solve_over_time/solve_over_time.py."""
    if not record['final_positive']:
        return None
    if record.get('first_solve_kind') == 'first_restart_snapshot':
        return min(float(record['first_solve_scheduled_seconds']), 3600.)
    return record['first_solve_budget_seconds']


def review_fields(review, cache):
    out = {k: review.get(k) for k in ('classification', 'original_classification', 'matching_equation',
                                      'explanation', 'source', 'correction_note')}
    out['indices'] = review.get('best_frontier_indices') or []
    if review.get('source') == 'known_exact_equation':
        out['evidence_explanation'] = cache[review['evidence']]['explanation']
    return out


def audit_order(items, seed):
    """Stratified shuffle: every prefix of 18 covers each (timepoint, dataset) once."""
    rng = random.Random(seed)
    strata = {}
    for item in items:
        strata.setdefault((item['tp'], item['dataset']), []).append(item['id'])
    for key in sorted(strata):
        rng.shuffle(strata[key])
    order = []
    for depth in range(max(map(len, strata.values()))):
        layer = [strata[key][depth] for key in sorted(strata) if depth < len(strata[key])]
        rng.shuffle(layer)
        order += layer
    return order


def carried_equation(item):
    """The accepted equation, if this early verdict merely copies an earlier acceptance of it.

    True when the frontier still holds, verbatim, an equation the judge already accepted for this
    problem: either the binary search carried the first recovery forward, or the engine reused a
    known exact equation. If that acceptance was right, so is this verdict; nothing new was judged.
    """
    llm = item['llm']
    if item['tp'] == '1h':
        return None
    if llm['provenance'] == 'direct':
        return llm['matching_equation'] if llm.get('source') == 'known_exact_equation' else None
    return llm['witness_equation'] if llm['positive'] and llm['witness_index'] is not None else None


def by_problem(order):
    """Queue order: one problem at a time, keeping the shuffle within each problem.

    The first 2k items of a problem are then k random seeds at each checkpoint.
    """
    rank = {key: i for i, key in enumerate(DATASETS)}
    return sorted(order, key=lambda item_id: rank[item_id.split('|')[0]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recovery-dir', type=Path, default=ROOT / 'runs/srbench2_9-17_1m_spliced_terra_recovery')
    parser.add_argument('--source-run', type=Path, default=ROOT / 'runs/709715-srbench2_9-17_60m_1m_spliced_snap5',
                        help='Run whose srbench_full_results.json the review was built from (for losses)')
    parser.add_argument('--method', default='709715', help='Method label inside the recovery review')
    parser.add_argument('--method-name', default='Evolved PySR (709715)')
    parser.add_argument('--early-seconds', type=float, default=30)
    parser.add_argument('--order-seed', type=int, default=0)
    parser.add_argument('--keep-carried', action='store_true',
                        help='Also queue early verdicts that only copy an earlier acceptance of an equation still in the frontier')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/empbench_human_audit/empbench_audit.html')
    args = parser.parse_args()

    recovery = args.recovery_dir
    state = json.loads((recovery / 'state.json').read_text())
    cache = state['cache']
    config = {'model': engine.MODEL, 'extra': engine.EXTRA, 'targets': engine.TARGETS, 'rubric': engine.RUBRIC,
              'schema': engine.REVIEW_SCHEMA, 'max_tokens': engine.MAX_TOKENS, 'adapter_version': 1}
    if engine.digest(config) != state['config_hash']:
        raise SystemExit('Judge prompt in the repository differs from the one this review used')
    records = [r for r in json.loads((recovery / 'first_recovery.json').read_text()) if r['method'] == args.method]
    if not records:
        raise SystemExit(f'No trials for method {args.method!r}')
    source = {(r['dataset'], r['seed']): r for r in
              json.loads((args.source_run / 'srbench_full_results.json').read_text())['results'].values()}
    early = args.early_seconds
    early_label = f'{early:g}s'

    items = []
    for record in sorted(records, key=lambda r: (r['dataset'], r['seed'])):
        dataset, seed = record['dataset'], record['seed']
        saved = json.loads((recovery / 'snapshots' / f"{record['id']:03d}.json").read_text())
        full, _ = source_snapshots(source[dataset, seed])
        if [engine.compact(o['frontier']) for o in full] != [o['frontier'] for o in saved]:
            raise SystemExit(f'Source run does not reproduce the reviewed snapshots: {dataset} {seed}')
        n_restarts = sum(o['kind'] == 'restart_end' for o in saved)
        solved_at = plotted_seconds(record)

        def rows(index):
            return [[row['complexity'], row['equation'], row.get('loss')] for row in full[index]['frontier']]

        def snapshot_info(index):
            s = saved[index]
            return dict(kind=s['kind'], seconds=s['seconds'], scheduled=s.get('scheduled_seconds'),
                        restart_index=s.get('restart_index'), n_restarts=n_restarts)

        # (1) End of search.
        final_review = cache[engine.digest([dataset, saved[-1]['frontier']])]
        llm = dict(review_fields(final_review, cache), positive=final_review['classification'] == 'exact',
                   provenance='direct', first_solve_seconds=solved_at)
        items.append(dict(id=f'{dataset}|{seed}|1h', dataset=dataset, seed=seed, tp='1h',
                          snap=snapshot_info(len(saved) - 1), rows=rows(len(saved) - 1), llm=llm))

        # (2) Early checkpoint: the latest frontier the figure's clock places at or before it.
        candidates = [i for i, s in enumerate(saved) if plot_time(s) <= early]
        if not candidates:
            raise SystemExit(f'No frontier at or before {early:g} s: {dataset} {seed}')
        index = candidates[-1]
        frontier = saved[index]['frontier']
        positive = solved_at is not None and solved_at <= early
        llm = dict(positive=positive, first_solve_seconds=solved_at,
                   last_negative_seconds=record.get('last_negative_seconds'))
        direct = cache.get(engine.digest([dataset, frontier]))
        if direct:
            if (direct['classification'] == 'exact') != positive:
                raise SystemExit(f'Direct review contradicts the plotted verdict: {dataset} {seed}')
            llm.update(review_fields(direct, cache), provenance='direct')
        else:
            llm.update(provenance='implied', final_classification=record['final_classification'])
            if positive:
                witness = record['recovery_review']
                llm.update(witness_equation=witness['matching_equation'],
                           witness_explanation=witness['explanation'],
                           witness_index=next((row['frontier_index'] for row in frontier
                                               if row['equation'] == witness['matching_equation']), None))
        snap = snapshot_info(index)
        snap['missing'] = [m['scheduled_seconds'] for m in record['missing_snapshots']
                           if plot_time(saved[index]) < m['scheduled_seconds'] <= early]
        items.append(dict(id=f'{dataset}|{seed}|{early_label}', dataset=dataset, seed=seed, tp=early_label,
                          snap=snap, rows=rows(index), llm=llm))

    # Shuffle before dropping anything, so the remaining items keep their relative order.
    shuffle = audit_order(items, args.order_seed)
    omitted = None
    if not args.keep_carried:
        final_pick = {i['id'].rsplit('|', 1)[0]: i['llm'].get('matching_equation') for i in items if i['tp'] == '1h'}
        skip = {i['id']: carried_equation(i) for i in items if carried_equation(i) is not None}
        omitted = dict(count=len(skip), checkpoint=early_label,
                       equation_is_final_pick=sum(eq == final_pick[i.rsplit('|', 1)[0]] for i, eq in skip.items()))
        items = [i for i in items if i['id'] not in skip]
        shuffle = [i for i in shuffle if i not in skip]

    jobs = sorted({(row[1], DATASETS[item['dataset']]['positive']) for item in items for row in item['rows']})
    workers = min(8, len(os.sched_getaffinity(0)))
    with multiprocessing.Pool(workers) as pool:
        simplified = {(eq, pos): text for eq, pos, text in pool.imap_unordered(simplify_one, jobs, chunksize=8)}
    for item in items:
        positive_inputs = DATASETS[item['dataset']]['positive']
        for row in item['rows']:
            row.append(simplified[row[1], positive_inputs])

    present = {item['dataset'] for item in items}
    datasets = {}
    for key in (k for k in DATASETS if k in present):
        meta = DATASETS[key]
        datasets[key] = dict(name=meta['name'], lhs=meta['lhs'], target=meta['target'],
                             vars=[dict(sym=s, desc=d) for s, d in meta['vars']],
                             positive=meta['positive'], kind=engine.TARGETS[key]['kind'],
                             accepted=engine.TARGETS[key]['target'],
                             prompt=developer_prompt(key) + engine.EXTRA)
        if 'cheats' in meta:
            check_cheats(key)
            datasets[key]['cheats'] = meta['cheats']
    data = dict(
        build=dict(created=datetime.datetime.now().isoformat(timespec='seconds'),
                   recovery_dir=str(recovery.relative_to(ROOT)), source_run=str(args.source_run.relative_to(ROOT)),
                   method=args.method, method_name=args.method_name, judge_model=engine.MODEL,
                   early_seconds=early, early_label=early_label, order_seed=args.order_seed, omitted=omitted,
                   config_hash=state['config_hash']),
        extra=engine.EXTRA.strip(), datasets=datasets, items=items, order=by_problem(shuffle))
    # The id keys saved decisions: it changes only when the audited content or the shuffle does,
    # not when the queue is regrouped.
    data['build']['id'] = hashlib.sha256(json.dumps(
        [data['build']['config_hash'], shuffle, [[i['id'], i['rows'], i['llm']] for i in items]],
        sort_keys=True).encode()).hexdigest()[:12]

    template = (HERE / 'template.html').read_text()
    marker = '/*__AUDIT_DATA__*/null'
    if template.count(marker) != 1:
        raise SystemExit('template.html must contain the data marker exactly once')
    payload = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(template.replace(marker, payload))

    for tp in ('1h', early_label):
        group = [i for i in items if i['tp'] == tp]
        implied = sum(i['llm']['provenance'] == 'implied' for i in group)
        print(f"{tp}: {sum(i['llm']['positive'] for i in group)}/{len(group)} queued items recovered per the judge; "
              f"{implied} verdicts implied by binary search")
    if omitted:
        print(f"omitted {omitted['count']} {early_label} items whose verdict copies an earlier acceptance of an equation still "
              f"in the frontier; for {omitted['equation_is_final_pick']} of them that equation is also the judge's 1h pick")
    aided = sum(row[3] is not None for i in items for row in i['rows'])
    print(f"{aided}/{sum(len(i['rows']) for i in items)} rows have a simplified form")
    print(f"build {data['build']['id']} -> {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == '__main__':
    main()
