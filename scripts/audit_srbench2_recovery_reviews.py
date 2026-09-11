#!/usr/bin/env python3
"""Bounded algebra checks of LLM-selected positives; no new frontier search/API calls."""
import json
import signal
from pathlib import Path
from srbench2_portfolio_recovery import OUT, write, digest
from audit_benchmark_positives_2026_09_08 import check_equation


def alarm(*args):raise TimeoutError('3-second selected-equation check')


def main():
    path=OUT/'positive_check.json'
    cache=json.loads(path.read_text()) if path.exists() else {}
    signal.signal(signal.SIGALRM,alarm)
    for p in sorted((OUT/'rounds').glob('*/reviews.json')):
        items=json.loads((p.parent/'items.json').read_text())
        for key,r in json.loads(p.read_text()).items():
            if r['classification']!='exact':continue
            dataset=items[key]['dataset'];equation=r['matching_equation']
            identity=digest([dataset,equation])
            if identity in cache:continue
            record={'dataset':dataset,'equation':equation,'request':key}
            try:
                signal.setitimer(signal.ITIMER_REAL,3)
                ok,reason=check_equation(dataset,equation)
                record.update(verified=ok,reason=reason)
            except Exception as exc:
                record.update(verified=None,reason=type(exc).__name__+': '+str(exc))
            finally:signal.setitimer(signal.ITIMER_REAL,0)
            cache[identity]=record
            write(path,cache)
    bad=[r for r in cache.values() if r['verified'] is not True]
    print('Checked',len(cache),'unique selected equations; unresolved/failed',len(bad))
    print(json.dumps(bad,indent=2))


if __name__=='__main__':main()
