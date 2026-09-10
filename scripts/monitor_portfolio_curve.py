#!/usr/bin/env python3
"""Monitor the authorized recovery-analysis array and resume timed-out shards.

Every SLURM action is appended as a guarded mode in submit_jobs.sh and executed
through that file. Does not touch other jobs. Run with the array ID explicitly.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.check_output(args, text=True, cwd=ROOT)


def journal(commands):
    mode = 'portfolio-monitor-' + datetime.now().strftime('%Y%m%d-%H%M%S')
    path = ROOT / 'submit_jobs.sh'
    body = '\n'.join('    ' + command for command in commands)
    block = f'if [[ "${{1:-}}" == "{mode}" ]]; then\n{body}\n    exit\nfi\n\n'
    source = path.read_text()
    heading = datetime.now().strftime('# %-m/%-d/%y') + '\n'
    if heading in source:
        source = source.replace(heading, heading + block, 1)
    else:
        source = source.replace('#!/usr/bin/env bash\n', '#!/usr/bin/env bash\n\n' + heading + block, 1)
    path.write_text(source)
    result = subprocess.run(['bash', 'submit_jobs.sh', mode], cwd=ROOT,
                            text=True, capture_output=True)
    print(mode, result.stdout, result.stderr, flush=True)
    if result.returncode:
        raise RuntimeError('SLURM action needs inspection; journal retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--array', type=int, required=True)
    args = parser.parse_args()
    output = ROOT / 'reports/portfolio_solve_over_time'
    plan = json.loads((output / 'group_plan.json').read_text())
    while True:
        completed = {i for i, item in enumerate(plan) if
                     (output / 'group_shards/datasets' / f'{item["cache_key"]}.json').exists()}
        states = run('sacct', '-j', str(args.array), '-X', '-n', '-P', '--format=JobID,State')
        active = set(run('squeue', '-r', '-j', str(args.array), '-h', '-o%i').split())
        counts = Counter()
        retry = []
        errors = []
        for line in states.splitlines():
            job, state, *_ = line.split('|')
            counts[state] += 1
            match = re.fullmatch(str(args.array) + r'_(\d+)', job)
            if not match or job in active or int(match[1]) in completed:
                continue
            if state == 'TIMEOUT':
                retry.append(job)
            elif state not in ('COMPLETED', 'RUNNING', 'PENDING'):
                errors.append((job, state))
        print(datetime.now().isoformat(timespec='seconds'),
              f'groups={len(completed)}/{len(plan)}', dict(counts), flush=True)
        if errors:
            raise RuntimeError(f'Unexpected failures: {errors}')
        if retry:
            journal(['for portfolio_group_job in ' + ' '.join(retry) +
                     '; do scontrol requeue "$portfolio_group_job" || exit; done'])
        if len(completed) == len(plan):
            print(run('python', 'scripts/analyze_portfolio_solve_over_time.py',
                      '--collect-groups', '--render-only'), flush=True)
            return
        time.sleep(45)


if __name__ == '__main__':
    main()
