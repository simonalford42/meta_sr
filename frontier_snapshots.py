"""Observe native PySR frontier CSVs without interrupting a running fit.

A separate lightweight process avoids Python/Julia GIL interactions. Timestamps
are wall seconds since model.fit was called (including fit startup), not CPU
seconds or inferred discovery times. Native publication age is saved explicitly.
"""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def read_frontier(directory, started):
    """Accept only matching primary/backup files, avoiding partially written CSVs."""
    paths = list(Path(directory).glob('*/hall_of_fame.csv'))
    for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            raw = path.read_bytes()
            if not raw.endswith(b'\n') or raw != Path(str(path)+'.bak').read_bytes():
                continue
            reader = csv.DictReader(io.StringIO(raw.decode()), strict=True)
            if reader.fieldnames != ['Complexity', 'Loss', 'Equation']:
                continue
            rows = []
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError('Incomplete CSV row')
                c, loss = int(row['Complexity']), float(row['Loss'])
                if not math.isfinite(loss) or c < 0 or not row['Equation']:
                    raise ValueError('Invalid frontier row')
                rows.append(dict(complexity=c, loss=loss, equation=row['Equation']))
            if any(a['complexity'] >= b['complexity'] or a['loss'] <= b['loss'] for a,b in zip(rows, rows[1:])):
                raise ValueError('Not a native Pareto frontier')
            return {'status':'ok', 'equations':rows, 'source_file':str(path),
                    'source_updated_elapsed_seconds':max(0, path.stat().st_mtime-started)}
        except (OSError, ValueError, csv.Error, UnicodeError):
            continue
    return {'status':'unavailable', 'equations':[], 'source_file':None,
            'source_updated_elapsed_seconds':None}


def watch(directory, destination, stop, interval, started, parent_pid):
    due = interval
    previous = 0.0
    def capture(final=False):
        nonlocal previous
        record = read_frontier(directory, started)
        # Brief retries if a scheduled read overlaps a native file write.
        for _ in range(20):
            if record['status']=='ok':break
            time.sleep(.01)
            record=read_frontier(directory, started)
        elapsed = time.time()-started
        record.update(elapsed_seconds=elapsed, scheduled_seconds=None if final else due,
                      milestone_kind='fit_wall_time', chunk_runtime=elapsed-previous,
                      final=final, interval_seconds=interval)
        with open(destination, 'a') as f:
            f.write(json.dumps(record, allow_nan=False)+'\n')
            f.flush()
        previous=elapsed
    while True:
        if Path(stop).exists() or os.getppid()!=parent_pid:
            capture(final=True)
            return
        if time.time()-started >= due:
            capture()
            # Never backfill missed deadlines with a later frontier.
            due=(math.floor((time.time()-started)/interval)+1)*interval
        time.sleep(min(.25, interval/10))


class FrontierSnapshotRecorder:
    def __init__(self, directory, destination, interval):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('snapshot interval must be positive and finite')
        self.directory = str(Path(directory).resolve())
        self.destination = str(Path(destination).resolve())
        self.interval = interval
        self.stop = self.destination+'.stop.'+uuid.uuid4().hex

    def __enter__(self):
        Path(self.destination).parent.mkdir(parents=True, exist_ok=True)
        Path(self.destination).write_text('')
        self.process = subprocess.Popen([
            sys.executable, str(Path(__file__).resolve()), '--directory', self.directory,
            '--destination', self.destination, '--stop', self.stop,
            '--interval', str(self.interval), '--started', str(time.time()),
            '--parent-pid', str(os.getpid())])
        return self

    def __exit__(self, exc_type, exc, traceback):
        Path(self.stop).write_text('stop\n')
        try:
            code = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
            raise RuntimeError('Frontier snapshot observer did not stop')
        if code != 0 and exc_type is None:
            raise RuntimeError(f'Frontier snapshot observer failed ({code})')
        if exc_type is None:
            records = [json.loads(line) for line in Path(self.destination).read_text().splitlines()]
            if not records or not records[-1]['final'] or records[-1]['status'] != 'ok':
                raise RuntimeError('No valid final native frontier was captured')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','destination','stop']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--interval',type=float,required=True)
    parser.add_argument('--started',type=float,required=True)
    parser.add_argument('--parent-pid',type=int,required=True)
    args=parser.parse_args()
    watch(**vars(args))
