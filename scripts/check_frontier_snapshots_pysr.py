"""Short local integration check of uninterrupted native PySR frontier snapshots."""
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from julia_env import configure_juliapkg_project
configure_juliapkg_project(ROOT)
import numpy as np
from pysr import PySRRegressor
from run_pysr_srbench import run_pysr_with_hof_checkpoints
from parallel_eval_pysr import _load_execution_trace

out=Path(tempfile.mkdtemp(prefix='meta_sr_snapshot_smoke_'))
x=np.linspace(.1,2,40).reshape(-1,1)
y=x[:,0]**2+2*x[:,0]
model=PySRRegressor(niterations=1000000,populations=2,population_size=20,
                   binary_operators=['+','*'],unary_operators=[],maxsize=10,
                   timeout_in_seconds=4,parallelism='serial',procs=0,
                   deterministic=True,random_state=123,progress=False,verbosity=0,
                   output_directory=str(out/'native'))
run_pysr_with_hof_checkpoints(x,y,['x0'],'snapshot_smoke',str(out),[],model,
                             hof_path=str(out/'hof.csv'),frontier_snapshot_seconds=.5)
trace=_load_execution_trace([str(out/'hof.csv.snapshots.jsonl')])
assert trace and trace[-1]['final'] and trace[-1]['status']=='ok'
assert any(not r['final'] and r['status']=='ok' for r in trace)
assert len(trace[-1]['equations'])==len(model.equations_)
print(json.dumps({'directory':str(out),'snapshots':len(trace),
                  'available':sum(r['status']=='ok' for r in trace),
                  'final_rows':len(trace[-1]['equations']),
                  'final_fit_elapsed_seconds':trace[-1]['elapsed_seconds']}))
