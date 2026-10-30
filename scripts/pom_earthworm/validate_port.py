"""Validate the numpy port against the NetLogo reference table (figshare 1412720, first 1e5 rows).
Runs our simulator on the same parameter rows and compares the 160 outputs."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/sca63/meta_sr/scripts/pom_earthworm")
import ewmodel as ew

D = "/home/sca63/meta_sr/runs/pom-earthworm-figshare"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
pri = pd.read_csv(f"{D}/priors_full_1e5.txt", sep=" ", nrows=n).iloc[:, 1:].to_numpy()
ref = pd.read_csv(f"{D}/results_full_1e5.txt", sep=" ", nrows=n).to_numpy()
rng = np.random.default_rng(0)
t0 = time.time()
ours = ew.run_all(pri, rng)
print(f"simulated {n} param sets x 4 experiments in {time.time()-t0:.1f}s")
ours2 = ew.run_all(pri, np.random.default_rng(1))   # second replicate: our own stochastic noise

def report(name, cols):
    a, b, c = ref[:, cols], ours[:, cols], ours2[:, cols]
    sd = a.std(0) + 1e-12
    d_ref = np.abs(a - b).mean(0) / sd
    d_self = np.abs(c - b).mean(0) / sd
    r = [np.corrcoef(a[:, j], b[:, j])[0, 1] if a[:, j].std() > 0 else np.nan for j in range(len(cols))]
    r2 = [np.corrcoef(c[:, j], b[:, j])[0, 1] if c[:, j].std() > 0 else np.nan for j in range(len(cols))]
    print(f"{name:8s} |ref-ours|/sd={np.nanmean(d_ref):.3f}  |rep-ours|/sd={np.nanmean(d_self):.3f}  "
          f"corr(ref,ours)={np.nanmean(r):.3f} corr(rep,ours)={np.nanmean(r2):.3f}  "
          f"mean ref={a.mean():.3f} ours={b.mean():.3f}")

for k, cols in ew.COLS.items():
    report(k, cols)
emp = np.loadtxt(f"{D}/all_data.txt")
print("dist-to-data quantiles ref vs ours (sd-scaled sq):")
sd = ref.std(0); sd[sd == 0] = 1
for X, nm in [(ref, "ref"), (ours, "ours")]:
    dist = (((X - emp) / sd) ** 2).sum(1)
    print(nm, np.quantile(dist, [0.001, 0.01, 0.1, 0.5]).round(1))
