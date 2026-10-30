"""Generate a rejection-ABC reference table: prior draws -> 160 summary statistics.
usage: python gen_table.py OUT_NPZ N SEED"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/sca63/meta_sr/scripts/pom_earthworm")
import ewmodel as ew

out, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(seed)
theta = ew.sample_prior(n, rng)
chunk = 5000
X = np.zeros((n, 160), np.float32)
t0 = time.time()
for i in range(0, n, chunk):
    X[i:i + chunk] = ew.run_all(theta[i:i + chunk], rng)
    print(f"{i + chunk}/{n} {time.time() - t0:.0f}s", flush=True)
np.savez_compressed(out, theta=theta, X=X)
