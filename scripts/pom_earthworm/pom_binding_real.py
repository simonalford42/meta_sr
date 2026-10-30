"""Which patterns bind the POM filter on the real data? (normalised deviation |sim-obs|/tol)."""
import sys; sys.path.insert(0, "/home/sca63/meta_sr/scripts/pom_earthworm")
import numpy as np, calib as C
th, X = C.load_table(); P = C.patterns(X)
emp = np.loadtxt(C.FIG_DATA); p = C.patterns(emp)[0]
tol = np.array([t*abs(v) if k=="rel" else (max(t*abs(v),3.0) if k=="absrel" else t)
                for (k,t),v in zip(C.PATTERNS.values(), p)])
R = np.abs(P - p)/tol
worst = R.max(1); o = np.argsort(worst)[:368]
print("best worst-ratio", worst[o[0]].round(2))
print("which pattern is the binding (argmax) one among accepted:")
names, cnt = np.unique(np.array(C.PAT_NAMES)[R[o].argmax(1)], return_counts=True)
print(dict(zip(names, cnt)))
print("min achievable ratio per pattern over table:", dict(zip(C.PAT_NAMES, R.min(0).round(2))))
# drop each pattern: how many accepted at base tolerance?
for i,k in enumerate(C.PAT_NAMES):
    m = np.delete(R, i, 1).max(1)
    print(k, "drop->n_acc@tol1:", int((m<=1).sum()), " @tol2:", int((m<=2).sum()))
print("per-experiment / pairwise feasibility (n accepted at tolerance 1 and 2):")
import itertools
exps = ["exp1","exp2","exp3","exp4"]
for r in [1,2,3]:
    for combo in itertools.combinations(exps, r):
        idx = [i for i,k in enumerate(C.PAT_NAMES) if C.PAT_EXP[k] in combo]
        m = R[:, idx].max(1)
        print("+".join(combo), int((m<=1).sum()), int((m<=2).sum()))
print("exp3: drop one pattern -> n accepted (tol 1 / 2), single pattern pass rate")
idx = [i for i,k in enumerate(C.PAT_NAMES) if C.PAT_EXP[k]=="exp3"]
for i in idx:
    rest = [j for j in idx if j!=i]
    m = R[:, rest].max(1)
    print(C.PAT_NAMES[i], int((m<=1).sum()), int((m<=2).sum()), (R[:,i]<=1).mean().round(3))
print("observed exp3 patterns:", {C.PAT_NAMES[i]: round(float(p[i]),3) for i in idx})
