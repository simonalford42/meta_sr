#!/usr/bin/env python3
"""Materialize all nine EmpiricalBench problems as PMLB-style datasets.

EmpiricalBench (Cranmer 2023, "Interpretable Machine Learning for Science with
PySR and SymbolicRegression.jl", arXiv:2305.01582) is a benchmark of 9 historical
empirical equations. In the paper's Table 3, Planck and Rydberg scored 0/5 for
every method tested (PySR, Operon, DSR, EQL, QLattice, SR-Transformer).

    The full nine-task EmpiricalBench: Hubble, Kepler, Newton, Bode, Leavitt,
    Schechter, ideal gas, Planck, and Rydberg.

Planck and Rydberg are faithful ports of the publication generator. The other
seven reuse the publication data vendored under PMLB ``first_principles_*``
names and add machine-readable reduced recovery formulas. Fidelity points:

  * Deterministic seeding: ``seed = sha256(key).digest()`` viewed as uint32s, fed
    to ``np.random.RandomState`` -> identical sampled points and noise draws.
  * The seven existing PMLB ``first_principles_*`` datasets are copied into
    canonical ``empirical_*`` aliases with usable reduced ground-truth formulas.
  * Planck and Rydberg are reproduced directly from the publication generator.
  * Noise: relative Gaussian applied to the target, *before* any log scaling
    (planck 10%, rydberg 1%).
  * The vendored targets retain the publication's target transforms. Inputs are
    not scaled, so the algorithm must cope with their raw ranges.
  * Computation is done in mpmath: Planck's ``exp(h nu/(k_B T))`` argument reaches
    ~4800, which overflows float64; mpmath keeps B representable so that the final
    ``log(B)`` is a sane float.

Output (no eval-pipeline changes needed -- these load by name like any SRBench
dataset via ``utils.load_srbench_dataset``):

    pmlb/datasets/empirical_planck/{empirical_planck.tsv.gz, metadata.yaml}
    pmlb/datasets/empirical_rydberg/{empirical_rydberg.tsv.gz, metadata.yaml}

Also writes ``splits/empirical_bench.txt``.

Usage:
    python scripts/gen_empirical_bench.py
    python scripts/gen_empirical_bench.py --check   # regenerate + report stats only
"""

import argparse
from hashlib import sha256
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PMLB = REPO / "pmlb" / "datasets"
SPLITS = REPO / "splits"


# --- faithful ports of pysr_paper/src/scripts/benchmark.py -------------------

def rand_logarithmic(low, high, size, rstate):
    log_low = np.log(low)
    log_high = np.log(high)
    log_data = rstate.rand(size) * (log_high - log_low) + log_low
    return np.exp(log_data)


def _seed_rstate(key: str) -> np.random.RandomState:
    """seed = sha256(key).digest() reinterpreted as uint32s (matches the paper)."""
    seed = np.frombuffer(sha256(key.encode()).digest(), dtype=np.uint32)
    return np.random.RandomState(seed)


# --- problem definitions (verbatim from equations.yml) -----------------------
#
# Each entry mirrors a ``data_generator`` block.  ``gt_formula`` is the
# ground-truth string stored in metadata.yaml for the symbolic-recovery check;
# it is the *log-scaled* target (because scale_dataset=True) with the physical
# constants baked in as numbers (the SR search must rediscover those numbers).

H = mp.mpf("6.62607004e-34")     # Planck constant, J s
K_B = mp.mpf("1.38064852e-23")   # Boltzmann constant, J/K
C = mp.mpf("299792458")          # speed of light, m/s
R_H = mp.mpf("1.097e7")          # Rydberg constant, m^-1


# These datasets already contain the publication data, but their PMLB metadata
# has missing/non-machine-readable equations.  EmpiricalBench judges recovery
# up to fitted constants, so use the paper's reduced functional forms.
ALIASES = {
    "empirical_hubble": ("first_principles_hubble", "v = D"),
    "empirical_kepler": ("first_principles_kepler", "P = sqrt(a**3)"),
    "empirical_newton": (
        "first_principles_newton", "logF = log(m1*m2) - 2*log(r)"
    ),
    "empirical_bode": (
        "first_principles_bode",
        "loga = log(0.4 + 0.3*exp(0.6931471805599453*n))",
    ),
    "empirical_leavitt": ("first_principles_leavitt", "M = log(P)"),
    "empirical_schechter": (
        "first_principles_schechter", "logn = -1.2*log(L) - L/2.5e8"
    ),
    "empirical_ideal_gas": (
        "first_principles_ideal_gas", "logP = log(n*T/V)"
    ),
}


def gen_alias(name):
    source, formula = ALIASES[name]
    source_path = PMLB / source / f"{source}.tsv.gz"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source EmpiricalBench dataset: {source_path}")
    frame = pd.read_csv(source_path, sep="\t", compression="gzip")
    X = frame.drop(columns=["target"])
    y = frame["target"].to_numpy()
    if name == "empirical_bode":
        # Bode's plot has a logarithmic y-axis, so the publication benchmark's
        # scale_dataset=True protocol searches against log(semi-major axis).
        y = np.log(y)
        # The publication uses the conventional Bode indices -inf, 0, ..., 6.
        # The vendored PMLB copy labels the finite rows 1, ..., 7.
        finite = X.iloc[:, 0] != -1000
        X.loc[finite, X.columns[0]] -= 1
    elif name == "empirical_leavitt":
        # The source stores log10(period), while the paper explicitly converts
        # back to period so symbolic regression must rediscover the logarithm.
        X.iloc[:, 0] = 10.0 ** X.iloc[:, 0]
        X = X.rename(columns={X.columns[0]: "P"})
    desc = (
        f"EmpiricalBench: {name.removeprefix('empirical_').replace('_', ' ')}.\n"
        "Canonical alias of the publication dataset with a reduced recovery target.\n\n"
        f"    {formula}\n"
    )
    return X, y, desc, list(X.columns)


def gen_planck():
    key = "planck"
    rstate = _seed_rstate(key)
    size = 100

    # ranges sampled in args order: nu, then T (both log)
    nu = rand_logarithmic(1e9, 1e16, size, rstate)
    T = rand_logarithmic(100, 6000, size, rstate)

    def callable_(nu_i, T_i):
        return 2 * H * nu_i**3 / (C**2) * (1 / (mp.exp(H * nu_i / (K_B * T_i)) - 1))

    y = np.array([callable_(mp.mpf(nu[i]), mp.mpf(T[i])) for i in range(size)],
                 dtype=object)

    # relative noise on the target (scale: None -> 0.1), applied before log scaling
    scale = 0.1
    y = y + rstate.randn(size) * y * scale

    # scale_dataset: yscale == log -> target becomes log(y)
    y = np.array([float(mp.log(yi)) for yi in y])

    X = pd.DataFrame({"nu": nu, "T": T})
    gt_formula = (
        "B = log(2 * 6.62607004e-34 * nu**3 / 299792458**2 "
        "/ (exp(6.62607004e-34 * nu / (1.38064852e-23 * T)) - 1))"
    )
    desc = (
        "EmpiricalBench: Planck's law (Cranmer 2023, arXiv:2305.01582).\n"
        "One of two EmpiricalBench problems unsolved by all methods (0/5 in the paper).\n"
        "Target is log-scaled (scale_dataset is True), so the recovery target is log(B).\n"
        "Inputs are NOT scaled. 100 points, 10% relative Gaussian noise on B before log.\n\n"
        f"    {gt_formula}\n\n"
        "    nu in [1e9,1e16]   # frequency, Hz, log-sampled\n"
        "    T in [100,6000]    # temperature, K, log-sampled\n"
    )
    return X, y, desc, ["nu", "T"]


def gen_rydberg():
    key = "rydberg"
    rstate = _seed_rstate(key)
    size = 50

    # custom integer generation, in the paper's RNG order
    n1_list, n2_list = [], []
    for _ in range(size):
        n_1 = rstate.randint(1, 7)       # integer in [1, 6]
        n_2 = rstate.randint(n_1 + 1, 8)  # integer in [n_1+1, 7]
        n1_list.append(float(n_1))
        n2_list.append(float(n_2))
    n_1 = np.array(n1_list)
    n_2 = np.array(n2_list)

    def callable_(n1_i, n2_i):
        return 1 / (R_H * (1 / n1_i**2 - 1 / n2_i**2))

    y = np.array([callable_(mp.mpf(n_1[i]), mp.mpf(n_2[i])) for i in range(size)],
                 dtype=object)

    scale = 0.01  # 1% relative noise on lambda
    y = y + rstate.randn(size) * y * scale

    y = np.array([float(mp.log(yi)) for yi in y])  # yscale == log

    X = pd.DataFrame({"n_1": n_1, "n_2": n_2})
    gt_formula = "lambda = log(1 / (1.097e7 * (1/n_1**2 - 1/n_2**2)))"
    desc = (
        "EmpiricalBench: Rydberg formula (Cranmer 2023, arXiv:2305.01582).\n"
        "One of two EmpiricalBench problems unsolved by all methods (0/5 in the paper).\n"
        "Target is log-scaled (scale_dataset is True), so the recovery target is log(lambda).\n"
        "Inputs are NOT scaled. 50 points, 1% relative Gaussian noise on lambda before log.\n\n"
        f"    {gt_formula}\n\n"
        "    n_1 in [1,6]   # principal quantum number, integer\n"
        "    n_2 in [2,7]   # principal quantum number after transition, integer > n_1\n"
    )
    return X, y, desc, ["n_1", "n_2"]


# --- materialization ---------------------------------------------------------

def write_dataset(name: str, X: pd.DataFrame, y: np.ndarray, desc: str,
                  var_names) -> None:
    out_dir = PMLB / name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = X.copy()
    df["target"] = y
    df.to_csv(out_dir / f"{name}.tsv.gz", sep="\t", index=False, compression="gzip")

    # metadata.yaml mirrors the SRBench/PMLB layout. load_srbench_dataset extracts
    # the formula as the first description line containing '=' that is not a range,
    # so the description text above the formula must contain no '=' characters.
    features = "\n".join(
        f"  - name: {v}\n    type: continuous" for v in var_names
    )
    indented_desc = "\n".join("    " + ln if ln else "" for ln in desc.splitlines())
    metadata = (
        f"# Generated by scripts/gen_empirical_bench.py\n"
        f"dataset: {name}\n"
        f"description: |\n{indented_desc}\n"
        f"source: EmpiricalBench, github.com/MilesCranmer/pysr_paper\n"
        f"publication: Cranmer (2023), arXiv:2305.01582\n"
        f"task: regression\n"
        f"keywords:\n  - symbolic regression\n  - physics\n  - empirical_bench\n"
        f"target:\n  type: continuous\n"
        f"features:\n{features}\n"
    )
    (out_dir / "metadata.yaml").write_text(metadata)
    print(f"  wrote {out_dir}  ({len(df)} rows, vars={var_names})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="Regenerate and print target stats / a few rows.")
    args = parser.parse_args()

    generators = {
        **{name: (lambda name=name: gen_alias(name)) for name in ALIASES},
        "empirical_planck": gen_planck,
        "empirical_rydberg": gen_rydberg,
    }

    for name, gen in generators.items():
        X, y, desc, var_names = gen()
        print(f"\n{name}:")
        write_dataset(name, X, y, desc, var_names)
        if args.check:
            print(f"    target(log) min={y.min():.4f} max={y.max():.4f} "
                  f"mean={y.mean():.4f}  finite={np.isfinite(y).all()}")
            print(X.head(3).to_string(index=False))

    SPLITS.mkdir(exist_ok=True)
    (SPLITS / "empirical_bench.txt").write_text("\n".join(generators.keys()) + "\n")
    print(f"\nWrote split: {SPLITS / 'empirical_bench.txt'}")


if __name__ == "__main__":
    main()
