"""Boolean function synthesis domain for the meta-SR evolution pipeline.

This module defines the "task" abstraction for a Boolean-function synthesis
domain, analogous to the SRBench datasets used by ``evolve_pysr.py``. A task is
a set of input rows ``X`` (each entry in {0,1}) and target outputs ``y`` (in
{0,1}), i.e. a (possibly sampled) truth table.

Two families of tasks:

* **Synthetic** (``generate_synthetic_task`` / the ``SYNTHETIC_GENERATORS``
  registry) - parametric Boolean functions (parity, majority, comparator,
  multiplexer, threshold, random DNF, random expression trees). Full truth
  tables when ``n_inputs`` is small, else random minterm samples. These form the
  *train* distribution that evolution runs on.

* **IWLS 2020** (``load_iwls_task``) - the held-out real benchmark suite
  (`data/boolean/iwls2020`). 100 single-output Boolean functions, each with
  train/validation/test minterm samples in Espresso PLA format. This is the
  *test* set.

The natural fitness signal is **accuracy** = fraction of rows whose rounded
prediction matches ``y`` (for 0/1 targets this equals ``1 - MSE`` when
predictions are in {0,1}), with exact truth-table match as the "solved"
criterion (analogous to the gt-match gate on SRBench).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
IWLS_DIR = REPO_ROOT / "data" / "boolean" / "iwls2020" / "benchmarks"

# Filename suffix per split (the PLA files use "valid" not "validation").
_IWLS_SPLIT_SUFFIX = {"train": "train", "validation": "valid", "valid": "valid", "test": "test"}
_IWLS_SPLIT_DIR = {"train": "train", "validation": "validation", "valid": "validation", "test": "test"}


@dataclass
class BooleanTask:
    """A Boolean-function synthesis task: (X in {0,1}^{m x n}, y in {0,1}^m)."""

    name: str
    n_inputs: int
    X: np.ndarray  # shape (n_samples, n_inputs), float64 in {0.0, 1.0}
    y: np.ndarray  # shape (n_samples,), float64 in {0.0, 1.0}
    kind: str = "synthetic"  # "synthetic" | "iwls"
    target: Optional[str] = None  # human-readable ground truth, if known
    is_full_table: bool = False  # True if X enumerates the entire 2^n table
    meta: Dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def positive_fraction(self) -> float:
        return float(np.mean(self.y)) if self.y.size else 0.0

    def summary(self) -> str:
        return (
            f"{self.name}[{self.kind}] n_in={self.n_inputs} "
            f"n_samples={self.n_samples} pos_frac={self.positive_fraction:.3f}"
            + (f" target={self.target}" if self.target else "")
        )


# ---------------------------------------------------------------------------
# Truth-table helpers
# ---------------------------------------------------------------------------

def full_input_table(n: int) -> np.ndarray:
    """All 2^n input rows, shape (2^n, n), MSB-first (column 0 = bit n-1)."""
    if n > 22:
        raise ValueError(f"full_input_table for n={n} would be huge; sample instead")
    idx = np.arange(2 ** n, dtype=np.int64)
    # column j corresponds to bit (n-1-j) so that row index reads MSB..LSB
    bits = ((idx[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1).astype(np.float64)
    return bits


def sample_input_table(n: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Random distinct-ish minterm samples (with replacement for large n)."""
    if n <= 22 and n_samples >= 2 ** n:
        return full_input_table(n)
    bits = rng.integers(0, 2, size=(n_samples, n)).astype(np.float64)
    return bits


def _int_columns(X: np.ndarray) -> np.ndarray:
    """Interpret each row of X (MSB-first) as an integer."""
    n = X.shape[1]
    weights = (1 << np.arange(n - 1, -1, -1)).astype(np.int64)
    return (X.astype(np.int64) * weights[None, :]).sum(axis=1)


# ---------------------------------------------------------------------------
# Synthetic target functions (each maps X in {0,1}^{m x n} -> y in {0,1}^m)
# ---------------------------------------------------------------------------

def _parity(X: np.ndarray) -> np.ndarray:
    return (X.astype(np.int64).sum(axis=1) % 2).astype(np.float64)


def _majority(X: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    return (X.astype(np.int64).sum(axis=1) * 2 > n).astype(np.float64)


def _threshold(k: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(X: np.ndarray) -> np.ndarray:
        return (X.astype(np.int64).sum(axis=1) >= k).astype(np.float64)
    return f


def _and_all(X: np.ndarray) -> np.ndarray:
    return (X.astype(np.int64).sum(axis=1) == X.shape[1]).astype(np.float64)


def _or_all(X: np.ndarray) -> np.ndarray:
    return (X.astype(np.int64).sum(axis=1) > 0).astype(np.float64)


def _comparator(k: int, strict: bool = True) -> Callable[[np.ndarray], np.ndarray]:
    """a > b (or a >= b) where inputs are [a_{k-1..0}, b_{k-1..0}] (2k bits)."""

    def f(X: np.ndarray) -> np.ndarray:
        a = _int_columns(X[:, :k])
        b = _int_columns(X[:, k:])
        return (a > b if strict else a >= b).astype(np.float64)

    return f


def _multiplexer(s: int) -> Callable[[np.ndarray], np.ndarray]:
    """Classic k-mux: s address bits select one of 2^s data bits.

    Total inputs = s + 2^s. Address bits first, then data bits.
    """

    def f(X: np.ndarray) -> np.ndarray:
        addr = _int_columns(X[:, :s])
        data = X[:, s:]
        return data[np.arange(X.shape[0]), addr]

    return f


def _make_random_dnf(n: int, n_terms: int, term_size: int, seed: int) -> Callable[[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    terms = []  # each term: list of (var_index, negated)
    for _ in range(n_terms):
        vars_ = rng.choice(n, size=min(term_size, n), replace=False)
        negs = rng.integers(0, 2, size=len(vars_))
        terms.append(list(zip(vars_.tolist(), negs.tolist())))

    def f(X: np.ndarray) -> np.ndarray:
        Xi = X.astype(np.int64)
        out = np.zeros(X.shape[0], dtype=bool)
        for term in terms:
            lit = np.ones(X.shape[0], dtype=bool)
            for (vi, neg) in term:
                col = Xi[:, vi] == (0 if neg else 1)
                lit &= col
            out |= lit
        return out.astype(np.float64)

    return f


def _make_random_expr(n: int, depth: int, seed: int) -> Callable[[np.ndarray], np.ndarray]:
    """Build a random Boolean expression tree over AND/OR/XOR/NOT and evaluate."""
    rng = np.random.default_rng(seed)

    def build(d):
        if d == 0 or rng.random() < 0.3:
            vi = int(rng.integers(0, n))
            neg = bool(rng.integers(0, 2))
            return ("lit", vi, neg)
        op = rng.choice(["and", "or", "xor"])
        return (op, build(d - 1), build(d - 1))

    tree = build(depth)

    def ev(node, Xi):
        if node[0] == "lit":
            _, vi, neg = node
            col = Xi[:, vi].astype(bool)
            return ~col if neg else col
        op, a, b = node
        la, lb = ev(a, Xi), ev(b, Xi)
        if op == "and":
            return la & lb
        if op == "or":
            return la | lb
        return la ^ lb

    def f(X: np.ndarray) -> np.ndarray:
        return ev(tree, X.astype(np.int64)).astype(np.float64)

    return f


# Registry of parametric synthetic task builders. Each entry: name -> spec dict
# with n_inputs and a function producing y from X.
def build_synthetic_registry() -> Dict[str, dict]:
    reg: Dict[str, dict] = {}

    # Parity (canonical "hard for greedy search")
    for n in (3, 4, 5, 6, 8, 10):
        reg[f"parity{n}"] = dict(n_inputs=n, fn=_parity, target=f"XOR of {n} inputs")

    # Majority / threshold (symmetric)
    for n in (3, 5, 7, 9):
        reg[f"majority{n}"] = dict(n_inputs=n, fn=_majority, target=f"majority of {n}")
    for n, k in ((5, 2), (6, 3), (8, 3), (8, 5)):
        reg[f"thresh{n}_{k}"] = dict(n_inputs=n, fn=_threshold(k), target=f"sum>={k} of {n}")

    # AND / OR of all inputs
    for n in (4, 6, 8):
        reg[f"and{n}"] = dict(n_inputs=n, fn=_and_all, target=f"AND of {n}")
        reg[f"or{n}"] = dict(n_inputs=n, fn=_or_all, target=f"OR of {n}")

    # Comparators (a > b), 2k inputs
    for k in (2, 3, 4, 5):
        reg[f"cmp{k}"] = dict(n_inputs=2 * k, fn=_comparator(k, strict=True), target=f"{k}-bit a>b")

    # Multiplexers: s address bits + 2^s data bits
    for s in (1, 2, 3):
        n = s + 2 ** s
        reg[f"mux{n}"] = dict(n_inputs=n, fn=_multiplexer(s), target=f"{n}-mux ({s} addr)")

    # Random DNF formulas
    for i in range(6):
        n = int([5, 6, 7, 8, 6, 7][i])
        reg[f"dnf_{n}_{i}"] = dict(
            n_inputs=n, fn=_make_random_dnf(n, n_terms=3 + i % 3, term_size=2 + i % 2, seed=1000 + i),
            target=f"random DNF (n={n}, seed={1000+i})",
        )

    # Random expression trees
    for i in range(6):
        n = int([5, 6, 7, 8, 6, 7][i])
        reg[f"expr_{n}_{i}"] = dict(
            n_inputs=n, fn=_make_random_expr(n, depth=3, seed=2000 + i),
            target=f"random expr tree (n={n}, seed={2000+i})",
        )

    return reg


SYNTHETIC_REGISTRY = build_synthetic_registry()


def generate_synthetic_task(
    name: str,
    max_samples: int = 4096,
    seed: int = 0,
) -> BooleanTask:
    """Materialize a synthetic task by name from ``SYNTHETIC_REGISTRY``."""
    if name not in SYNTHETIC_REGISTRY:
        raise KeyError(f"unknown synthetic task {name!r}; have {sorted(SYNTHETIC_REGISTRY)[:5]}...")
    spec = SYNTHETIC_REGISTRY[name]
    n = spec["n_inputs"]
    rng = np.random.default_rng(seed)
    if n <= 22 and 2 ** n <= max_samples:
        X = full_input_table(n)
        is_full = True
    else:
        X = sample_input_table(n, max_samples, rng)
        is_full = False
    y = spec["fn"](X).astype(np.float64)
    return BooleanTask(
        name=name, n_inputs=n, X=X, y=y, kind="synthetic",
        target=spec.get("target"), is_full_table=is_full,
        meta={"seed": seed, "max_samples": max_samples},
    )


# ---------------------------------------------------------------------------
# IWLS 2020 PLA loader (held-out real benchmark)
# ---------------------------------------------------------------------------

def parse_pla(path: os.PathLike | str) -> tuple[np.ndarray, np.ndarray, int]:
    """Parse an Espresso PLA file into (X, y, n_inputs).

    Handles the single-output IWLS minterm-sample format:
        .i N / .o 1 / .p P / .type fr / then P rows of "<bits> <0|1>".
    Rows with don't-care ('-') input bits are expanded over the '-' positions
    (rare in these sampled files); '-' outputs are skipped.
    """
    n_inputs = None
    n_outputs = None
    rows_X: List[List[float]] = []
    rows_y: List[float] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("."):
                parts = line.split()
                key = parts[0]
                if key == ".i":
                    n_inputs = int(parts[1])
                elif key == ".o":
                    n_outputs = int(parts[1])
                elif key == ".e":
                    break
                # .p, .type, .ilb, .ob ignored
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            in_str, out_str = parts[0], parts[-1]
            out_char = out_str[0]
            if out_char not in "01":
                continue  # skip don't-care outputs
            expansions = [in_str]
            if "-" in in_str:
                # expand don't-cares (typically none in the sampled files)
                expansions = _expand_dashes(in_str)
            for e in expansions:
                rows_X.append([1.0 if c == "1" else 0.0 for c in e])
                rows_y.append(float(int(out_char)))
    if n_inputs is None:
        raise ValueError(f"{path}: no .i header found")
    if n_outputs not in (None, 1):
        raise ValueError(f"{path}: expected single-output (.o 1), got {n_outputs}")
    X = np.asarray(rows_X, dtype=np.float64)
    y = np.asarray(rows_y, dtype=np.float64)
    return X, y, n_inputs


def _expand_dashes(in_str: str) -> List[str]:
    dash_positions = [i for i, c in enumerate(in_str) if c == "-"]
    if not dash_positions:
        return [in_str]
    out = []
    for combo in range(2 ** len(dash_positions)):
        chars = list(in_str)
        for j, pos in enumerate(dash_positions):
            chars[pos] = "1" if (combo >> j) & 1 else "0"
        out.append("".join(chars))
    return out


def iwls_pla_path(ex_id: str, split: str) -> Path:
    """Path to an IWLS PLA file. ex_id like 'ex30'; split train|validation|test."""
    suffix = _IWLS_SPLIT_SUFFIX[split]
    sub = _IWLS_SPLIT_DIR[split]
    return IWLS_DIR / sub / f"{ex_id}.{suffix}.pla"


def load_iwls_task(ex_id: str, split: str = "test", max_samples: Optional[int] = None,
                   seed: int = 0) -> BooleanTask:
    """Load one IWLS 2020 function/split as a BooleanTask.

    ``ex_id``: 'ex00'..'ex99'. ``split``: 'train'|'validation'|'test'.
    ``max_samples``: optional random subsample of the minterms.
    """
    path = iwls_pla_path(ex_id, split)
    X, y, n_inputs = parse_pla(path)
    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        sel = rng.choice(X.shape[0], size=max_samples, replace=False)
        X, y = X[sel], y[sel]
    return BooleanTask(
        name=f"iwls:{ex_id}", n_inputs=n_inputs, X=X, y=y, kind="iwls",
        target=None, is_full_table=False,
        meta={"ex_id": ex_id, "split": split, "path": str(path)},
    )


def iwls_input_widths() -> Dict[str, int]:
    """Map ex_id -> #inputs by reading the .i header of each train PLA (cheap)."""
    widths: Dict[str, int] = {}
    train_dir = IWLS_DIR / "train"
    for p in sorted(train_dir.glob("*.train.pla")):
        ex = p.name.split(".")[0]
        with open(p) as fh:
            for line in fh:
                if line.startswith(".i"):
                    widths[ex] = int(line.split()[1])
                    break
    return widths


def tractable_iwls_ids(max_inputs: int = 24) -> List[str]:
    """IWLS function ids whose input width <= max_inputs (POC-tractable subset)."""
    widths = iwls_input_widths()
    return sorted([ex for ex, w in widths.items() if w <= max_inputs], key=lambda e: (widths[e], e))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of rows where rounded prediction matches the {0,1} target."""
    if y_true.size == 0:
        return 0.0
    pred = np.round(np.nan_to_num(y_pred, nan=0.5))
    return float(np.mean(pred == y_true))


def is_solved(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """Exact truth-table match on the provided rows (analog of gt-match)."""
    return accuracy(y_true, y_pred) == 1.0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect Boolean-domain tasks")
    ap.add_argument("--synthetic", action="store_true", help="list synthetic tasks")
    ap.add_argument("--iwls", action="store_true", help="list tractable IWLS tasks")
    ap.add_argument("--max-inputs", type=int, default=24)
    args = ap.parse_args()

    if args.synthetic:
        print(f"# {len(SYNTHETIC_REGISTRY)} synthetic tasks")
        for name in SYNTHETIC_REGISTRY:
            t = generate_synthetic_task(name)
            print("  ", t.summary())
    if args.iwls:
        ids = tractable_iwls_ids(args.max_inputs)
        widths = iwls_input_widths()
        print(f"# {len(ids)} IWLS tasks with <= {args.max_inputs} inputs")
        for ex in ids:
            print(f"   {ex}  n_in={widths[ex]}")
