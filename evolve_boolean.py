"""Evolve a custom PySR *mutation* operator for the Boolean-synthesis domain.

This is a compact, local (non-SLURM) analog of ``evolve_pysr.py``: it uses an
LLM (via ``operator_types.generate_operator_code_batch``) to propose Julia
mutation-operator code, evaluates each candidate by running Boolean-PySR on a
set of synthetic training tasks (parallel across a persistent worker pool), and
keeps the best over a few generations.

The evolved object is exactly what ``evolve_pysr.py`` produces - a Julia
mutation function as a code string - so it transfers unchanged to the SRBench
pipeline; only the *task distribution* (Boolean truth tables) and *operator set*
(band/bor/bxor/bnot) differ.

Design notes:
* The main process does NO Julia work - only LLM calls + orchestration. All
  PySR fits run in ``spawn`` workers (each with its own Julia session). Malformed
  mutation code therefore just makes its fits error out (score 0), which is a
  cleaner signal than a separate in-process Julia validation and avoids standing
  up a 9th Julia runtime in the driver.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from boolean_eval import BooleanEvaluator, BooleanJob
from boolean_tasks import BooleanTask, generate_synthetic_task
from operator_types import (
    JuliaOperator,
    MutationOperatorType,
    OperatorGenerationSpec,
    generate_operator_code_batch,
)


@dataclass
class Candidate:
    name: str  # "baseline" or the Julia function name
    code: Optional[str]  # Julia mutation source, or None for baseline
    operator: Optional[JuliaOperator]
    mode: str = "baseline"
    parent_name: Optional[str] = None
    generation: int = 0
    score: float = float("-inf")  # mean train accuracy
    solve_rate: float = 0.0
    per_task: Dict[str, float] = field(default_factory=dict)
    error_rate: float = 0.0

    @property
    def mutation_code(self) -> Optional[Dict[str, str]]:
        if self.code is None:
            return None
        return {self.name: self.code}

    def short(self) -> str:
        tag = "baseline" if self.code is None else f"{self.name}({self.mode})"
        return f"{tag} score={self.score:.3f} solve={self.solve_rate:.2f}"


# Harder synthetic tasks (more inputs) so the baseline can't already ace them -
# this leaves headroom for evolved mutations to create real selection pressure.
DEFAULT_TRAIN_TASKS = [
    "parity6", "parity8", "majority7", "cmp4",
    "cmp5", "mux11", "dnf_8_3", "expr_8_3",
]


@dataclass
class EvolveConfig:
    train_tasks: List[str] = field(default_factory=lambda: list(DEFAULT_TRAIN_TASKS))
    population_size: int = 6
    n_generations: int = 3
    n_offspring: int = 4
    n_seeds: int = 1
    niterations: int = 40
    max_samples: int = 4096
    model: str = "openai/gpt-5.4-mini"
    reasoning_effort: str = "medium"
    mutation_weight: float = 5.0
    n_workers: int = 8
    seed: int = 0
    out_dir: str = "runs_local/boolean_evolve"


class BooleanEvolver:
    def __init__(self, cfg: EvolveConfig):
        self.cfg = cfg
        self.mut_type = MutationOperatorType()
        self.reference = self.mut_type.load_reference()
        self.rng = np.random.default_rng(cfg.seed)
        self.train_tasks: List[BooleanTask] = [
            generate_synthetic_task(name, max_samples=cfg.max_samples, seed=cfg.seed)
            for name in cfg.train_tasks
        ]
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log: List[dict] = []
        # Best candidate that actually carries a mutation operator (code != None),
        # tracked across all generations so the "evolved" condition always has a
        # genuine evolved operator to report even if plain baseline scores higher.
        self.best_with_code: Optional[Candidate] = None

    # --- LLM proposal ------------------------------------------------------
    def _propose(self, parent: Optional[Candidate], mode: str, variation_seed: int) -> Optional[Candidate]:
        spec = OperatorGenerationSpec(
            op_type=self.mut_type,
            reference=self.reference,
            parent=parent.operator if (parent and parent.operator) else None,
            mode=mode,
            model=self.model_for(mode),
            variation_seed=variation_seed,
            reasoning_effort=self.cfg.reasoning_effort,
            use_cache=True,
        )
        try:
            results = generate_operator_code_batch([spec])
        except Exception as exc:  # noqa: BLE001
            print(f"    [propose] generation failed: {exc!r}", flush=True)
            return None
        if not results:
            return None
        code, func_name, _model = results[0]
        if not code or not func_name:
            print("    [propose] empty/invalid code, skipping", flush=True)
            return None
        operator = self.mut_type.create_operator(
            func_name, code, generation=0,
            parent_name=(parent.name if parent else None), mode=mode,
        )
        return Candidate(
            name=func_name, code=code, operator=operator, mode=mode,
            parent_name=(parent.name if parent else None),
        )

    def model_for(self, mode: str) -> str:
        return self.cfg.model

    # --- evaluation --------------------------------------------------------
    def _jobs_for(self, cand: Candidate) -> List[BooleanJob]:
        jobs = []
        for task in self.train_tasks:
            for s in range(self.cfg.n_seeds):
                jobs.append(BooleanJob(
                    train_task=task,
                    eval_task=None,
                    custom_mutation_code=cand.mutation_code,
                    custom_mutation_weight=self.cfg.mutation_weight,
                    pysr_kwargs={"niterations": self.cfg.niterations},
                    seed=self.cfg.seed + s,
                    tag=cand.name,
                ))
        return jobs

    def _score_candidates(self, evaluator: BooleanEvaluator, cands: List[Candidate]) -> None:
        # Flatten all jobs across candidates so the pool stays saturated.
        job_index: List[tuple] = []  # (cand_idx, task_name)
        all_jobs: List[BooleanJob] = []
        for ci, cand in enumerate(cands):
            for job in self._jobs_for(cand):
                job_index.append((ci, job.train_task.name))
                all_jobs.append(job)
        done = evaluator.run(all_jobs)
        # Aggregate per candidate.
        acc: Dict[int, List[float]] = {ci: [] for ci in range(len(cands))}
        solved: Dict[int, List[bool]] = {ci: [] for ci in range(len(cands))}
        errors: Dict[int, List[bool]] = {ci: [] for ci in range(len(cands))}
        per_task: Dict[int, Dict[str, float]] = {ci: {} for ci in range(len(cands))}
        for (ci, task_name), job in zip(job_index, done):
            r = job.result or {}
            a = float(r.get("train_acc", 0.0))
            acc[ci].append(a)
            solved[ci].append(bool(r.get("train_solved", False)))
            errors[ci].append(bool(r.get("error")))
            per_task[ci][task_name] = max(per_task[ci].get(task_name, 0.0), a)
        for ci, cand in enumerate(cands):
            cand.score = float(np.mean(acc[ci])) if acc[ci] else 0.0
            cand.solve_rate = float(np.mean(solved[ci])) if solved[ci] else 0.0
            cand.error_rate = float(np.mean(errors[ci])) if errors[ci] else 0.0
            cand.per_task = per_task[ci]
            # Track the best mutation-bearing candidate across the whole run.
            if cand.code is not None:
                if (self.best_with_code is None
                        or (cand.score, cand.solve_rate)
                        > (self.best_with_code.score, self.best_with_code.solve_rate)):
                    self.best_with_code = cand

    # --- evolution loop ----------------------------------------------------
    def run(self) -> Candidate:
        cfg = self.cfg
        t0 = time.time()
        print(f"[evolve] {len(self.train_tasks)} train tasks, pop={cfg.population_size}, "
              f"gens={cfg.n_generations}, offspring={cfg.n_offspring}, workers={cfg.n_workers}",
              flush=True)

        with BooleanEvaluator(n_workers=cfg.n_workers) as evaluator:
            # Initial population: baseline + explore proposals.
            baseline = Candidate(name="baseline", code=None, operator=None, mode="baseline")
            population: List[Candidate] = [baseline]
            n_explore = cfg.population_size - 1
            print(f"[gen 0] proposing {n_explore} explore candidates...", flush=True)
            for i in range(n_explore):
                cand = self._propose(None, "explore", variation_seed=int(self.rng.integers(1, 10**6)))
                if cand is not None:
                    population.append(cand)
            self._score_candidates(evaluator, population)
            population.sort(key=lambda c: c.score, reverse=True)
            self._log_generation(0, population)

            for gen in range(1, cfg.n_generations + 1):
                print(f"[gen {gen}] proposing {cfg.n_offspring} offspring...", flush=True)
                offspring: List[Candidate] = []
                for _ in range(cfg.n_offspring):
                    parent = self._tournament(population)
                    mode = self._pick_mode(parent)
                    cand = self._propose(parent, mode, variation_seed=int(self.rng.integers(1, 10**6)))
                    if cand is not None:
                        offspring.append(cand)
                if offspring:
                    self._score_candidates(evaluator, offspring)
                # Elitist survival over population + offspring.
                combined = population + offspring
                combined.sort(key=lambda c: (c.score, c.solve_rate), reverse=True)
                population = combined[: cfg.population_size]
                self._log_generation(gen, population, offspring)

        best = max(population, key=lambda c: (c.score, c.solve_rate))
        dt = time.time() - t0
        print(f"[evolve] done in {dt:.0f}s. best: {best.short()}", flush=True)
        self._save_best(best)
        return best

    def _tournament(self, population: List[Candidate], k: int = 2) -> Candidate:
        idx = self.rng.choice(len(population), size=min(k, len(population)), replace=False)
        return max((population[i] for i in idx), key=lambda c: c.score)

    def _pick_mode(self, parent: Candidate) -> str:
        if parent.code is None:
            return "explore"
        return str(self.rng.choice(["refine", "explore", "simplify"]))

    # --- logging -----------------------------------------------------------
    def _log_generation(self, gen: int, population: List[Candidate],
                        offspring: Optional[List[Candidate]] = None) -> None:
        best = population[0]
        print(f"  [gen {gen}] best={best.short()}  pop_scores="
              + ", ".join(f"{c.score:.2f}" for c in population), flush=True)
        entry = {
            "generation": gen,
            "best_name": best.name,
            "best_score": best.score,
            "best_solve_rate": best.solve_rate,
            "population": [
                {"name": c.name, "mode": c.mode, "score": c.score,
                 "solve_rate": c.solve_rate, "error_rate": c.error_rate,
                 "per_task": c.per_task, "parent": c.parent_name}
                for c in population
            ],
        }
        self.log.append(entry)
        (self.out_dir / "evolve_log.json").write_text(json.dumps(self.log, indent=2))

    def _cand_payload(self, cand: Candidate) -> dict:
        return {
            "name": cand.name,
            "mode": cand.mode,
            "score": cand.score,
            "solve_rate": cand.solve_rate,
            "per_task": cand.per_task,
            "custom_mutation_code": cand.mutation_code,
            "config": self.cfg.__dict__,
        }

    def _save_best(self, best: Candidate) -> None:
        (self.out_dir / "best_operator.json").write_text(
            json.dumps(self._cand_payload(best), indent=2))
        if best.code:
            (self.out_dir / "best_operator.jl").write_text(best.code)
        # Always persist the best mutation-bearing operator too, so the evolved
        # condition can be evaluated even when plain baseline scored highest.
        if self.best_with_code is not None:
            (self.out_dir / "best_evolved_operator.json").write_text(
                json.dumps(self._cand_payload(self.best_with_code), indent=2))
            (self.out_dir / "best_evolved_operator.jl").write_text(self.best_with_code.code)
            print(f"[evolve] best-with-code: {self.best_with_code.short()}", flush=True)
        print(f"[evolve] saved best -> {self.out_dir}/best_operator.json", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evolve a Boolean-domain PySR mutation operator")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--population", type=int, default=6)
    ap.add_argument("--offspring", type=int, default=4)
    ap.add_argument("--niterations", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", type=str, default="openai/gpt-5.4-mini")
    ap.add_argument("--effort", type=str, default="medium")
    ap.add_argument("--out", type=str, default="runs_local/boolean_evolve")
    args = ap.parse_args()

    cfg = EvolveConfig(
        n_generations=args.generations, population_size=args.population,
        n_offspring=args.offspring, niterations=args.niterations,
        n_workers=args.workers, model=args.model, reasoning_effort=args.effort,
        out_dir=args.out,
    )
    BooleanEvolver(cfg).run()
