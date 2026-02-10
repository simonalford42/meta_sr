#!/usr/bin/env python3
"""
Evolve Julia selection operators for PySR using LLMs.

This script evolves custom selection operators for SymbolicRegression.jl/PySR
by generating Julia code with an LLM, validating it, and evaluating
performance on SRBench datasets via SLURM.

A selection operator decides which population member is chosen as a parent
for mutation or crossover. The default is tournament selection with
adaptive parsimony.
"""

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from completions import chat_completion, get_content
from evolve_pysr import (
    extract_julia_code,
    extract_function_name,
    pre_validate_julia_syntax,
    select_parent,
    select_survivors,
    EvolutionLogger,
    TARGET_NOISE_LEVELS,
    _build_target_noise_map,
    _evaluate_configs_with_noise_map,
)
from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights,
    get_default_pysr_kwargs,
)
from utils import load_dataset_names_from_split, TeeLogger


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class JuliaSelection:
    """A Julia selection operator for PySR."""
    name: str
    code: str
    score: Optional[float] = None
    score_vector: Optional[List[float]] = None
    generation: int = 0
    parent_name: Optional[str] = None
    mode: str = "explore"

    def to_pysr_config(self, pysr_kwargs: Dict = None) -> PySRConfig:
        """Convert to PySRConfig for evaluation."""
        if pysr_kwargs is None:
            pysr_kwargs = get_default_pysr_kwargs()

        mutation_weights = get_default_mutation_weights()

        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            custom_selection_code=self.code,
            name=self.name,
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'JuliaSelection':
        return cls(**d)


# =============================================================================
# Julia Code Validation
# =============================================================================

def validate_julia_selection_code(name: str, code: str) -> Tuple[bool, str]:
    """
    Validate Julia selection code by attempting to load it.

    Returns:
        (is_valid, error_message)
    """
    is_valid, error = pre_validate_julia_syntax(code)
    if not is_valid:
        return False, error

    try:
        from juliacall import Main as jl

        jl.seval("using SymbolicRegression")
        jl.seval("using SymbolicRegression.CustomSelectionModule")

        jl.seval("clear_dynamic_selections!()")

        escaped_code = code.replace('"""', '\\"\\"\\"')
        jl.seval(f'load_selection_from_string!(:{name}, raw"""{escaped_code}""")')

        available = list(jl.seval("list_available_selections()"))
        if name not in [str(m) for m in available]:
            return False, f"Selection '{name}' not found in registry after loading"

        return True, ""

    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return False, error_msg


# =============================================================================
# LLM Code Generation
# =============================================================================

def load_selection_reference() -> str:
    """Load the SELECTION_REFERENCE.md file as context for LLM."""
    ref_path = Path(__file__).resolve().parent / "SymbolicRegression.jl/src/custom_selection/SELECTION_REFERENCE.md"
    if ref_path.exists():
        return ref_path.read_text()
    raise FileNotFoundError(f"Could not find SELECTION_REFERENCE.md at {ref_path}")


def _build_explore_prompt(selection_reference: str, variation_seed: int = 0) -> str:
    """Build prompt for exploring new selection ideas."""
    ideas = [
        "Lexicase selection: Sequentially filter candidates on shuffled evaluation criteria",
        "Epsilon-lexicase: Like lexicase but with tolerance threshold for near-best candidates",
        "Fitness-proportionate: Select with probability proportional to fitness (roulette wheel)",
        "Boltzmann/softmax: Use temperature-controlled selection pressure",
        "Rank-based: Assign selection probability based on rank rather than raw fitness",
        "Novelty-based: Prefer members whose expression structure is rare in the population",
        "Multi-objective: Consider both fitness and complexity using Pareto dominance",
        "Age-fitness Pareto: Combine age and fitness in multi-objective selection",
    ]
    selected_ideas = ideas[variation_seed % len(ideas):] + ideas[:variation_seed % len(ideas)]
    ideas_text = "\n".join(f"- {idea}" for idea in selected_ideas[:4])

    return f"""You are an expert in symbolic regression and genetic programming.

Your task is to create a NEW custom selection operator for PySR/SymbolicRegression.jl.
The selection operator decides which population member is chosen as a PARENT for mutation or crossover.

## Reference: Selection API and Default Implementation
{selection_reference}

## Requirements
1. Create a NOVEL selection strategy that differs from the default tournament selection
2. The function should help symbolic regression search find better expressions
3. Use proper Julia syntax and the available API
4. MUST return a PopMember (the dispatch will copy it)
5. Can use running_search_statistics for adaptive behavior

## Ideas to consider (pick one or invent your own):
{ideas_text}

## Output Format
Return ONLY the Julia function code, nothing else. The function should be named descriptively.
Do not include markdown code blocks or explanations.

Example format:
function my_selection_name(
    pop::Population{{T,L,N}},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{{T,L,N}} where {{T,L,N}}
    # Implementation
    return selected_member
end
"""


def _build_refine_prompt(parent_code: str, selection_reference: str, feedback: str = "") -> str:
    """Build prompt for refining an existing selection operator."""
    feedback_section = ""
    if feedback:
        feedback_section = f"\n## Feedback on parent selection:\n{feedback}\n"

    return f"""You are an expert in symbolic regression and genetic programming.

Your task is to IMPROVE an existing custom selection operator for PySR/SymbolicRegression.jl.

## Parent Selection Code
```julia
{parent_code}
```
{feedback_section}
## Reference: Selection API
{selection_reference}

## Requirements
1. Keep the core idea but improve the implementation
2. Consider: better edge case handling, smarter heuristics, combining strategies
3. MUST return a PopMember
4. Use proper Julia syntax

## Output Format
Return ONLY the improved Julia function code, nothing else.
Use a NEW function name (append _v2, _improved, etc. or rename descriptively).
Do not include markdown code blocks or explanations.
"""


def _build_crossover_prompt(parent1_code: str, parent2_code: str, selection_reference: str) -> str:
    """Build prompt for crossing over two selection operators."""
    return f"""You are an expert in symbolic regression and genetic programming.

Your task is to COMBINE ideas from two selection operators into a new one.

## Parent Selection 1
```julia
{parent1_code}
```

## Parent Selection 2
```julia
{parent2_code}
```

## Reference: Selection API
{selection_reference}

## Requirements
1. Create a NEW selection operator that combines the best ideas from both parents
2. Don't just concatenate - synthesize a coherent new approach
3. MUST return a PopMember
4. Use proper Julia syntax

## Output Format
Return ONLY the new Julia function code, nothing else.
Give it a new descriptive name.
Do not include markdown code blocks or explanations.
"""


def generate_selection_code(
    parent: Optional[JuliaSelection],
    selection_reference: str,
    model: str = "openai/gpt-5-mini",
    mode: str = "explore",
    parent2: Optional[JuliaSelection] = None,
    feedback: str = "",
    variation_seed: int = 0,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> Tuple[str, str]:
    """Generate new Julia selection code using an LLM."""
    if mode == "explore":
        prompt = _build_explore_prompt(selection_reference, variation_seed)
    elif mode == "refine":
        if parent is None:
            raise ValueError("refine mode requires a parent")
        prompt = _build_refine_prompt(parent.code, selection_reference, feedback)
    elif mode == "crossover":
        if parent is None or parent2 is None:
            raise ValueError("crossover mode requires two parents")
        prompt = _build_crossover_prompt(parent.code, parent2.code, selection_reference)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    response = chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        sample_index=variation_seed,
        use_cache=use_cache,
    )

    content = get_content(response)
    code = extract_julia_code(content)
    if not code:
        return "", ""

    func_name = extract_function_name(code)
    return code, func_name


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_selections(
    selections: List[JuliaSelection],
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    seed: int = 42,
    n_runs: int = 1,
    target_noise_map: Optional[Dict[str, float]] = None,
) -> List[Tuple[float, List[float], List[Dict]]]:
    """Evaluate multiple selection operators in parallel via SLURM."""
    if not selections:
        return []

    configs = [s.to_pysr_config(pysr_kwargs) for s in selections]

    return _evaluate_configs_with_noise_map(
        evaluator=evaluator,
        configs=configs,
        dataset_names=dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
    )


def evaluate_baseline(
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    seed: int = 42,
    n_runs: int = 1,
    target_noise_map: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[float], List[Dict]]:
    """Evaluate PySR with default selection (baseline)."""
    mutation_weights = get_default_mutation_weights()

    config = PySRConfig(
        mutation_weights=mutation_weights,
        pysr_kwargs=pysr_kwargs,
        name="baseline",
    )

    results = _evaluate_configs_with_noise_map(
        evaluator=evaluator,
        configs=[config],
        dataset_names=dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
    )
    avg_r2, r2_vector, result_details = results[0]
    return avg_r2, r2_vector, result_details


# =============================================================================
# Main Evolution Loop
# =============================================================================

def run_evolution(
    n_generations: int,
    population_size: int,
    n_offspring: int,
    dataset_names: List[str],
    model: str,
    temperature: float,
    seed: int,
    output_dir: str,
    pysr_kwargs: Dict,
    slurm_partition: str,
    slurm_time_limit: str,
    slurm_mem_per_cpu: str,
    max_samples: int,
    job_timeout: float,
    use_cache: bool = True,
    n_runs: int = 1,
    target_noise: float = 0.0,
    random_target_noise: bool = False,
) -> JuliaSelection:
    """Run the evolution loop for selection operators."""
    rng = random.Random(seed)
    np.random.seed(seed)

    logger = EvolutionLogger(output_dir)
    target_noise_map = None
    if random_target_noise:
        target_noise_map = _build_target_noise_map(dataset_names, seed, TARGET_NOISE_LEVELS)

    logger.set_config({
        "operator_type": "selection",
        "n_generations": n_generations,
        "population_size": population_size,
        "n_offspring": n_offspring,
        "n_datasets": len(dataset_names),
        "dataset_names": dataset_names,
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "pysr_kwargs": pysr_kwargs,
        "max_samples": max_samples,
        "n_runs": n_runs,
        "target_noise": target_noise,
        "random_target_noise": random_target_noise,
    })

    evaluator = PySRSlurmEvaluator(
        results_dir=output_dir,
        partition=slurm_partition,
        time_limit=slurm_time_limit,
        mem_per_cpu=slurm_mem_per_cpu,
        dataset_max_samples=max_samples,
        data_seed=seed,
        job_timeout=job_timeout,
        target_noise=target_noise,
    )

    selection_reference = load_selection_reference()

    # Evaluate baseline
    print("=" * 60)
    print("Evaluating baseline (default selection)...")
    print("=" * 60)
    baseline_r2, baseline_vector, baseline_details = evaluate_baseline(
        evaluator, dataset_names, pysr_kwargs, seed,
        n_runs=n_runs, target_noise_map=target_noise_map,
    )
    print(f"Baseline avg R²: {baseline_r2:.4f}")
    logger.log_baseline(baseline_r2, baseline_vector)

    # Generate initial population
    print("\n" + "=" * 60)
    print(f"Generating initial population ({population_size} selection operators)...")
    print("=" * 60)

    population: List[JuliaSelection] = []
    attempts = 0
    max_attempts = population_size * 3

    while len(population) < population_size and attempts < max_attempts:
        attempts += 1
        print(f"\nGenerating selection {len(population) + 1}/{population_size} (attempt {attempts})...")

        code, func_name = generate_selection_code(
            parent=None,
            selection_reference=selection_reference,
            model=model,
            mode="explore",
            variation_seed=attempts,
            temperature=temperature,
            use_cache=use_cache,
        )

        if not code or not func_name:
            print("  Failed to generate code")
            continue

        unique_name = f"{func_name}_init_{len(population)}"
        code = code.replace(f"function {func_name}(", f"function {unique_name}(", 1)

        is_valid, error = validate_julia_selection_code(unique_name, code)
        if not is_valid:
            print(f"  Validation failed: {error[:100]}...")
            continue

        selection = JuliaSelection(
            name=unique_name,
            code=code,
            generation=0,
            mode="explore",
        )
        population.append(selection)
        print(f"  Created: {unique_name}")

    if len(population) == 0:
        raise RuntimeError("Failed to generate any valid selection operators")

    print(f"\nGenerated {len(population)} valid selection operators")

    # Evaluate initial population
    print("\n" + "=" * 60)
    print(f"Evaluating initial population ({len(population)} operators in parallel)...")
    print("=" * 60)

    try:
        results = evaluate_selections(
            population, evaluator, dataset_names, pysr_kwargs, seed,
            n_runs=n_runs, target_noise_map=target_noise_map,
        )
        for selection, (avg_r2, r2_vector, _) in zip(population, results):
            selection.score = avg_r2
            selection.score_vector = r2_vector
            print(f"  {selection.name}: {avg_r2:.4f}")
    except Exception as e:
        print(f"  Batch evaluation failed: {e}")
        for selection in population:
            selection.score = -1.0
            selection.score_vector = []

    population.sort(key=lambda s: s.score if s.score else -1, reverse=True)
    best = population[0]
    print(f"\nBest initial selection: {best.name} (score: {best.score:.4f})")

    # Evolution loop
    for gen in range(1, n_generations + 1):
        print("\n" + "=" * 60)
        print(f"Generation {gen}/{n_generations}")
        print("=" * 60)

        offspring: List[JuliaSelection] = []
        offspring_attempts = 0
        max_offspring_attempts = n_offspring * 3

        while len(offspring) < n_offspring and offspring_attempts < max_offspring_attempts:
            offspring_attempts += 1

            mode = rng.choice(["explore", "refine", "refine", "refine", "crossover"])

            if mode == "explore":
                parent = None
                parent2 = None
            elif mode == "refine":
                parent = select_parent(population, rng)
                parent2 = None
            else:
                parent = select_parent(population, rng)
                parent2 = select_parent([s for s in population if s != parent], rng)

            code, func_name = generate_selection_code(
                parent=parent,
                selection_reference=selection_reference,
                model=model,
                mode=mode,
                parent2=parent2,
                variation_seed=gen * 100 + offspring_attempts,
                temperature=temperature,
                use_cache=use_cache,
            )

            if not code or not func_name:
                continue

            unique_name = f"{func_name}_gen{gen}_{len(offspring)}"
            code = code.replace(f"function {func_name}(", f"function {unique_name}(", 1)

            is_valid, error = validate_julia_selection_code(unique_name, code)
            if not is_valid:
                print(f"  Validation failed for {unique_name}: {error[:80]}...")
                continue

            selection = JuliaSelection(
                name=unique_name,
                code=code,
                generation=gen,
                parent_name=parent.name if parent else None,
                mode=mode,
            )
            offspring.append(selection)
            print(f"  Created: {unique_name} (mode={mode})")

        print(f"\nGenerated {len(offspring)} offspring")

        # Evaluate offspring
        print(f"\nEvaluating {len(offspring)} offspring in parallel...")
        try:
            results = evaluate_selections(
                offspring, evaluator, dataset_names, pysr_kwargs, seed,
                n_runs=n_runs, target_noise_map=target_noise_map,
            )
            for selection, (avg_r2, r2_vector, _) in zip(offspring, results):
                selection.score = avg_r2
                selection.score_vector = r2_vector
                print(f"  {selection.name}: {avg_r2:.4f}")
        except Exception as e:
            print(f"  Batch evaluation failed: {e}")
            for selection in offspring:
                selection.score = -1.0
                selection.score_vector = []

        population = select_survivors(population, offspring, population_size)
        best = population[0]

        print(f"\nGeneration {gen} complete:")
        print(f"  Best: {best.name} (score: {best.score:.4f})")
        print(f"  Baseline: {baseline_r2:.4f}")
        print(f"  Improvement: {best.score - baseline_r2:+.4f}")

        logger.log_generation(gen, population, offspring, best)

    logger.finalize(best)

    print("\n" + "=" * 60)
    print("Evolution complete!")
    print("=" * 60)
    print(f"Best selection: {best.name}")
    print(f"Best score: {best.score:.4f}")
    print(f"Baseline: {baseline_r2:.4f}")
    print(f"Improvement: {best.score - baseline_r2:+.4f}")

    return best


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evolve Julia selection operators for PySR using LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--offspring", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=3)

    parser.add_argument("--split", type=str, default="splits/train_hard.txt")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--target_noise", type=float, default=0.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random_target_noise", action="store_true")
    group.add_argument("--no_random_target_noise", dest="random_target_noise", action="store_false")
    parser.set_defaults(random_target_noise=True)

    parser.add_argument("--max_evals", type=int, default=100000)
    parser.add_argument("--timeout", type=int, default=3000)

    parser.add_argument("--model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--time_limit", type=str, default="04:00:00")
    parser.add_argument("--mem_per_cpu", type=str, default="8G")
    parser.add_argument("--job_timeout", type=float, default=3000.0)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")

    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/evolve_selection_{timestamp}"

    dataset_names = load_dataset_names_from_split(args.split)
    print(f"Loaded {len(dataset_names)} datasets from {args.split}")

    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals
    pysr_kwargs["timeout_in_seconds"] = args.timeout

    best = run_evolution(
        n_generations=args.generations,
        population_size=args.population,
        n_offspring=args.offspring,
        dataset_names=dataset_names,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        output_dir=args.output_dir,
        pysr_kwargs=pysr_kwargs,
        slurm_partition=args.partition,
        slurm_time_limit=args.time_limit,
        slurm_mem_per_cpu=args.mem_per_cpu,
        max_samples=args.max_samples,
        job_timeout=args.job_timeout,
        use_cache=not args.no_cache,
        n_runs=args.n_runs,
        target_noise=args.target_noise,
        random_target_noise=args.random_target_noise,
    )

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
