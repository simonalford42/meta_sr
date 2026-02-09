#!/usr/bin/env python3
"""
Evolve Julia mutation operators for PySR using LLMs.

This script evolves custom mutation operators for SymbolicRegression.jl/PySR
by generating Julia code with an LLM, validating it, and evaluating
performance on SRBench datasets via SLURM.
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
from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights,
    get_default_pysr_kwargs,
)
from utils import load_dataset_names_from_split, TeeLogger

TARGET_NOISE_LEVELS = [0.0, 0.001, 0.01, 0.1]


def _stable_target_noise(dataset_name: str, seed: int, noise_levels: List[float]) -> float:
    """Deterministically assign a target noise level based on dataset name + seed."""
    digest = hashlib.sha256(f"{seed}:{dataset_name}".encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "little") % len(noise_levels)
    return noise_levels[idx]


def _build_target_noise_map(
    dataset_names: List[str],
    seed: int,
    noise_levels: List[float],
) -> Dict[str, float]:
    """Map each dataset name to a deterministic target noise level."""
    return {name: _stable_target_noise(name, seed, noise_levels) for name in dataset_names}


def _evaluate_configs_with_noise_map(
    evaluator: PySRSlurmEvaluator,
    configs: List[PySRConfig],
    dataset_names: List[str],
    seed: int,
    n_runs: int,
    target_noise_map: Optional[Dict[str, float]] = None,
) -> List[Tuple[float, List[float], List[Dict]]]:
    """Evaluate configs with optional per-dataset target noise mapping.

    The noise map is passed directly to evaluate_configs, which sets
    per-task noise levels so all datasets are evaluated in a single batch.
    """
    return evaluator.evaluate_configs(
        configs, dataset_names, seed=seed, n_runs=n_runs, target_noise_map=target_noise_map
    )


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class JuliaMutation:
    """A Julia mutation operator for PySR."""
    name: str                           # e.g., "gradient_guided_gen3_1"
    code: str                           # Julia code string
    weight: float = 0.5                 # Mutation probability weight
    score: Optional[float] = None       # Average R² across datasets
    score_vector: Optional[List[float]] = None  # Per-dataset R² scores
    generation: int = 0                 # Which generation this was created
    parent_name: Optional[str] = None   # Parent mutation (if refined/crossed)
    mode: str = "explore"               # How it was created: explore/refine/crossover

    def to_pysr_config(self, pysr_kwargs: Dict = None) -> PySRConfig:
        """Convert to PySRConfig for evaluation."""
        if pysr_kwargs is None:
            pysr_kwargs = get_default_pysr_kwargs()

        mutation_weights = get_default_mutation_weights()
        mutation_weights["weight_custom_mutation_1"] = self.weight

        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            custom_mutation_code={self.name: self.code},
            allow_custom_mutations=True,
            name=self.name,
        )

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'JuliaMutation':
        """Create from dict."""
        return cls(**d)


# =============================================================================
# Julia Code Validation
# =============================================================================

def pre_validate_julia_syntax(code: str) -> Tuple[bool, str]:
    """
    Pre-validate Julia code for common LLM-generated syntax errors.

    These checks run before the Julia parser to give better error messages
    and avoid polluting the run logs with syntax errors.

    Returns:
        (is_valid, error_message) - error_message is empty if valid
    """
    import re

    # Check for repeated field names in named tuples
    # Pattern: (name=..., name=...) where name appears twice
    named_tuple_pattern = r'\(\s*(\w+)\s*=\s*[^,)]+\s*,\s*\1\s*='
    if re.search(named_tuple_pattern, code):
        return False, "Repeated field name in named tuple (e.g., (left=x, left=y) should be (left=x, right=y))"

    # Check for invalid try-catch syntax: "catch <value>" instead of "catch; <value>" or "catch e; <value>"
    # Pattern: catch followed by a number or expression that's not a variable name followed by newline/end
    invalid_catch_pattern = r'\bcatch\s+(\d+[\d.eE+-]*|[^;\s\w])'
    if re.search(invalid_catch_pattern, code):
        return False, "Invalid try-catch syntax: use 'catch; ...' or 'catch e; ...' not 'catch <value>'"

    # Check for const inside function (not at module level)
    # This is tricky - we look for 'const' that appears after 'function' without an 'end' in between
    # Simple heuristic: if 'const' appears indented (has leading whitespace)
    const_in_func_pattern = r'^[ \t]+const\s+'
    if re.search(const_in_func_pattern, code, re.MULTILINE):
        return False, "Cannot use 'const' inside function body (Julia syntax error)"

    return True, ""


def validate_julia_code(name: str, code: str) -> Tuple[bool, str]:
    """
    Validate Julia mutation code by attempting to load it.

    Args:
        name: Name for the mutation function
        code: Julia code string

    Returns:
        (is_valid, error_message) - error_message is empty if valid
    """
    # First, run pre-validation checks for common LLM errors
    is_valid, error = pre_validate_julia_syntax(code)
    if not is_valid:
        return False, error

    try:
        from juliacall import Main as jl

        # Import the custom mutations module
        jl.seval("using SymbolicRegression")
        jl.seval("using SymbolicRegression.CustomMutationsModule")

        # Clear any previous dynamic mutations
        jl.seval("clear_dynamic_mutations!()")

        # Try to load the mutation
        escaped_code = code.replace('"""', '\\"\\"\\"')
        jl.seval(f'load_mutation_from_string!(:{name}, raw"""{escaped_code}""")')

        # Check it's in the registry
        available = list(jl.seval("list_available_mutations()"))
        if name not in [str(m) for m in available]:
            return False, f"Mutation '{name}' not found in registry after loading"

        return True, ""

    except Exception as e:
        error_msg = str(e)
        # Truncate very long error messages
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return False, error_msg


# =============================================================================
# LLM Code Generation
# =============================================================================

def load_mutations_reference() -> str:
    """Load the MUTATIONS_REFERENCE2.md file as context for LLM."""
    base = Path(__file__).resolve().parent / "SymbolicRegression.jl/src/custom_mutations"
    ref_path = base / "MUTATIONS_REFERENCE2.md"
    if ref_path.exists():
        return ref_path.read_text()
    else:
        # Fall back to original if v2 doesn't exist
        ref_path = base / "MUTATIONS_REFERENCE.md"
        if ref_path.exists():
            return ref_path.read_text()
        raise FileNotFoundError(f"Could not find MUTATIONS_REFERENCE.md or MUTATIONS_REFERENCE2.md")

def _build_explore_prompt(mutation_reference: str, variation_seed: int = 0) -> str:
    """Build prompt for exploring new mutation ideas."""
    # Add variation to avoid cache hits
    ideas = [
        "Pattern-based: Insert common mathematical patterns (e.g., polynomial terms, trig identities)",
        "Structure-aware: Target specific tree structures for modification",
        "Simplification-focused: Identify and simplify redundant patterns",
        "Feature-focused: Encourage using underutilized input variables",
        "Constant-aware: Smart constant insertion or modification",
        "Depth-balancing: Rebalance tree depth for better search",
        "Symmetry-aware: Detect and exploit symmetric patterns",
        "Gradient-guided: Use loss gradient information to guide changes",
    ]
    # Rotate ideas based on seed
    selected_ideas = ideas[variation_seed % len(ideas):] + ideas[:variation_seed % len(ideas)]
    ideas_text = "\n".join(f"- {idea}" for idea in selected_ideas[:4])

    return f"""You are an expert in symbolic regression and genetic programming.

Your task is to create a NEW custom mutation operator for PySR/SymbolicRegression.jl.
The mutation should help discover better symbolic expressions.

## Reference: Existing Mutations and API
{mutation_reference}

## Requirements
1. Create a NOVEL mutation that does something different from existing mutations
2. The mutation should be useful for symbolic regression search
3. Use proper Julia syntax and the available API

## Ideas to consider (pick one or invent your own):
{ideas_text}

## Output Format
Return ONLY the Julia function code, nothing else. The function should be named descriptively.
Do not include markdown code blocks or explanations.

Example format:
function my_mutation_name(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {{T,N<:AbstractExpressionNode{{T}}}}
    # Implementation
    return tree
end
"""


def _build_refine_prompt(parent_code: str, mutation_reference: str, feedback: str = "") -> str:
    """Build prompt for refining an existing mutation."""
    feedback_section = ""
    if feedback:
        feedback_section = f"\n## Feedback on parent mutation:\n{feedback}\n"

    return f"""You are an expert in symbolic regression and genetic programming.

Your task is to IMPROVE an existing custom mutation operator for PySR/SymbolicRegression.jl.

## Parent Mutation Code
```julia
{parent_code}
```
{feedback_section}
## Reference: Mutations API
{mutation_reference}

## Requirements
1. Keep the core idea but improve the implementation
2. Consider: better edge case handling, more efficient sampling, smarter heuristics
3. The mutation should still be useful for symbolic regression search
4. Use proper Julia syntax

## Output Format
Return ONLY the improved Julia function code, nothing else.
Use a NEW function name (append _v2, _improved, etc. or rename descriptively).
Do not include markdown code blocks or explanations.
"""


def _build_crossover_prompt(parent1_code: str, parent2_code: str, mutation_reference: str) -> str:
    """Build prompt for crossing over two mutations."""
    return f"""You are an expert in symbolic regression and genetic programming.

Your task is to COMBINE ideas from two mutation operators into a new one.

## Parent Mutation 1
```julia
{parent1_code}
```

## Parent Mutation 2
```julia
{parent2_code}
```

## Reference: Mutations API
{mutation_reference}

## Requirements
1. Create a NEW mutation that combines the best ideas from both parents
2. Don't just concatenate - synthesize a coherent new approach
3. The mutation should be useful for symbolic regression search
4. Use proper Julia syntax

## Output Format
Return ONLY the new Julia function code, nothing else.
Give it a new descriptive name.
Do not include markdown code blocks or explanations.
"""


def extract_julia_code(response: str) -> str:
    """Extract Julia function code from LLM response."""
    text = response.strip()

    # Remove markdown code blocks if present
    if "```julia" in text:
        start = text.find("```julia") + len("```julia")
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()

    # Ensure we have a function definition
    if "function " not in text:
        return ""

    return text


def extract_function_name(code: str) -> str:
    """Extract function name from Julia code."""
    import re
    match = re.search(r'function\s+(\w+)\s*\(', code)
    if match:
        return match.group(1)
    return ""


def generate_mutation_code(
    parent: Optional[JuliaMutation],
    mutation_reference: str,
    model: str = "openai/gpt-5-mini",
    mode: str = "explore",
    parent2: Optional[JuliaMutation] = None,
    feedback: str = "",
    variation_seed: int = 0,
    temperature: float = 0.0,  # Default to 0 for reproducibility
    use_cache: bool = True,
) -> Tuple[str, str]:
    """
    Generate new Julia mutation code using an LLM.

    Args:
        parent: Parent mutation to refine (None for explore mode)
        mutation_reference: MUTATIONS_REFERENCE.md content
        model: LLM model to use
        mode: "explore" (new), "refine" (improve parent), or "crossover"
        parent2: Second parent for crossover mode
        feedback: Optional feedback about parent's performance
        use_cache: Whether to use LLM response caching (default True)

    Returns:
        (code, function_name) tuple
    """
    if mode == "explore":
        prompt = _build_explore_prompt(mutation_reference, variation_seed)
    elif mode == "refine":
        if parent is None:
            raise ValueError("refine mode requires a parent mutation")
        prompt = _build_refine_prompt(parent.code, mutation_reference, feedback)
    elif mode == "crossover":
        if parent is None or parent2 is None:
            raise ValueError("crossover mode requires two parent mutations")
        prompt = _build_crossover_prompt(parent.code, parent2.code, mutation_reference)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Call LLM
    # Use variation_seed as sample_index to vary cache key while maintaining determinism
    # No max_tokens limit - let reasoning models use as many tokens as needed
    response = chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        sample_index=variation_seed,
        use_cache=use_cache,
    )

    # Extract code from response
    content = get_content(response)
    code = extract_julia_code(content)
    if not code:
        return "", ""

    func_name = extract_function_name(code)
    return code, func_name


# =============================================================================
# Evolution Functions
# =============================================================================

def evaluate_baseline(
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    seed: int = 42,
    n_runs: int = 1,
    target_noise_map: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[float], List[Dict]]:
    """Evaluate PySR without any custom mutations (baseline)."""
    mutation_weights = get_default_mutation_weights()
    # Ensure custom mutations are disabled
    for i in range(1, 6):
        mutation_weights[f"weight_custom_mutation_{i}"] = 0.0

    config = PySRConfig(
        mutation_weights=mutation_weights,
        pysr_kwargs=pysr_kwargs,
        custom_mutation_code=None,
        allow_custom_mutations=False,
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


def evaluate_mutation(
    mutation: JuliaMutation,
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    seed: int = 42,
    n_runs: int = 1,
    target_noise_map: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[float]]:
    """Evaluate a single mutation via SLURM."""
    config = mutation.to_pysr_config(pysr_kwargs)
    results = _evaluate_configs_with_noise_map(
        evaluator=evaluator,
        configs=[config],
        dataset_names=dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
    )
    avg_r2, r2_vector, _ = results[0]
    return avg_r2, r2_vector


def evaluate_mutations(
    mutations: List[JuliaMutation],
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    seed: int = 42,
    n_runs: int = 1,
    target_noise_map: Optional[Dict[str, float]] = None,
) -> List[Tuple[float, List[float], List[Dict]]]:
    """
    Evaluate multiple mutations in parallel via a single SLURM job array.

    Args:
        mutations: List of JuliaMutation objects to evaluate
        evaluator: PySRSlurmEvaluator instance
        dataset_names: List of dataset names to evaluate on
        pysr_kwargs: PySR configuration
        seed: Random seed
        n_runs: Number of runs per mutation per dataset (scores are averaged)

    Returns:
        List of (avg_r2, r2_vector, result_details) tuples, one per mutation.
        result_details is a list of dicts with keys: dataset, avg_r2, run_r2_scores, etc.
    """
    if not mutations:
        return []

    # Convert all mutations to configs
    configs = [m.to_pysr_config(pysr_kwargs) for m in mutations]

    # Evaluate all configs in a single SLURM batch
    results = _evaluate_configs_with_noise_map(
        evaluator=evaluator,
        configs=configs,
        dataset_names=dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
    )

    return results


def select_parent(population: List[JuliaMutation], rng: random.Random) -> JuliaMutation:
    """Select a parent using tournament selection."""
    # Tournament size 2
    candidates = rng.sample(population, min(2, len(population)))
    return max(candidates, key=lambda m: m.score if m.score is not None else -1)


def select_survivors(
    population: List[JuliaMutation],
    offspring: List[JuliaMutation],
    population_size: int,
) -> List[JuliaMutation]:
    """Select best individuals from population + offspring."""
    combined = population + offspring
    # Filter out those without scores
    scored = [m for m in combined if m.score is not None]
    # Sort by score descending
    scored.sort(key=lambda m: m.score, reverse=True)
    return scored[:population_size]


# =============================================================================
# Logging
# =============================================================================

class EvolutionLogger:
    """Tracks and saves evolution run data."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up tee logging
        self.log_file = self.output_dir / "run.log"
        self.tee = TeeLogger(str(self.log_file))
        sys.stdout = self.tee

        # Initialize run data
        self.run_data = {
            "start_time": datetime.now().isoformat(),
            "config": {},
            "baseline": {},
            "generations": [],
        }

    def set_config(self, config: Dict):
        """Save run configuration."""
        self.run_data["config"] = config
        self._save()

    def log_baseline(self, avg_r2: float, r2_vector: List[float]):
        """Log baseline results."""
        self.run_data["baseline"] = {
            "avg_r2": avg_r2,
            "r2_vector": r2_vector,
        }
        self._save()

    def log_generation(
        self,
        generation: int,
        population: List[JuliaMutation],
        offspring: List[JuliaMutation],
        best: JuliaMutation,
    ):
        """Log data for a generation."""
        gen_data = {
            "generation": generation,
            "population": [m.to_dict() for m in population],
            "offspring": [m.to_dict() for m in offspring],
            "best_name": best.name,
            "best_score": best.score,
        }
        self.run_data["generations"].append(gen_data)
        self._save()

        # Also save best mutation code separately
        best_file = self.output_dir / f"best_mutation_gen{generation}.jl"
        best_file.write_text(f"# Best mutation from generation {generation}\n"
                             f"# Score: {best.score}\n\n{best.code}")

    def _save(self):
        """Save run data to JSON."""
        with open(self.output_dir / "run_data.json", "w") as f:
            json.dump(self.run_data, f, indent=2)

    def finalize(self, best: JuliaMutation):
        """Save final results."""
        self.run_data["end_time"] = datetime.now().isoformat()
        self.run_data["best_mutation"] = best.to_dict()
        self._save()

        # Save best mutation as standalone file
        final_file = self.output_dir / "best_mutation_final.jl"
        final_file.write_text(f"# Best mutation from evolution run\n"
                              f"# Score: {best.score}\n"
                              f"# Generation: {best.generation}\n\n{best.code}")
        print(f"\nFinal best mutation saved to: {final_file}")


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
) -> JuliaMutation:
    """
    Run the evolution loop.

    Args:
        n_generations: Number of generations to evolve
        population_size: Number of mutations to maintain
        n_offspring: Number of offspring per generation
        dataset_names: List of dataset names to evaluate on
        model: LLM model for code generation
        temperature: LLM temperature (0=deterministic)
        seed: Random seed
        output_dir: Directory for outputs
        pysr_kwargs: PySR configuration
        slurm_*: SLURM job configuration
        max_samples: Max samples per dataset
        job_timeout: SLURM job timeout in seconds
        use_cache: Whether to use LLM response caching (default True)
        n_runs: Number of evaluation runs per mutation per dataset (default 1)
        target_noise: Gaussian noise level for target (default 0.0)
        random_target_noise: If True, use per-dataset noise drawn from TARGET_NOISE_LEVELS

    Returns:
        Best mutation found
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Set up logging
    logger = EvolutionLogger(output_dir)
    target_noise_map = None
    if random_target_noise:
        target_noise_map = _build_target_noise_map(dataset_names, seed, TARGET_NOISE_LEVELS)

    logger.set_config({
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
        "target_noise_levels": TARGET_NOISE_LEVELS if random_target_noise else None,
        "target_noise_map": target_noise_map,
    })

    # Set up evaluator
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

    # Load mutation reference
    mutation_reference = load_mutations_reference()

    # Evaluate baseline
    print("=" * 60)
    print("Evaluating baseline (no custom mutations)...")
    print("=" * 60)
    baseline_r2, baseline_vector, baseline_details = evaluate_baseline(
        evaluator,
        dataset_names,
        pysr_kwargs,
        seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
    )
    # Show per-run averages across all datasets when n_runs > 1
    if n_runs > 1 and baseline_details:
        per_run_avgs = []
        for run_idx in range(n_runs):
            run_scores = [d["run_r2_scores"][run_idx] for d in baseline_details
                          if len(d.get("run_r2_scores", [])) > run_idx]
            if run_scores:
                per_run_avgs.append(np.mean(run_scores))
        runs_str = ", ".join(f"{s:.2f}" for s in per_run_avgs)
        print(f"Baseline avg R²: {baseline_r2:.4f} [{runs_str}]")
    else:
        print(f"Baseline avg R²: {baseline_r2:.4f}")
    logger.log_baseline(baseline_r2, baseline_vector)

    # Generate initial population
    print("\n" + "=" * 60)
    print(f"Generating initial population ({population_size} mutations)...")
    print("=" * 60)

    population: List[JuliaMutation] = []
    attempts = 0
    max_attempts = population_size * 3  # Allow some failures

    while len(population) < population_size and attempts < max_attempts:
        attempts += 1
        print(f"\nGenerating mutation {len(population) + 1}/{population_size} (attempt {attempts})...")

        code, func_name = generate_mutation_code(
            parent=None,
            mutation_reference=mutation_reference,
            model=model,
            mode="explore",
            variation_seed=attempts,
            temperature=temperature,
            use_cache=use_cache,
        )

        if not code or not func_name:
            print("  Failed to generate code")
            continue

        # Make name unique for initial population
        unique_name = f"{func_name}_init_{len(population)}"
        code = code.replace(f"function {func_name}(", f"function {unique_name}(", 1)

        # Validate
        is_valid, error = validate_julia_code(unique_name, code)
        if not is_valid:
            print(f"  Validation failed: {error[:100]}...")
            continue

        mutation = JuliaMutation(
            name=unique_name,
            code=code,
            weight=0.5,
            generation=0,
            mode="explore",
        )
        population.append(mutation)
        print(f"  Created: {unique_name}")

    if len(population) == 0:
        raise RuntimeError("Failed to generate any valid mutations")

    print(f"\nGenerated {len(population)} valid mutations")

    # Evaluate initial population (all mutations in parallel via single SLURM job)
    print("\n" + "=" * 60)
    print(f"Evaluating initial population ({len(population)} mutations in parallel)...")
    print("=" * 60)

    try:
        results = evaluate_mutations(
            population,
            evaluator,
            dataset_names,
            pysr_kwargs,
            seed,
            n_runs=n_runs,
            target_noise_map=target_noise_map,
        )
        for mutation, (avg_r2, r2_vector, result_details) in zip(population, results):
            mutation.score = avg_r2
            mutation.score_vector = r2_vector
            # Show per-run averages across all datasets when n_runs > 1
            if n_runs > 1 and result_details:
                # Compute average across datasets for each run
                per_run_avgs = []
                for run_idx in range(n_runs):
                    run_scores = [d["run_r2_scores"][run_idx] for d in result_details
                                  if len(d.get("run_r2_scores", [])) > run_idx]
                    if run_scores:
                        per_run_avgs.append(np.mean(run_scores))
                runs_str = ", ".join(f"{s:.2f}" for s in per_run_avgs)
                print(f"  {mutation.name}: Avg {avg_r2:.4f} [{runs_str}]")
            else:
                print(f"  {mutation.name}: {avg_r2:.4f}")
    except Exception as e:
        print(f"  Batch evaluation failed: {e}")
        for mutation in population:
            mutation.score = -1.0
            mutation.score_vector = []

    # Sort population by score
    population.sort(key=lambda m: m.score if m.score else -1, reverse=True)
    best = population[0]
    print(f"\nBest initial mutation: {best.name} (score: {best.score:.4f})")

    # Evolution loop
    for gen in range(1, n_generations + 1):
        print("\n" + "=" * 60)
        print(f"Generation {gen}/{n_generations}")
        print("=" * 60)

        offspring: List[JuliaMutation] = []
        offspring_attempts = 0
        max_offspring_attempts = n_offspring * 3

        while len(offspring) < n_offspring and offspring_attempts < max_offspring_attempts:
            offspring_attempts += 1

            # Choose mode (bias toward refine, occasionally explore or crossover)
            mode = rng.choice(["explore", "refine", "refine", "refine", "crossover"])

            if mode == "explore":
                parent = None
                parent2 = None
            elif mode == "refine":
                parent = select_parent(population, rng)
                parent2 = None
            else:  # crossover
                parent = select_parent(population, rng)
                parent2 = select_parent([m for m in population if m != parent], rng)

            # Generate code
            code, func_name = generate_mutation_code(
                parent=parent,
                mutation_reference=mutation_reference,
                model=model,
                mode=mode,
                parent2=parent2,
                variation_seed=gen * 100 + offspring_attempts,
                temperature=temperature,
                use_cache=use_cache,
            )

            if not code or not func_name:
                continue

            # Make name unique
            unique_name = f"{func_name}_gen{gen}_{len(offspring)}"

            # Update function name in code
            code = code.replace(f"function {func_name}(", f"function {unique_name}(", 1)

            # Validate
            is_valid, error = validate_julia_code(unique_name, code)
            if not is_valid:
                print(f"  Validation failed for {unique_name}: {error[:80]}...")
                continue

            mutation = JuliaMutation(
                name=unique_name,
                code=code,
                weight=0.5,
                generation=gen,
                parent_name=parent.name if parent else None,
                mode=mode,
            )
            offspring.append(mutation)
            print(f"  Created: {unique_name} (mode={mode})")

        print(f"\nGenerated {len(offspring)} offspring")

        # Evaluate offspring (all mutations in parallel via single SLURM job)
        print(f"\nEvaluating {len(offspring)} offspring in parallel...")
        try:
            results = evaluate_mutations(
                offspring,
                evaluator,
                dataset_names,
                pysr_kwargs,
                seed,
                n_runs=n_runs,
                target_noise_map=target_noise_map,
            )
            for mutation, (avg_r2, r2_vector, result_details) in zip(offspring, results):
                mutation.score = avg_r2
                mutation.score_vector = r2_vector
                # Show per-run averages across all datasets when n_runs > 1
                if n_runs > 1 and result_details:
                    # Compute average across datasets for each run
                    per_run_avgs = []
                    for run_idx in range(n_runs):
                        run_scores = [d["run_r2_scores"][run_idx] for d in result_details
                                      if len(d.get("run_r2_scores", [])) > run_idx]
                        if run_scores:
                            per_run_avgs.append(np.mean(run_scores))
                    runs_str = ", ".join(f"{s:.2f}" for s in per_run_avgs)
                    print(f"  {mutation.name}: Avg {avg_r2:.4f} [{runs_str}]")
                else:
                    print(f"  {mutation.name}: {avg_r2:.4f}")
        except Exception as e:
            print(f"  Batch evaluation failed: {e}")
            for mutation in offspring:
                mutation.score = -1.0
                mutation.score_vector = []

        # Selection
        population = select_survivors(population, offspring, population_size)
        best = population[0]

        print(f"\nGeneration {gen} complete:")
        print(f"  Best: {best.name} (score: {best.score:.4f})")
        print(f"  Baseline: {baseline_r2:.4f}")
        print(f"  Improvement: {best.score - baseline_r2:+.4f}")

        logger.log_generation(gen, population, offspring, best)

    # Finalize
    logger.finalize(best)

    print("\n" + "=" * 60)
    print("Evolution complete!")
    print("=" * 60)
    print(f"Best mutation: {best.name}")
    print(f"Best score: {best.score:.4f}")
    print(f"Baseline: {baseline_r2:.4f}")
    print(f"Improvement: {best.score - baseline_r2:+.4f}")

    return best


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evolve Julia mutation operators for PySR using LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Evolution settings
    parser.add_argument("--generations", type=int, default=20,
                        help="Number of generations to evolve")
    parser.add_argument("--population", type=int, default=4,
                        help="Population size")
    parser.add_argument("--offspring", type=int, default=4,
                        help="Number of offspring per generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of evaluation runs per mutation per dataset (scores are averaged)")

    # Dataset settings
    parser.add_argument("--split", type=str, default="splits/train_hard.txt",
                        help="Path to dataset split file")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum samples per dataset")
    parser.add_argument("--target_noise", type=float, default=0.0,
                        help="Fixed Gaussian noise level for target (SRBench standard levels: 0.0, 0.001, 0.01, 0.1)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random_target_noise", action="store_true",
                       help="Assign per-dataset target noise from {0.0, 0.001, 0.01, 0.1} using the seed")
    group.add_argument("--no_random_target_noise", dest="random_target_noise", action="store_false",
                       help="Disable per-dataset target noise and use --target_noise instead")
    parser.set_defaults(random_target_noise=True)

    # PySR settings
    parser.add_argument("--max_evals", type=int, default=100000,
                        help="Maximum evaluations per PySR run")
    parser.add_argument("--timeout", type=int, default=3000,
                        help="PySR timeout in seconds")

    # LLM settings
    parser.add_argument("--model", type=str, default="openai/gpt-5-mini",
                        help="LLM model for code generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM temperature (0=deterministic, higher=more random)")

    # SLURM settings
    parser.add_argument("--partition", type=str, default="default_partition",
                        help="SLURM partition")
    parser.add_argument("--time_limit", type=str, default="04:00:00",
                        help="SLURM time limit per job")
    parser.add_argument("--mem_per_cpu", type=str, default="8G",
                        help="SLURM memory per CPU")
    parser.add_argument("--job_timeout", type=float, default=3000.0,
                        help="Max time to wait for SLURM job completion")

    # Output settings
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: outputs/evolve_pysr_TIMESTAMP)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable LLM response caching (force fresh API calls)")

    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/evolve_pysr_{timestamp}"

    # Load datasets
    dataset_names = load_dataset_names_from_split(args.split)
    print(f"Loaded {len(dataset_names)} datasets from {args.split}")

    # Set up PySR kwargs
    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals
    pysr_kwargs["timeout_in_seconds"] = args.timeout

    # Run evolution
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
