#!/usr/bin/env python3
"""
Evolve Julia operators (mutation, survival, or selection) for PySR using LLMs.

Unified evolution script that generates Julia code with an LLM, validates it,
and evaluates performance on SRBench datasets via SLURM.

Usage:
    python evolve_pysr.py --operator_type mutation --split splits/train.txt --generations 20
    python evolve_pysr.py --operator_type survival --split splits/train_hard.txt --generations 20
    python evolve_pysr.py --operator_type selection --split splits/train_hard.txt --generations 20
    python evolve_pysr.py --operator_type all --generations 30  # joint round-robin evolution
    python evolve_pysr.py --operator_type mutation,survival --generations 20  # subset
"""

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# =============================================================================
# Shared Utilities
# =============================================================================

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
    fitness_metric: str = "r2",
) -> List[Tuple[float, List[float], List[Dict]]]:
    """Evaluate configs with optional per-dataset target noise mapping."""
    return evaluator.evaluate_configs(
        configs,
        dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
    )


def extract_julia_code(response: str) -> str:
    """Extract Julia function code from LLM response."""
    text = response.strip()

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

    if "function " not in text:
        return ""

    return text


def extract_function_name(code: str) -> str:
    """Extract function name from Julia code."""
    match = re.search(r'function\s+(\w+)\s*\(', code)
    if match:
        return match.group(1)
    return ""


def pre_validate_julia_syntax(code: str) -> Tuple[bool, str]:
    """Pre-validate Julia code for common LLM-generated syntax errors."""
    named_tuple_pattern = r'\(\s*(\w+)\s*=\s*[^,)]+\s*,\s*\1\s*='
    if re.search(named_tuple_pattern, code):
        return False, "Repeated field name in named tuple (e.g., (left=x, left=y) should be (left=x, right=y))"

    invalid_catch_pattern = r'\bcatch\s+(\d+[\d.eE+-]*|[^;\s\w])'
    if re.search(invalid_catch_pattern, code):
        return False, "Invalid try-catch syntax: use 'catch; ...' or 'catch e; ...' not 'catch <value>'"

    const_in_func_pattern = r'^[ \t]+const\s+'
    if re.search(const_in_func_pattern, code, re.MULTILINE):
        return False, "Cannot use 'const' inside function body (Julia syntax error)"

    return True, ""


def compute_per_run_avgs(
    result_details: List[Dict],
    n_runs: int,
    fitness_metric: str,
) -> List[float]:
    """Compute per-run averages using the same missing-score policy as aggregate scoring."""
    score_key = "run_r2_scores" if fitness_metric == "r2" else "run_gt_scores"
    missing_fill = 0.0 if fitness_metric == "gt" else -1.0
    per_run_avgs: List[float] = []

    for run_idx in range(n_runs):
        run_scores: List[float] = []
        for detail in result_details:
            run_values = detail.get(score_key, [])
            run_scores.append(run_values[run_idx] if len(run_values) > run_idx else missing_fill)
        per_run_avgs.append(float(np.mean(run_scores)) if run_scores else missing_fill)

    return per_run_avgs


def select_parent(population: list, rng: random.Random):
    """Select a parent using tournament selection (size 2)."""
    candidates = rng.sample(population, min(2, len(population)))
    return max(candidates, key=lambda m: m.score if m.score is not None else -1)


def select_survivors(population: list, offspring: list, population_size: int) -> list:
    """Select best individuals from population + offspring."""
    combined = population + offspring
    scored = [m for m in combined if m.score is not None]
    scored.sort(key=lambda m: m.score, reverse=True)
    return scored[:population_size]


# =============================================================================
# Unified Operator Dataclass
# =============================================================================

@dataclass
class JuliaOperator:
    """A Julia operator (mutation, survival, or selection) for PySR."""
    name: str
    code: str
    score: Optional[float] = None
    score_vector: Optional[List[float]] = None
    generation: int = 0
    parent_name: Optional[str] = None
    mode: str = "explore"
    weight: Optional[float] = None  # Only used for mutation operators
    hp_specs: Optional[List[Dict]] = None  # Cached HyperparameterSpec dicts from LLM identification

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'JuliaOperator':
        # Filter to only known fields for backwards compatibility
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


# =============================================================================
# Operator Bundle (for joint evolution of multiple operator types)
# =============================================================================

@dataclass
class OperatorBundle:
    """A bundle of operators (mutation, survival, selection) evaluated together.

    Used for round-robin joint evolution where each generation evolves one
    operator type while keeping the others fixed. The full bundle is evaluated
    as a unit so operator interactions are captured.
    """
    operators: Dict[str, Optional[JuliaOperator]] = field(default_factory=dict)
    score: Optional[float] = None
    score_vector: Optional[List[float]] = None
    best_hparams: Optional[Dict[str, Any]] = None  # Best PySR hparams found by HPO

    @staticmethod
    def create_default() -> 'OperatorBundle':
        """Create a bundle with all default (no custom) operators."""
        return OperatorBundle(operators={})

    def get_operator(self, type_name: str) -> Optional[JuliaOperator]:
        return self.operators.get(type_name)

    def copy_with(self, type_name: str, operator: JuliaOperator) -> 'OperatorBundle':
        """Create a copy with one operator replaced.

        Deep-copies all retained operators so bundles don't share mutable state
        (e.g., HPO mutating .code or .hp_specs on a shared operator).
        Carries forward best_hparams from the parent bundle.
        """
        new_ops = {
            k: copy.deepcopy(v) if k != type_name else operator
            for k, v in self.operators.items()
        }
        new_ops[type_name] = operator
        return OperatorBundle(
            operators=new_ops,
            best_hparams=copy.deepcopy(self.best_hparams) if self.best_hparams else None,
        )

    def to_pysr_config(self, pysr_kwargs: Dict) -> PySRConfig:
        """Convert bundle to PySRConfig with all custom operators set.

        If best_hparams is set (from HPO), merges those into pysr_kwargs
        and mutation_weights accordingly.
        """
        mutation_weights = get_default_mutation_weights()
        config_kwargs: Dict = {}

        mut = self.operators.get("mutation")
        if mut is not None:
            weight = mut.weight if mut.weight is not None else 0.5
            mutation_weights["weight_custom_mutation_1"] = weight
            config_kwargs["custom_mutation_code"] = {mut.name: mut.code}
            config_kwargs["allow_custom_mutations"] = True
        else:
            for i in range(1, 6):
                mutation_weights[f"weight_custom_mutation_{i}"] = 0.0
            config_kwargs["allow_custom_mutations"] = False

        surv = self.operators.get("survival")
        if surv is not None:
            config_kwargs["custom_survival_code"] = surv.code

        sel = self.operators.get("selection")
        if sel is not None:
            config_kwargs["custom_selection_code"] = sel.code

        # Merge HPO-tuned hparams if available
        # Skip op_* keys (operator-specific hparams stored for reference only)
        merged_pysr_kwargs = dict(pysr_kwargs)
        if self.best_hparams:
            for key, val in self.best_hparams.items():
                if key.startswith("op_"):
                    continue  # operator-specific hparam, not a PySR kwarg
                elif key.startswith("weight_"):
                    mutation_weights[key] = val
                else:
                    merged_pysr_kwargs[key] = val

        # Build name from operator names
        name_parts = []
        for t in ["mutation", "survival", "selection"]:
            op = self.operators.get(t)
            name_parts.append(op.name if op else "default")
        name = "__".join(name_parts)

        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=merged_pysr_kwargs,
            name=name,
            **config_kwargs,
        )

    def to_dict(self) -> Dict:
        return {
            "operators": {
                k: v.to_dict() if v is not None else None
                for k, v in self.operators.items()
            },
            "score": self.score,
            "score_vector": self.score_vector,
            "best_hparams": self.best_hparams,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'OperatorBundle':
        operators = {}
        for k, v in d.get("operators", {}).items():
            operators[k] = JuliaOperator.from_dict(v) if v is not None else None
        return cls(
            operators=operators,
            score=d.get("score"),
            score_vector=d.get("score_vector"),
            best_hparams=d.get("best_hparams"),
        )

    @property
    def display_name(self) -> str:
        parts = []
        for t in ["mutation", "survival", "selection"]:
            op = self.operators.get(t)
            parts.append(op.name if op else "default")
        return " | ".join(parts)


# =============================================================================
# Operator Type Definitions
# =============================================================================

class OperatorType(ABC):
    """Base class defining operator-type-specific behavior."""

    name: str  # "mutation", "survival", "selection"

    # Julia validation config
    julia_module: str
    load_func: str
    clear_func: str
    list_func: str

    @abstractmethod
    def load_reference(self) -> str:
        """Load the reference documentation for this operator type."""

    @abstractmethod
    def build_explore_prompt(self, reference: str, variation_seed: int) -> str:
        """Build LLM prompt for exploring new operator ideas."""

    @abstractmethod
    def build_refine_prompt(self, parent_code: str, reference: str, feedback: str) -> str:
        """Build LLM prompt for refining an existing operator."""

    @abstractmethod
    def build_crossover_prompt(self, p1_code: str, p2_code: str, reference: str) -> str:
        """Build LLM prompt for crossing over two operators."""

    @abstractmethod
    def to_pysr_config(self, operator: JuliaOperator, pysr_kwargs: Dict) -> PySRConfig:
        """Convert an operator to a PySRConfig for evaluation."""

    @abstractmethod
    def baseline_config(self, pysr_kwargs: Dict) -> PySRConfig:
        """Create a baseline PySRConfig (no custom operator)."""

    def create_operator(self, name: str, code: str, generation: int = 0,
                        parent_name: Optional[str] = None, mode: str = "explore") -> JuliaOperator:
        """Create a new JuliaOperator with type-specific defaults."""
        return JuliaOperator(
            name=name, code=code, generation=generation,
            parent_name=parent_name, mode=mode,
        )


class MutationOperatorType(OperatorType):
    name = "mutation"
    julia_module = "CustomMutationsModule"
    load_func = "load_mutation_from_string!"
    clear_func = "clear_dynamic_mutations!"
    list_func = "list_available_mutations"

    def load_reference(self) -> str:
        base = Path(__file__).resolve().parent / "SymbolicRegression.jl/src/custom_mutations"
        ref_path = base / "MUTATIONS_REFERENCE2.md"
        if ref_path.exists():
            return ref_path.read_text()
        ref_path = base / "MUTATIONS_REFERENCE.md"
        if ref_path.exists():
            return ref_path.read_text()
        raise FileNotFoundError(f"Could not find MUTATIONS_REFERENCE.md or MUTATIONS_REFERENCE2.md")

    def build_explore_prompt(self, reference: str, variation_seed: int = 0) -> str:
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
        selected_ideas = ideas[variation_seed % len(ideas):] + ideas[:variation_seed % len(ideas)]
        ideas_text = "\n".join(f"- {idea}" for idea in selected_ideas[:4])

        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to create a NEW custom mutation operator for PySR/SymbolicRegression.jl.
The mutation should help discover better symbolic expressions.

## Reference: Existing Mutations and API
{reference}

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

    def build_refine_prompt(self, parent_code: str, reference: str, feedback: str = "") -> str:
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
{reference}

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

    def build_crossover_prompt(self, p1_code: str, p2_code: str, reference: str) -> str:
        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to COMBINE ideas from two mutation operators into a new one.

## Parent Mutation 1
```julia
{p1_code}
```

## Parent Mutation 2
```julia
{p2_code}
```

## Reference: Mutations API
{reference}

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

    def to_pysr_config(self, operator: JuliaOperator, pysr_kwargs: Dict) -> PySRConfig:
        mutation_weights = get_default_mutation_weights()
        weight = operator.weight if operator.weight is not None else 0.5
        mutation_weights["weight_custom_mutation_1"] = weight
        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            custom_mutation_code={operator.name: operator.code},
            allow_custom_mutations=True,
            name=operator.name,
        )

    def baseline_config(self, pysr_kwargs: Dict) -> PySRConfig:
        mutation_weights = get_default_mutation_weights()
        for i in range(1, 6):
            mutation_weights[f"weight_custom_mutation_{i}"] = 0.0
        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            custom_mutation_code=None,
            allow_custom_mutations=False,
            name="baseline",
        )

    def create_operator(self, name: str, code: str, generation: int = 0,
                        parent_name: Optional[str] = None, mode: str = "explore") -> JuliaOperator:
        return JuliaOperator(
            name=name, code=code, generation=generation,
            parent_name=parent_name, mode=mode, weight=0.5,
        )


class SurvivalOperatorType(OperatorType):
    name = "survival"
    julia_module = "CustomSurvivalModule"
    load_func = "load_survival_from_string!"
    clear_func = "clear_dynamic_survivals!"
    list_func = "list_available_survivals"

    def load_reference(self) -> str:
        ref_path = Path(__file__).resolve().parent / "SymbolicRegression.jl/src/custom_survival/SURVIVAL_REFERENCE.md"
        if ref_path.exists():
            return ref_path.read_text()
        raise FileNotFoundError(f"Could not find SURVIVAL_REFERENCE.md at {ref_path}")

    def build_explore_prompt(self, reference: str, variation_seed: int = 0) -> str:
        ideas = [
            "Worst-fitness: Replace the member with the highest cost/loss",
            "Complexity-aware: Replace the most bloated member (highest complexity)",
            "Combined age+fitness: Weight both age and fitness to find replacement",
            "Diversity-preserving: Replace members from overcrowded fitness regions",
            "Tournament-based: Run a mini-tournament and replace the worst",
            "Similarity-based: Replace the member most similar to the incoming offspring",
            "Stagnation-based: Replace members that haven't improved in a while",
            "Random: Uniform random replacement for baseline comparison",
        ]
        selected_ideas = ideas[variation_seed % len(ideas):] + ideas[:variation_seed % len(ideas)]
        ideas_text = "\n".join(f"- {idea}" for idea in selected_ideas[:4])

        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to create a NEW custom survival operator for PySR/SymbolicRegression.jl.
The survival operator decides which population member gets REPLACED when a new offspring is created.

## Reference: Survival API and Default Implementation
{reference}

## Requirements
1. Create a NOVEL survival strategy that differs from the default (replace-oldest)
2. The function should help symbolic regression search find better expressions
3. Use proper Julia syntax and the available API
4. MUST handle the `exclude_indices` keyword argument
5. MUST return a valid index (1 to pop.n)

## Ideas to consider (pick one or invent your own):
{ideas_text}

## Output Format
Return ONLY the Julia function code, nothing else. The function should be named descriptively.
Do not include markdown code blocks or explanations.

Example format:
function my_survival_name(
    pop::Population{{T,L,N}},
    options::AbstractOptions;
    exclude_indices::Vector{{Int}}=Int[],
)::Int where {{T,L,N}}
    # Implementation
    return idx
end
"""

    def build_refine_prompt(self, parent_code: str, reference: str, feedback: str = "") -> str:
        feedback_section = ""
        if feedback:
            feedback_section = f"\n## Feedback on parent survival:\n{feedback}\n"

        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to IMPROVE an existing custom survival operator for PySR/SymbolicRegression.jl.

## Parent Survival Code
```julia
{parent_code}
```
{feedback_section}
## Reference: Survival API
{reference}

## Requirements
1. Keep the core idea but improve the implementation
2. Consider: better edge case handling, smarter heuristics, combining strategies
3. MUST handle the `exclude_indices` keyword argument
4. MUST return a valid index (1 to pop.n)
5. Use proper Julia syntax

## Output Format
Return ONLY the improved Julia function code, nothing else.
Use a NEW function name (append _v2, _improved, etc. or rename descriptively).
Do not include markdown code blocks or explanations.
"""

    def build_crossover_prompt(self, p1_code: str, p2_code: str, reference: str) -> str:
        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to COMBINE ideas from two survival operators into a new one.

## Parent Survival 1
```julia
{p1_code}
```

## Parent Survival 2
```julia
{p2_code}
```

## Reference: Survival API
{reference}

## Requirements
1. Create a NEW survival operator that combines the best ideas from both parents
2. Don't just concatenate - synthesize a coherent new approach
3. MUST handle the `exclude_indices` keyword argument
4. MUST return a valid index (1 to pop.n)
5. Use proper Julia syntax

## Output Format
Return ONLY the new Julia function code, nothing else.
Give it a new descriptive name.
Do not include markdown code blocks or explanations.
"""

    def to_pysr_config(self, operator: JuliaOperator, pysr_kwargs: Dict) -> PySRConfig:
        mutation_weights = get_default_mutation_weights()
        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            custom_survival_code=operator.code,
            name=operator.name,
        )

    def baseline_config(self, pysr_kwargs: Dict) -> PySRConfig:
        mutation_weights = get_default_mutation_weights()
        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            name="baseline",
        )


class SelectionOperatorType(OperatorType):
    name = "selection"
    julia_module = "CustomSelectionModule"
    load_func = "load_selection_from_string!"
    clear_func = "clear_dynamic_selections!"
    list_func = "list_available_selections"

    def load_reference(self) -> str:
        ref_path = Path(__file__).resolve().parent / "SymbolicRegression.jl/src/custom_selection/SELECTION_REFERENCE.md"
        if ref_path.exists():
            return ref_path.read_text()
        raise FileNotFoundError(f"Could not find SELECTION_REFERENCE.md at {ref_path}")

    def build_explore_prompt(self, reference: str, variation_seed: int = 0) -> str:
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
{reference}

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

    def build_refine_prompt(self, parent_code: str, reference: str, feedback: str = "") -> str:
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
{reference}

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

    def build_crossover_prompt(self, p1_code: str, p2_code: str, reference: str) -> str:
        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to COMBINE ideas from two selection operators into a new one.

## Parent Selection 1
```julia
{p1_code}
```

## Parent Selection 2
```julia
{p2_code}
```

## Reference: Selection API
{reference}

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

    def to_pysr_config(self, operator: JuliaOperator, pysr_kwargs: Dict) -> PySRConfig:
        mutation_weights = get_default_mutation_weights()
        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            custom_selection_code=operator.code,
            name=operator.name,
        )

    def baseline_config(self, pysr_kwargs: Dict) -> PySRConfig:
        mutation_weights = get_default_mutation_weights()
        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            name="baseline",
        )


OPERATOR_TYPES: Dict[str, OperatorType] = {
    "mutation": MutationOperatorType(),
    "survival": SurvivalOperatorType(),
    "selection": SelectionOperatorType(),
}


# =============================================================================
# Julia Code Validation
# =============================================================================

def validate_julia_code(name: str, code: str, op_type: OperatorType) -> Tuple[bool, str]:
    """Validate Julia operator code by attempting to load it."""
    is_valid, error = pre_validate_julia_syntax(code)
    if not is_valid:
        return False, error

    try:
        from juliacall import Main as jl

        jl.seval("using SymbolicRegression")
        jl.seval(f"using SymbolicRegression.{op_type.julia_module}")

        jl.seval(f"{op_type.clear_func}()")

        escaped_code = code.replace('"""', '\\"\\"\\"')
        jl.seval(f'{op_type.load_func}(:{name}, raw"""{escaped_code}""")')

        available = list(jl.seval(f"{op_type.list_func}()"))
        if name not in [str(m) for m in available]:
            return False, f"{op_type.name.title()} '{name}' not found in registry after loading"

        return True, ""

    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return False, error_msg


# =============================================================================
# LLM Code Generation
# =============================================================================

def generate_operator_code(
    op_type: OperatorType,
    reference: str,
    parent: Optional[JuliaOperator] = None,
    parent2: Optional[JuliaOperator] = None,
    model: str = "openai/gpt-5-mini",
    mode: str = "explore",
    feedback: str = "",
    variation_seed: int = 0,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> Tuple[str, str]:
    """Generate new Julia operator code using an LLM."""
    if mode == "explore":
        prompt = op_type.build_explore_prompt(reference, variation_seed)
    elif mode == "refine":
        if parent is None:
            raise ValueError("refine mode requires a parent")
        prompt = op_type.build_refine_prompt(parent.code, reference, feedback)
    elif mode == "crossover":
        if parent is None or parent2 is None:
            raise ValueError("crossover mode requires two parents")
        prompt = op_type.build_crossover_prompt(parent.code, parent2.code, reference)
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

def evaluate_bundles(
    bundles: List[OperatorBundle],
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    seed: int = 42,
    n_runs: int = 1,
    target_noise_map: Optional[Dict[str, float]] = None,
    fitness_metric: str = "r2",
) -> List[Tuple[float, List[float], List[Dict]]]:
    """Evaluate multiple operator bundles in parallel via SLURM."""
    if not bundles:
        return []

    configs = [b.to_pysr_config(pysr_kwargs) for b in bundles]

    return _evaluate_configs_with_noise_map(
        evaluator=evaluator,
        configs=configs,
        dataset_names=dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
    )


def evaluate_operators(
    operators: List[JuliaOperator],
    op_type: OperatorType,
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    seed: int = 42,
    n_runs: int = 1,
    target_noise_map: Optional[Dict[str, float]] = None,
    fitness_metric: str = "r2",
) -> List[Tuple[float, List[float], List[Dict]]]:
    """Evaluate multiple operators in parallel via SLURM."""
    if not operators:
        return []

    configs = [op_type.to_pysr_config(op, pysr_kwargs) for op in operators]

    return _evaluate_configs_with_noise_map(
        evaluator=evaluator,
        configs=configs,
        dataset_names=dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
    )


def evaluate_baseline(
    op_type: OperatorType,
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    seed: int = 42,
    n_runs: int = 1,
    target_noise_map: Optional[Dict[str, float]] = None,
    fitness_metric: str = "r2",
) -> Tuple[float, List[float], List[Dict]]:
    """Evaluate PySR with default operator (baseline)."""
    config = op_type.baseline_config(pysr_kwargs)

    results = _evaluate_configs_with_noise_map(
        evaluator=evaluator,
        configs=[config],
        dataset_names=dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
    )
    avg_r2, r2_vector, result_details = results[0]
    return avg_r2, r2_vector, result_details


# =============================================================================
# Logging
# =============================================================================

class EvolutionLogger:
    """Tracks and saves evolution run data."""

    def __init__(self, output_dir: str, operator_type: str = "operator"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.operator_type = operator_type

        self.log_file = self.output_dir / "run.log"
        self.tee = TeeLogger(str(self.log_file))
        sys.stdout = self.tee

        self.run_data = {
            "start_time": datetime.now().isoformat(),
            "config": {},
            "baseline": {},
            "generations": [],
        }

    def set_config(self, config: Dict):
        self.run_data["config"] = config
        self._save()

    def log_baseline(self, avg_r2: float, r2_vector: List[float]):
        self.run_data["baseline"] = {
            "avg_r2": avg_r2,
            "r2_vector": r2_vector,
        }
        self._save()

    def log_generation(
        self,
        generation: int,
        population: List[JuliaOperator],
        offspring: List[JuliaOperator],
        best: JuliaOperator,
    ):
        gen_data = {
            "generation": generation,
            "population": [m.to_dict() for m in population],
            "offspring": [m.to_dict() for m in offspring],
            "best_name": best.name,
            "best_score": best.score,
        }
        self.run_data["generations"].append(gen_data)
        self._save()

        best_file = self.output_dir / f"best_{self.operator_type}_gen{generation}.jl"
        best_file.write_text(f"# Best {self.operator_type} from generation {generation}\n"
                             f"# Score: {best.score}\n\n{best.code}")

    def _save(self):
        with open(self.output_dir / "run_data.json", "w") as f:
            json.dump(self.run_data, f, indent=2)

    def log_bundle_generation(
        self,
        generation: int,
        population: List[OperatorBundle],
        offspring: List[OperatorBundle],
        best: OperatorBundle,
        evolved_type: str,
    ):
        gen_data = {
            "generation": generation,
            "evolved_type": evolved_type,
            "population": [b.to_dict() for b in population],
            "offspring": [b.to_dict() for b in offspring],
            "best_name": best.display_name,
            "best_score": best.score,
        }
        self.run_data["generations"].append(gen_data)
        self._save()

        # Save best bundle's operators
        for type_name, op in best.operators.items():
            if op is not None:
                best_file = self.output_dir / f"best_{type_name}_gen{generation}.jl"
                best_file.write_text(
                    f"# Best {type_name} from generation {generation}\n"
                    f"# Bundle score: {best.score}\n\n{op.code}"
                )

    def finalize(self, best: JuliaOperator):
        self.run_data["end_time"] = datetime.now().isoformat()
        self.run_data[f"best_{self.operator_type}"] = best.to_dict()
        self._save()

        final_file = self.output_dir / f"best_{self.operator_type}_final.jl"
        final_file.write_text(f"# Best {self.operator_type} from evolution run\n"
                              f"# Score: {best.score}\n"
                              f"# Generation: {best.generation}\n\n{best.code}")
        print(f"\nFinal best {self.operator_type} saved to: {final_file}")

    def finalize_bundle(self, best: OperatorBundle):
        self.run_data["end_time"] = datetime.now().isoformat()
        self.run_data["best_bundle"] = best.to_dict()
        self._save()

        for type_name, op in best.operators.items():
            if op is not None:
                final_file = self.output_dir / f"best_{type_name}_final.jl"
                final_file.write_text(
                    f"# Best {type_name} from bundle evolution\n"
                    f"# Bundle score: {best.score}\n\n{op.code}"
                )
                print(f"  Best {type_name} saved to: {final_file}")


# =============================================================================
# Main Evolution Loop (DEPRECATED - use run_bundle_evolution with single type)
# =============================================================================

def run_evolution(
    op_type: OperatorType,
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
    fitness_metric: str = "gt",
    repo_root: Optional[str] = None,
    julia_project: Optional[str] = None,
    python_juliapkg_project: Optional[str] = None,
    julia_depot_path: Optional[str] = None,
) -> JuliaOperator:
    """Run the evolution loop for the specified operator type."""
    rng = random.Random(seed)
    np.random.seed(seed)

    logger = EvolutionLogger(output_dir, operator_type=op_type.name)
    target_noise_map = None
    if random_target_noise:
        target_noise_map = _build_target_noise_map(dataset_names, seed, TARGET_NOISE_LEVELS)

    logger.set_config({
        "operator_type": op_type.name,
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
        "fitness_metric": fitness_metric,
        "repo_root": repo_root,
        "julia_project": julia_project,
        "python_juliapkg_project": python_juliapkg_project,
        "julia_depot_path": julia_depot_path,
    })
    metric_label = "R²" if fitness_metric == "r2" else "GT match rate"

    evaluator = PySRSlurmEvaluator(
        results_dir=output_dir,
        partition=slurm_partition,
        time_limit=slurm_time_limit,
        mem_per_cpu=slurm_mem_per_cpu,
        dataset_max_samples=max_samples,
        data_seed=seed,
        job_timeout=job_timeout,
        target_noise=target_noise,
        repo_root=repo_root,
        julia_project=julia_project,
        python_juliapkg_project=python_juliapkg_project,
        julia_depot_path=julia_depot_path,
    )

    reference = op_type.load_reference()

    # Evaluate baseline
    print("=" * 60)
    print(f"Evaluating baseline (default {op_type.name})...")
    print("=" * 60)
    baseline_r2, baseline_vector, baseline_details = evaluate_baseline(
        op_type, evaluator, dataset_names, pysr_kwargs, seed,
        n_runs=n_runs, target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
    )
    if n_runs > 1 and baseline_details:
        per_run_avgs = compute_per_run_avgs(
            baseline_details, n_runs=n_runs, fitness_metric=fitness_metric,
        )
        runs_str = ", ".join(f"{s:.4f}" for s in per_run_avgs)
        print(f"Baseline avg {metric_label}: {baseline_r2:.4f} [{runs_str}]")
    else:
        print(f"Baseline avg {metric_label}: {baseline_r2:.4f}")
    logger.log_baseline(baseline_r2, baseline_vector)

    # Generate initial population
    print("\n" + "=" * 60)
    print(f"Generating initial population ({population_size} {op_type.name} operators)...")
    print("=" * 60)

    population: List[JuliaOperator] = []
    attempts = 0
    max_attempts = population_size * 3

    while len(population) < population_size and attempts < max_attempts:
        attempts += 1
        print(f"\nGenerating {op_type.name} {len(population) + 1}/{population_size} (attempt {attempts})...")

        code, func_name = generate_operator_code(
            op_type=op_type,
            reference=reference,
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

        is_valid, error = validate_julia_code(unique_name, code, op_type)
        if not is_valid:
            print(f"  Validation failed: {error[:100]}...")
            continue

        operator = op_type.create_operator(
            name=unique_name, code=code, generation=0, mode="explore",
        )
        population.append(operator)
        print(f"  Created: {unique_name}")

    if len(population) == 0:
        raise RuntimeError(f"Failed to generate any valid {op_type.name} operators")

    print(f"\nGenerated {len(population)} valid {op_type.name} operators")

    # Evaluate initial population
    print("\n" + "=" * 60)
    print(f"Evaluating initial population ({len(population)} operators in parallel)...")
    print("=" * 60)

    try:
        results = evaluate_operators(
            population, op_type, evaluator, dataset_names, pysr_kwargs, seed,
            n_runs=n_runs, target_noise_map=target_noise_map,
            fitness_metric=fitness_metric,
        )
        for operator, (avg_r2, r2_vector, result_details) in zip(population, results):
            operator.score = avg_r2
            operator.score_vector = r2_vector
            if n_runs > 1 and result_details:
                per_run_avgs = compute_per_run_avgs(
                    result_details, n_runs=n_runs, fitness_metric=fitness_metric,
                )
                runs_str = ", ".join(f"{s:.4f}" for s in per_run_avgs)
                print(f"  {operator.name}: Avg {avg_r2:.4f} [{runs_str}]")
            else:
                print(f"  {operator.name}: {avg_r2:.4f}")
    except Exception as e:
        print(f"  Batch evaluation failed: {e}")
        for operator in population:
            operator.score = -1.0
            operator.score_vector = []

    population.sort(key=lambda s: s.score if s.score else -1, reverse=True)
    best = population[0]
    print(f"\nBest initial {op_type.name}: {best.name} (score: {best.score:.4f})")

    # Evolution loop
    for gen in range(1, n_generations + 1):
        print("\n" + "=" * 60)
        print(f"Generation {gen}/{n_generations}")
        print("=" * 60)

        offspring: List[JuliaOperator] = []
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

            code, func_name = generate_operator_code(
                op_type=op_type,
                reference=reference,
                parent=parent,
                parent2=parent2,
                model=model,
                mode=mode,
                variation_seed=gen * 100 + offspring_attempts,
                temperature=temperature,
                use_cache=use_cache,
            )

            if not code or not func_name:
                continue

            unique_name = f"{func_name}_gen{gen}_{len(offspring)}"
            code = code.replace(f"function {func_name}(", f"function {unique_name}(", 1)

            is_valid, error = validate_julia_code(unique_name, code, op_type)
            if not is_valid:
                print(f"  Validation failed for {unique_name}: {error[:80]}...")
                continue

            operator = op_type.create_operator(
                name=unique_name, code=code, generation=gen,
                parent_name=parent.name if parent else None, mode=mode,
            )
            offspring.append(operator)
            print(f"  Created: {unique_name} (mode={mode})")

        print(f"\nGenerated {len(offspring)} offspring")

        # Evaluate offspring
        print(f"\nEvaluating {len(offspring)} offspring in parallel...")
        try:
            results = evaluate_operators(
                offspring, op_type, evaluator, dataset_names, pysr_kwargs, seed,
                n_runs=n_runs, target_noise_map=target_noise_map,
                fitness_metric=fitness_metric,
            )
            for operator, (avg_r2, r2_vector, result_details) in zip(offspring, results):
                operator.score = avg_r2
                operator.score_vector = r2_vector
                if n_runs > 1 and result_details:
                    per_run_avgs = compute_per_run_avgs(
                        result_details, n_runs=n_runs, fitness_metric=fitness_metric,
                    )
                    runs_str = ", ".join(f"{s:.4f}" for s in per_run_avgs)
                    print(f"  {operator.name}: Avg {avg_r2:.4f} [{runs_str}]")
                else:
                    print(f"  {operator.name}: {avg_r2:.4f}")
        except Exception as e:
            print(f"  Batch evaluation failed: {e}")
            for operator in offspring:
                operator.score = -1.0
                operator.score_vector = []

        population = select_survivors(population, offspring, population_size)
        best = population[0]

        print(f"\nGeneration {gen} complete:")
        print(f"  Best: {best.name} (score: {best.score:.4f})")
        print(f"  Baseline ({metric_label}): {baseline_r2:.4f}")
        print(f"  Improvement: {best.score - baseline_r2:+.4f}")

        logger.log_generation(gen, population, offspring, best)

    logger.finalize(best)

    print("\n" + "=" * 60)
    print("Evolution complete!")
    print("=" * 60)
    print(f"Best {op_type.name}: {best.name}")
    print(f"Best score: {best.score:.4f}")
    print(f"Baseline ({metric_label}): {baseline_r2:.4f}")
    print(f"Improvement: {best.score - baseline_r2:+.4f}")

    return best


# =============================================================================
# Bundle Evolution Loop (joint evolution of multiple operator types)
# =============================================================================

def run_bundle_evolution(
    operator_type_names: List[str],
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
    fitness_metric: str = "gt",
    hp_tuning_trials: int = 0,
    repo_root: Optional[str] = None,
    julia_project: Optional[str] = None,
    python_juliapkg_project: Optional[str] = None,
    julia_depot_path: Optional[str] = None,
) -> OperatorBundle:
    """Run round-robin bundle evolution across multiple operator types.

    Each generation evolves one operator type (cycling round-robin), while
    keeping the other operators in each bundle fixed. The full bundle is
    evaluated as a unit so operator interactions are captured.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    op_types = [OPERATOR_TYPES[name] for name in operator_type_names]
    references = {name: OPERATOR_TYPES[name].load_reference() for name in operator_type_names}

    logger = EvolutionLogger(output_dir, operator_type="bundle")
    target_noise_map = None
    if random_target_noise:
        target_noise_map = _build_target_noise_map(dataset_names, seed, TARGET_NOISE_LEVELS)

    logger.set_config({
        "operator_types": operator_type_names,
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
        "fitness_metric": fitness_metric,
        "repo_root": repo_root,
        "julia_project": julia_project,
        "python_juliapkg_project": python_juliapkg_project,
        "julia_depot_path": julia_depot_path,
    })
    metric_label = "R²" if fitness_metric == "r2" else "GT match rate"

    evaluator = PySRSlurmEvaluator(
        results_dir=output_dir,
        partition=slurm_partition,
        time_limit=slurm_time_limit,
        mem_per_cpu=slurm_mem_per_cpu,
        dataset_max_samples=max_samples,
        data_seed=seed,
        job_timeout=job_timeout,
        target_noise=target_noise,
        repo_root=repo_root,
        julia_project=julia_project,
        python_juliapkg_project=python_juliapkg_project,
        julia_depot_path=julia_depot_path,
    )

    # Evaluate baseline (all defaults)
    print("=" * 60)
    print("Evaluating baseline (all default operators)...")
    print("=" * 60)
    baseline_bundle = OperatorBundle.create_default()
    baseline_config = baseline_bundle.to_pysr_config(pysr_kwargs)
    baseline_results = _evaluate_configs_with_noise_map(
        evaluator=evaluator,
        configs=[baseline_config],
        dataset_names=dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
    )
    baseline_score, baseline_vector, baseline_details = baseline_results[0]
    baseline_bundle.score = baseline_score
    baseline_bundle.score_vector = baseline_vector

    if n_runs > 1 and baseline_details:
        per_run_avgs = compute_per_run_avgs(baseline_details, n_runs=n_runs, fitness_metric=fitness_metric)
        runs_str = ", ".join(f"{s:.4f}" for s in per_run_avgs)
        print(f"Baseline avg {metric_label}: {baseline_score:.4f} [{runs_str}]")
    else:
        print(f"Baseline avg {metric_label}: {baseline_score:.4f}")
    logger.log_baseline(baseline_score, baseline_vector)

    # Generate initial population of bundles
    # Each bundle starts with one random operator per type
    print("\n" + "=" * 60)
    print(f"Generating initial population ({population_size} bundles)...")
    print(f"Operator types: {', '.join(operator_type_names)}")
    print("=" * 60)

    population: List[OperatorBundle] = []
    max_bundle_attempts = population_size * 2
    bundle_attempts = 0
    while len(population) < population_size and bundle_attempts < max_bundle_attempts:
        bundle_idx = bundle_attempts
        bundle_attempts += 1
        bundle = OperatorBundle.create_default()
        n_generated = 0
        print(f"\nBundle {len(population) + 1}/{population_size} (attempt {bundle_idx + 1}):")

        for type_name in operator_type_names:
            op_type = OPERATOR_TYPES[type_name]
            reference = references[type_name]

            # Try to generate a valid operator for this type
            generated = False
            for attempt in range(3):
                code, func_name = generate_operator_code(
                    op_type=op_type,
                    reference=reference,
                    model=model,
                    mode="explore",
                    variation_seed=bundle_idx * 100 + attempt,
                    temperature=temperature,
                    use_cache=use_cache,
                )
                if not code or not func_name:
                    continue

                unique_name = f"{func_name}_init_{bundle_idx}"
                code = code.replace(f"function {func_name}(", f"function {unique_name}(", 1)

                is_valid, error = validate_julia_code(unique_name, code, op_type)
                if not is_valid:
                    print(f"  {type_name}: validation failed (attempt {attempt + 1}): {error[:80]}...")
                    continue

                operator = op_type.create_operator(
                    name=unique_name, code=code, generation=0, mode="explore",
                )
                bundle = bundle.copy_with(type_name, operator)
                print(f"  {type_name}: {unique_name}")
                generated = True
                n_generated += 1
                break

            if not generated:
                print(f"  {type_name}: failed to generate after 3 attempts")

        if n_generated == 0:
            print(f"  Skipping bundle (no operators generated)")
            continue

        if n_generated < len(operator_type_names):
            print(f"  Warning: bundle has {n_generated}/{len(operator_type_names)} operators")

        population.append(bundle)

    if not population:
        raise RuntimeError("Failed to generate any valid bundles")

    # Evaluate initial population
    print("\n" + "=" * 60)
    print(f"Evaluating initial population ({len(population)} bundles)...")
    print("=" * 60)

    try:
        results = evaluate_bundles(
            population, evaluator, dataset_names, pysr_kwargs, seed,
            n_runs=n_runs, target_noise_map=target_noise_map,
            fitness_metric=fitness_metric,
        )
        for bundle, (avg_score, score_vector, result_details) in zip(population, results):
            bundle.score = avg_score
            bundle.score_vector = score_vector
            if n_runs > 1 and result_details:
                per_run_avgs = compute_per_run_avgs(result_details, n_runs=n_runs, fitness_metric=fitness_metric)
                runs_str = ", ".join(f"{s:.4f}" for s in per_run_avgs)
                print(f"  {bundle.display_name}: Avg {avg_score:.4f} [{runs_str}]")
            else:
                print(f"  {bundle.display_name}: {avg_score:.4f}")
    except Exception as e:
        print(f"  Batch evaluation failed: {e}")
        for bundle in population:
            bundle.score = -1.0
            bundle.score_vector = []

    population.sort(key=lambda b: b.score if b.score is not None else -1, reverse=True)
    best = population[0]
    print(f"\nBest initial bundle: {best.display_name} (score: {best.score:.4f})")

    # Evolution loop (round-robin across operator types)
    for gen in range(1, n_generations + 1):
        # Round-robin: pick which operator type to evolve this generation
        current_type_name = operator_type_names[(gen - 1) % len(operator_type_names)]
        current_op_type = OPERATOR_TYPES[current_type_name]
        reference = references[current_type_name]

        print("\n" + "=" * 60)
        print(f"Generation {gen}/{n_generations} — Evolving {current_type_name.upper()}")
        print("=" * 60)

        offspring_bundles: List[OperatorBundle] = []
        offspring_attempts = 0
        max_offspring_attempts = n_offspring * 3

        while len(offspring_bundles) < n_offspring and offspring_attempts < max_offspring_attempts:
            offspring_attempts += 1

            # Select parent bundle
            parent_bundle = select_parent(population, rng)
            parent_op = parent_bundle.get_operator(current_type_name)

            # Choose mode: if parent has no custom operator for this type, explore
            if parent_op is None:
                mode = "explore"
                parent = None
                parent2 = None
            else:
                mode = rng.choice(["explore", "refine", "refine", "refine", "crossover"])
                if mode == "explore":
                    parent = None
                    parent2 = None
                elif mode == "refine":
                    parent = parent_op
                    parent2 = None
                else:  # crossover
                    parent = parent_op
                    # Find a second parent from a different bundle
                    other_candidates = [b for b in population if b != parent_bundle]
                    if not other_candidates:
                        # Only one bundle in population, fall back to refine
                        mode = "refine"
                        parent2 = None
                    else:
                        other_bundle = select_parent(other_candidates, rng)
                        parent2 = other_bundle.get_operator(current_type_name)
                        if parent2 is None:
                            # Other parent has no custom operator, fall back to refine
                            mode = "refine"
                            parent2 = None

            code, func_name = generate_operator_code(
                op_type=current_op_type,
                reference=reference,
                parent=parent,
                parent2=parent2,
                model=model,
                mode=mode,
                variation_seed=gen * 100 + offspring_attempts,
                temperature=temperature,
                use_cache=use_cache,
            )

            if not code or not func_name:
                continue

            unique_name = f"{func_name}_gen{gen}_{len(offspring_bundles)}"
            code = code.replace(f"function {func_name}(", f"function {unique_name}(", 1)

            is_valid, error = validate_julia_code(unique_name, code, current_op_type)
            if not is_valid:
                print(f"  Validation failed for {unique_name}: {error[:80]}...")
                continue

            new_op = current_op_type.create_operator(
                name=unique_name, code=code, generation=gen,
                parent_name=parent.name if parent else None, mode=mode,
            )
            # Create new bundle: keep all other operators from parent, replace evolved type
            new_bundle = parent_bundle.copy_with(current_type_name, new_op)
            offspring_bundles.append(new_bundle)
            print(f"  Created: {unique_name} (mode={mode})")

        print(f"\nGenerated {len(offspring_bundles)} offspring bundles")

        # Evaluate offspring bundles
        print(f"\nEvaluating {len(offspring_bundles)} offspring bundles...")
        try:
            results = evaluate_bundles(
                offspring_bundles, evaluator, dataset_names, pysr_kwargs, seed,
                n_runs=n_runs, target_noise_map=target_noise_map,
                fitness_metric=fitness_metric,
            )
            for bundle, (avg_score, score_vector, result_details) in zip(offspring_bundles, results):
                bundle.score = avg_score
                bundle.score_vector = score_vector
                if n_runs > 1 and result_details:
                    per_run_avgs = compute_per_run_avgs(result_details, n_runs=n_runs, fitness_metric=fitness_metric)
                    runs_str = ", ".join(f"{s:.4f}" for s in per_run_avgs)
                    print(f"  {bundle.display_name}: Avg {avg_score:.4f} [{runs_str}]")
                else:
                    print(f"  {bundle.display_name}: {avg_score:.4f}")
        except Exception as e:
            print(f"  Batch evaluation failed: {e}")
            for bundle in offspring_bundles:
                bundle.score = -1.0
                bundle.score_vector = []

        population = select_survivors(population, offspring_bundles, population_size)
        best = population[0]

        # HPO tuning step
        if hp_tuning_trials > 0:
            print(f"\n--- HPO Tuning ({hp_tuning_trials} trials per bundle) ---")
            from hpo_evolve_pysr import tune_population
            score_before = best.score
            population = tune_population(
                population=population,
                evaluator=evaluator,
                dataset_names=dataset_names,
                pysr_kwargs=pysr_kwargs,
                operator_type_names=operator_type_names,
                n_trials=hp_tuning_trials,
                seed=seed,
                output_dir=output_dir,
                model=model,
                n_runs=n_runs,
                target_noise_map=target_noise_map,
                fitness_metric=fitness_metric,
            )
            best = population[0]
            if best.score != score_before:
                print(f"  HPO: {score_before:.4f} -> {best.score:.4f} ({best.score - score_before:+.4f})")

        print(f"\nGeneration {gen} complete:")
        print(f"  Evolved: {current_type_name}")
        print(f"  Best: {best.display_name} (score: {best.score:.4f})")
        print(f"  Baseline ({metric_label}): {baseline_score:.4f}")
        print(f"  Improvement: {best.score - baseline_score:+.4f}")

        logger.log_bundle_generation(gen, population, offspring_bundles, best, current_type_name)

    logger.finalize_bundle(best)

    print("\n" + "=" * 60)
    print("Bundle evolution complete!")
    print("=" * 60)
    print(f"Best bundle: {best.display_name}")
    print(f"Best score: {best.score:.4f}")
    print(f"Baseline ({metric_label}): {baseline_score:.4f}")
    print(f"Improvement: {best.score - baseline_score:+.4f}")

    return best


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evolve Julia operators (mutation/survival/selection) for PySR using LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--operator_type", type=str, required=True,
                        help="Type of operator to evolve: mutation, survival, selection, "
                             "all (all three jointly), or comma-separated list (e.g. mutation,survival)")

    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--offspring", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--fitness_metric", type=str, default="gt", choices=["r2", "gt"],
                        help="Meta-evolution fitness metric: r2 or gt (whole-frontier symbolic match rate)")
    parser.add_argument("--hp_tuning_trials", type=int, default=0,
                        help="HPO trials per bundle per generation (0=disabled)")

    parser.add_argument("--split", type=str, default=None,
                        help="Path to dataset split file (default: train.txt for mutation, train_hard.txt for survival/selection)")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--target_noise", type=float, default=0.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random_target_noise", action="store_true")
    group.add_argument("--no_random_target_noise", dest="random_target_noise", action="store_false")
    parser.set_defaults(random_target_noise=True)

    parser.add_argument("--max_evals", type=int, default=None,
                        help="Maximum evaluations per PySR run (default: 1e6 for mutation, 100000 for survival/selection)")
    parser.add_argument("--timeout", type=int, default=3000)

    parser.add_argument("--model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--time_limit", type=str, default="04:00:00")
    parser.add_argument("--mem_per_cpu", type=str, default="8G")
    parser.add_argument("--job_timeout", type=float, default=3000.0)
    parser.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parent),
                        help="Repo root containing PySR and SymbolicRegression.jl.")
    parser.add_argument("--julia-project", type=str, default=None,
                        help="Explicit JULIA_PROJECT path (default: <repo-root>/SymbolicRegression.jl).")
    parser.add_argument("--python-juliapkg-project", type=str, default=None,
                        help="Optional PYTHON_JULIAPKG_PROJECT for an isolated Julia package environment.")
    parser.add_argument("--julia-depot-path", type=str, default=None,
                        help="Optional JULIA_DEPOT_PATH override.")

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")

    args = parser.parse_args()

    # Parse operator type(s)
    if args.operator_type == "all":
        operator_type_names = ["mutation", "survival", "selection"]
    else:
        operator_type_names = [t.strip() for t in args.operator_type.split(",")]
        for name in operator_type_names:
            if name not in OPERATOR_TYPES:
                parser.error(f"Unknown operator type: {name}. Choose from: mutation, survival, selection, all")

    # Apply defaults based on which types are included
    has_mutation = "mutation" in operator_type_names
    if args.split is None:
        args.split = "splits/train.txt" if has_mutation else "splits/train_hard.txt"

    if args.max_evals is None:
        args.max_evals = int(1e6) if has_mutation else 100000

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        type_label = "+".join(operator_type_names) if len(operator_type_names) > 1 else operator_type_names[0]
        args.output_dir = f"outputs/evolve_{type_label}_{timestamp}"

    dataset_names = load_dataset_names_from_split(args.split)
    print(f"Loaded {len(dataset_names)} datasets from {args.split}")

    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals
    pysr_kwargs["timeout_in_seconds"] = args.timeout

    common_kwargs = dict(
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
        fitness_metric=args.fitness_metric,
        hp_tuning_trials=args.hp_tuning_trials,
        repo_root=args.repo_root,
        julia_project=args.julia_project or str((Path(args.repo_root) / "SymbolicRegression.jl").resolve()),
        python_juliapkg_project=args.python_juliapkg_project,
        julia_depot_path=args.julia_depot_path,
    )

    if len(operator_type_names) > 1:
        print(f"Bundle evolution: {', '.join(operator_type_names)} (round-robin)")
    else:
        print(f"Evolving: {operator_type_names[0]}")

    best = run_bundle_evolution(
        operator_type_names=operator_type_names,
        **common_kwargs,
    )

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
