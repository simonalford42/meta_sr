#!/usr/bin/env python3
"""
Operator dataclasses, operator-type ABC + subclasses, Julia code validation,
and LLM-driven operator-code generation.

Extracted from evolve_pysr.py during the refactor. Bodies are byte-identical
to the originals in evolve_pysr_old.py.
"""

import copy
import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from completions import chat_completion, get_content
from parallel_eval_pysr import (
    PySRConfig,
    get_default_mutation_weights,
    get_default_pysr_kwargs,
)

class ModelEnsemble:
    """Ensemble of LLM models with weighted random sampling.

    Mirrors OpenEvolve's LLMEnsemble: each call to sample() picks a model
    based on normalized weights, using a seeded RNG for reproducibility.
    """

    def __init__(self, models: List[Tuple[str, float]], seed: int = 42):
        if not models:
            raise ValueError("ModelEnsemble requires at least one model")
        self.models = [(name, weight) for name, weight in models]
        total = sum(w for _, w in self.models)
        self.weights = [w / total for _, w in self.models]
        self.rng = random.Random(seed)

    def sample(self) -> str:
        """Sample a model name based on weights."""
        idx = self.rng.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        name = self.models[idx][0]
        if len(self.models) > 1:
            print(f"      [Ensemble] sampled model: {name}")
        return name

    @classmethod
    def from_str(cls, spec: str, seed: int = 42) -> 'ModelEnsemble':
        """Parse a spec like 'model1:0.8,model2:0.2' or just 'model1'.

        Format per entry: model_name[:weight]  (weight defaults to 1.0)
        """
        models = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                # Could be model:weight or scheme://host/model:weight
                # Split on the *last* colon to handle URLs
                last_colon = part.rfind(":")
                weight_candidate = part[last_colon + 1:]
                try:
                    weight = float(weight_candidate)
                    name = part[:last_colon]
                except ValueError:
                    # Last colon is part of the model name (e.g. no weight)
                    name = part
                    weight = 1.0
            else:
                name = part
                weight = 1.0
            models.append((name, weight))
        return cls(models, seed=seed)

    def to_config_dict(self) -> List[Dict[str, Any]]:
        """Serialize for logging."""
        return [{"model": name, "weight": weight} for name, weight in self.models]

    def __repr__(self) -> str:
        parts = [f"{name}:{w:.2f}" for name, w in self.models]
        return f"ModelEnsemble([{', '.join(parts)}])"

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

_BRAINSTORM_INSTRUCTION = (
    "The SR algorithm failed to discover the ground-truth equation for this task. "
    "Examine how the Pareto front of best equations evolved over the course of the search, "
    "and consider proposing an operator that would better reach the GT structure."
)

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
    result_details: Optional[List[Dict]] = None  # Per-dataset evaluation details
    weight: Optional[float] = None  # Only used for mutation operators
    model: Optional[str] = None  # LLM model that generated this operator
    hp_specs: Optional[List[Dict]] = None  # Cached HyperparameterSpec dicts from LLM identification
    seeds_evaluated: int = 0  # Number of PySR seeds accumulated in result_details (racing mode)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'JuliaOperator':
        # Filter to only known fields for backwards compatibility
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

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
    result_details: Optional[List[Dict]] = None  # Per-dataset evaluation details
    best_hparams: Optional[Dict[str, Any]] = None  # Best PySR hparams found by HPO
    seeds_evaluated: int = 0  # Number of PySR seeds accumulated in result_details (racing mode)

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
            "result_details": self.result_details,
            "best_hparams": self.best_hparams,
            "seeds_evaluated": self.seeds_evaluated,
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
            result_details=d.get("result_details"),
            best_hparams=d.get("best_hparams"),
            seeds_evaluated=d.get("seeds_evaluated", 0),
        )

    @property
    def display_name(self) -> str:
        parts = []
        for t in ["mutation", "survival", "selection"]:
            op = self.operators.get(t)
            parts.append(op.name if op else "default")
        return " | ".join(parts)

class OperatorType(ABC):
    """Base class defining operator-type-specific behavior."""

    name: str  # "mutation", "survival", "selection"

    # Julia validation config
    julia_module: str
    load_func: str
    clear_func: str
    list_func: str
    smoke_test_julia: str = ""  # Julia code template for runtime smoke test

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
    smoke_test_julia = """
    let
        using SymbolicRegression: Options, Node, AbstractExpressionNode
        using SymbolicRegression.CustomMutationsModule: apply_custom_mutation
        using Random: Xoshiro
        options = Options(;
            binary_operators=[+, -, *, /],
            unary_operators=[sin, cos],
        )
        # Build a small tree: x1 + 0.5
        tree = Node(Float64; op=1, l=Node(Float64; feature=1), r=Node(Float64; val=0.5))
        rng = Xoshiro(42)
        result = apply_custom_mutation(:{name}, tree, options, 3, rng)
        @assert result isa AbstractExpressionNode "Smoke test: mutation must return a Node, got $(typeof(result))"
    end
    """

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

    def build_task_aware_explore_prompt(
        self,
        reference: str,
        unsolved_tasks_text: str,
        variation_seed: int = 0,
    ) -> str:
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
The mutation should help discover better symbolic expressions — in particular, it should
help PySR reach the kinds of structures appearing in the unsolved target equations below,
which neither the baseline nor the current population has managed to discover.

## Reference: Existing Mutations and API
{reference}

## Unsolved target equation(s) (for inspiration only — do NOT hard-code)
{unsolved_tasks_text}

Think about what structural moves (e.g. inserting particular subexpressions, rewriting
patterns, exploring certain operators or constants) would make it likelier for a search
using this mutation to discover expressions of that form. Then design a mutation whose
proposals bias the search toward such structures while remaining a general operator.

## Requirements
1. Create a NOVEL mutation that does something different from existing mutations.
2. Do NOT hard-code the target equations — the mutation must be a general operator
   useful across many symbolic regression problems.
3. Use proper Julia syntax and the available API.

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

    def build_task_aware_crossover_prompt(
        self,
        p1_code: str,
        p2_code: str,
        reference: str,
        p1_tasks_text: str,
        p2_tasks_text: str,
    ) -> str:
        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to COMBINE two mutation operators so that the resulting operator can solve
BOTH of the complementary task sets below. Each parent already solves a different subset
of tasks (that the baseline cannot solve). Your job is to synthesize a new mutation that
generalizes so it can help the search reach both target equations.

## Parent Mutation 1 (solves these tasks the other parent and baseline do not)
```julia
{p1_code}
```

Ground-truth equations Parent 1 solves (that Parent 2 / baseline do not):
{p1_tasks_text}

## Parent Mutation 2 (solves these tasks the other parent and baseline do not)
```julia
{p2_code}
```

Ground-truth equations Parent 2 solves (that Parent 1 / baseline do not):
{p2_tasks_text}

## Reference: Mutations API
{reference}

## Requirements
1. Create a NEW mutation that combines the best ideas from both parents so it can help
   PySR discover the kinds of structures present in BOTH task sets above.
2. Do NOT hard-code the target equations — the mutation must be a general operator that
   works across many symbolic regression problems. Use the equations only as inspiration
   for the structural moves your mutation should make available.
3. Don't just concatenate — synthesize a coherent new approach.
4. Use proper Julia syntax and the available API.

## Output Format
Return ONLY the new Julia function code, nothing else.
Give it a new descriptive name.
Do not include markdown code blocks or explanations.
"""

    def build_task_aware_refine_prompt(
        self,
        parent_code: str,
        reference: str,
        unsolved_tasks_text: str,
    ) -> str:
        return f"""You are an expert in symbolic regression and genetic programming.

Your task is to IMPROVE an existing custom mutation operator for PySR so that it can
help the search solve specific target equations it has so far FAILED to discover.

## Parent Mutation Code
```julia
{parent_code}
```

## Unsolved target equation(s)
The parent mutation has not helped PySR discover these ground-truth equations yet:
{unsolved_tasks_text}

Think about what structural moves (e.g. inserting particular subexpressions, rewriting
patterns, exploring certain operators or constants) would make it likelier for a
search using this mutation to reach expressions of that form. Then modify the mutation
to make those moves more likely.

## Reference: Mutations API
{reference}

## Requirements
1. Do NOT hard-code the target equation — the mutation must remain a general operator
   useful across many problems. Use the target equation only as motivation.
2. Keep the core idea of the parent but bias it toward the structures above.
3. Use proper Julia syntax.

## Output Format
Return ONLY the improved Julia function code, nothing else.
Use a NEW function name (append _v2, _improved, etc. or rename descriptively).
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
    smoke_test_julia = """
    let
        using SymbolicRegression: Options, Dataset
        using SymbolicRegression.PopulationModule: Population
        using SymbolicRegression.CustomSurvivalModule: apply_custom_survival
        options = Options(;
            binary_operators=[+, -, *, /],
            unary_operators=[sin, cos],
            populations=1,
            population_size=20,
            tournament_selection_n=5,
        )
        X = randn(Float64, 3, 30)
        y = randn(Float64, 30)
        dataset = Dataset(X, y)
        pop = Population(dataset; options=options, population_size=20, nfeatures=3)
        idx = apply_custom_survival(pop, options; exclude_indices=Int[])
        @assert idx isa Integer "Smoke test: survival must return Int, got $(typeof(idx))"
        @assert 1 <= idx <= pop.n "Smoke test: survival returned index $idx, must be in 1:$(pop.n)"
    end
    """

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
    smoke_test_julia = """
    let
        using SymbolicRegression: Options, Dataset
        using SymbolicRegression.PopMemberModule: PopMember
        using SymbolicRegression.PopulationModule: Population
        using SymbolicRegression.AdaptiveParsimonyModule: RunningSearchStatistics
        using SymbolicRegression.CustomSelectionModule: apply_custom_selection
        options = Options(;
            binary_operators=[+, -, *, /],
            unary_operators=[sin, cos],
            populations=1,
            population_size=20,
            tournament_selection_n=5,
        )
        X = randn(Float64, 3, 30)
        y = randn(Float64, 30)
        dataset = Dataset(X, y)
        pop = Population(dataset; options=options, population_size=20, nfeatures=3)
        rss = RunningSearchStatistics(; options=options)
        result = apply_custom_selection(pop, rss, options)
        @assert result isa PopMember "Smoke test: selection must return PopMember, got $(typeof(result))"
    end
    """

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

def validate_julia_code(name: str, code: str, op_type: OperatorType) -> Tuple[bool, str]:
    """Validate Julia operator code by attempting to load it and smoke-testing it."""
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

        # Smoke test: actually invoke the operator on synthetic inputs
        if op_type.smoke_test_julia:
            smoke_code = op_type.smoke_test_julia.replace(":{name}", f":{name}")
            jl.seval(smoke_code)

        return True, ""

    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return False, error_msg

def smoke_test_operator(name: str, code: str, op_type: OperatorType) -> Tuple[bool, str]:
    """Run a runtime smoke test on an already-loaded operator.

    Loads the operator fresh and invokes it on synthetic inputs.
    Returns (passed, error_message).
    """
    if not op_type.smoke_test_julia:
        return True, ""
    try:
        from juliacall import Main as jl

        jl.seval("using SymbolicRegression")
        jl.seval(f"using SymbolicRegression.{op_type.julia_module}")
        jl.seval(f"{op_type.clear_func}()")

        escaped_code = code.replace('"""', '\\"\\"\\"')
        jl.seval(f'{op_type.load_func}(:{name}, raw"""{escaped_code}""")')

        smoke_code = op_type.smoke_test_julia.replace(":{name}", f":{name}")
        jl.seval(smoke_code)
        return True, ""
    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return False, error_msg

def append_validation_log(
    log_prompt_dir: Optional[Path],
    op_type: OperatorType,
    mode: str,
    generation: int,
    variation_seed: int,
    is_valid: bool,
    error: str,
    unique_name: str,
) -> None:
    """Append a validation-outcome section to the prompt log file for this generation attempt."""
    if log_prompt_dir is None:
        return
    try:
        fname = f"gen{max(generation, 0):03d}_{op_type.name}_{mode}_seed{variation_seed}.md"
        path = log_prompt_dir / fname
        if not path.exists():
            return
        section = (
            f"\n## Validation\n\n"
            f"- unique_name: `{unique_name}`\n"
            f"- result: {'PASS' if is_valid else 'FAIL'}\n"
        )
        if not is_valid:
            section += f"- error:\n```\n{error}\n```\n"
        with open(path, "a") as f:
            f.write(section)
    except Exception as e:
        print(f"  [prompt-log] Failed to append validation: {e}")

def smoke_test_bundle(bundle: 'OperatorBundle') -> Tuple[bool, List[str]]:
    """Smoke test all operators in a bundle.

    Returns (all_passed, list_of_error_messages).
    """
    errors = []
    for type_name in ("mutation", "survival", "selection"):
        op = bundle.get_operator(type_name)
        if op is None:
            continue
        op_type = OPERATOR_TYPES[type_name]
        passed, error = smoke_test_operator(op.name, op.code, op_type)
        if not passed:
            errors.append(f"{type_name}/{op.name}: {error}")
    return len(errors) == 0, errors

def generate_operator_code(
    op_type: OperatorType,
    reference: str,
    parent: Optional[JuliaOperator] = None,
    parent2: Optional[JuliaOperator] = None,
    model: str = "openai/gpt-5-mini",
    model_ensemble: Optional[ModelEnsemble] = None,
    mode: str = "explore",
    feedback: str = "",
    variation_seed: int = 0,
    temperature: float = 0.0,
    use_cache: bool = True,
    task_info: Optional[Dict[str, str]] = None,
    log_prompt_dir: Optional[Path] = None,
    log_generation: int = -1,
) -> Tuple[str, str, str]:
    """Generate new Julia operator code using an LLM.

    For task-aware modes, `task_info` should supply:
      - mode="task_refine": {"unsolved_tasks_text": "..."}
      - mode="task_crossover": {"p1_tasks_text": "...", "p2_tasks_text": "..."}

    Returns (code, func_name, selected_model).
    """
    if mode == "explore":
        prompt = op_type.build_explore_prompt(reference, variation_seed)
    elif mode == "task_explore":
        if not hasattr(op_type, "build_task_aware_explore_prompt"):
            raise ValueError(f"task_explore not supported for operator type {op_type.name}")
        if not task_info or "unsolved_tasks_text" not in task_info:
            raise ValueError("task_explore mode requires task_info['unsolved_tasks_text']")
        prompt = op_type.build_task_aware_explore_prompt(
            reference, task_info["unsolved_tasks_text"], variation_seed,
        )
    elif mode == "refine":
        if parent is None:
            raise ValueError("refine mode requires a parent")
        prompt = op_type.build_refine_prompt(parent.code, reference, feedback)
    elif mode == "crossover":
        if parent is None or parent2 is None:
            raise ValueError("crossover mode requires two parents")
        prompt = op_type.build_crossover_prompt(parent.code, parent2.code, reference)
    elif mode == "task_refine":
        if parent is None:
            raise ValueError("task_refine mode requires a parent")
        if not hasattr(op_type, "build_task_aware_refine_prompt"):
            raise ValueError(f"task_refine not supported for operator type {op_type.name}")
        if not task_info or "unsolved_tasks_text" not in task_info:
            raise ValueError("task_refine mode requires task_info['unsolved_tasks_text']")
        prompt = op_type.build_task_aware_refine_prompt(
            parent.code, reference, task_info["unsolved_tasks_text"],
        )
    elif mode == "task_crossover":
        if parent is None or parent2 is None:
            raise ValueError("task_crossover mode requires two parents")
        if not hasattr(op_type, "build_task_aware_crossover_prompt"):
            raise ValueError(f"task_crossover not supported for operator type {op_type.name}")
        if not task_info or "p1_tasks_text" not in task_info or "p2_tasks_text" not in task_info:
            raise ValueError("task_crossover mode requires task_info['p1_tasks_text'] and ['p2_tasks_text']")
        prompt = op_type.build_task_aware_crossover_prompt(
            parent.code, parent2.code, reference,
            task_info["p1_tasks_text"], task_info["p2_tasks_text"],
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Optional generic appendix: execution trace from a recent search using this
    # bundle, plus a brainstorm instruction. Applied after the type-specific
    # prompt so it works uniformly across mutation/survival/selection.
    if task_info and task_info.get("execution_trace_text"):
        prompt = (
            f"{prompt}\n\n"
            f"## Execution trace from a recent search using this bundle\n"
            f"{task_info['execution_trace_text']}\n\n"
            f"{_BRAINSTORM_INSTRUCTION}\n"
        )

    # Use ensemble to pick model if available, otherwise use single model.
    # If the API call fails (even after internal retries), resample a different
    # model from the ensemble and try again a few times before giving up.
    max_model_attempts = 4 if model_ensemble else 1
    tried_models: List[str] = []
    response = None
    selected_model = model_ensemble.sample() if model_ensemble else model
    for model_attempt in range(max_model_attempts):
        tried_models.append(selected_model)
        try:
            response = chat_completion(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                sample_index=variation_seed + model_attempt * 10_000,
                use_cache=use_cache,
            )
            break
        except Exception as e:
            print(f"  chat_completion failed with {selected_model}: {type(e).__name__}: {e}")
            if model_attempt + 1 >= max_model_attempts or not model_ensemble:
                print(f"  Giving up after trying models: {tried_models}")
                raise
            # Sample a different model if possible
            for _ in range(10):
                candidate = model_ensemble.sample()
                if candidate not in tried_models:
                    selected_model = candidate
                    break
            else:
                selected_model = model_ensemble.sample()
            print(f"  Retrying with different model: {selected_model}")

    content = get_content(response)
    code = extract_julia_code(content)
    func_name = extract_function_name(code) if code else ""

    # Log prompt + response + extracted code to disk.
    if log_prompt_dir is not None:
        try:
            log_prompt_dir.mkdir(parents=True, exist_ok=True)
            fname = f"gen{max(log_generation, 0):03d}_{op_type.name}_{mode}_seed{variation_seed}.md"
            header = (
                f"<!-- op_type={op_type.name} mode={mode} "
                f"generation={log_generation} variation_seed={variation_seed} "
                f"model={selected_model} func_name={func_name} -->\n\n"
            )
            body = (
                header
                + "## Prompt\n\n"
                + prompt
                + "\n\n## Raw Response\n\n"
                + (content or "(empty)")
                + "\n\n## Extracted Code\n\n```julia\n"
                + (code or "(no code extracted)")
                + "\n```\n"
            )
            (log_prompt_dir / fname).write_text(body)
        except Exception as e:
            print(f"  [prompt-log] Failed to write prompt: {e}")

    if not code:
        return "", "", selected_model

    return code, func_name, selected_model
