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

        loss = self.operators.get("loss")
        if loss is not None:
            config_kwargs["custom_loss_code"] = loss.code

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
        for t in ["mutation", "survival", "selection", "loss"]:
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
        for t in ["mutation", "survival", "selection", "loss"]:
            op = self.operators.get(t)
            parts.append(op.name if op else "default")
        return " | ".join(parts)



class OperatorType(ABC):
    """Base class defining operator-type-specific behavior.

    All four prompt modes (explore, refine, simplify, crossover) are built on
    this base class. They are parameterized by `self.name` plus two subclass
    hooks that fill in the type-specific bits:

      `_extra_requirements()` — optional type-specific requirement bullets that
                               apply across ALL prompt modes (explore, refine,
                               simplify, crossover). Default: empty list. Loss
                               uses this for the "DO NOT penalize complexity"
                               constraint that holds regardless of mode.

      `_explore_extras()`     — optional explore-only signature/note variants
                               (default all empty); mutation uses this for its
                               data-aware vs structural toggle, which changes
                               the function's arity.

    Subclasses end up small: they fill in Julia validation config and (only
    when needed) the constraint and explore hooks. The LLM picks up the
    operator's role from the reference doc loaded by `load_reference()`.
    """

    name: str  # "mutation", "survival", "selection", "loss"

    # Julia validation config
    julia_module: str
    load_func: str
    clear_func: str
    list_func: str
    smoke_test_julia: str = ""  # Julia code template for runtime smoke test

    # Relative path (under SymbolicRegression.jl/src/) to a .jl file containing
    # this operator type's default baseline implementation — behavior-identical
    # to PySR's built-in default, exposed as a named custom operator so the
    # meta-evolution loop has a concrete parent to refine from. Subclasses set
    # this; loaded lazily by `load_default_baseline_operator()`.
    default_baseline_rel_path: Optional[str] = None

    @abstractmethod
    def load_reference(self) -> str:
        """Load the reference documentation for this operator type."""

    def _extra_requirements(self) -> List[str]:
        """Optional type-specific requirement bullets shared by all prompt modes.

        Returned bullets are prepended to the requirements list in every prompt
        (explore, refine, simplify, crossover). Use this for hard constraints
        that are invariant across modes — e.g., the loss operator's "do not
        penalize complexity" rule. Default: empty.
        """
        return []

    def _explore_extras(self, variation_seed: int) -> Tuple[str, List[str], str]:
        """Optional explore-only extras for per-type signature/note variants.

        Returns (intro_tail, extra_reqs, example_section):
            - `intro_tail`: empty, OR text spliced in after the meta-intro
              (must include any leading "\\n\\n" needed to separate from preceding text)
            - `extra_reqs`: additional MUST/Required bullets prepended to the
              requirements list (before shared core)
            - `example_section`: empty, OR formatted example block

        Default: all empty. Mutation overrides to provide its data-aware vs
        structural toggle (which changes the function's arity).
        """
        return "", [], ""

    @abstractmethod
    def to_pysr_config(self, operator: JuliaOperator, pysr_kwargs: Dict) -> PySRConfig:
        """Convert an operator to a PySRConfig for evaluation."""

    @abstractmethod
    def baseline_config(self, pysr_kwargs: Dict) -> PySRConfig:
        """Create a baseline PySRConfig (no custom operator)."""

    def build_explore_prompt(self, reference: str, variation_seed: int = 0) -> str:
        intro_tail, explore_extra_reqs, example_section = self._explore_extras(variation_seed)
        return (
            "You are an expert in symbolic regression, physics, and genetic programming.\n\n"
            f"Your task is to create a NEW custom {self.name} operator for PySR/SymbolicRegression.jl.\n"
            "Your proposal is being considered as part of a meta-evolutionary loop that samples\n"
            "and evaluates many proposed improvements to the PySR algorithm, so be creative in your proposal.\n"
            "Our goal is to improve the PySR symbolic regression algorithm to maximize the percent of tasks\n"
            "for which PySR discovers the correct ground truth expression.\n"
            "Example equations from this dataset include 0.5 sin(x - y) - sin(x) or q/(4*pi*epsilon*r*(1-v/c)).\n"
            f"{intro_tail}\n\n"
            f"## Reference: relevant API\n{reference}\n\n"
            "## Requirements\n"
            + "\n".join(f"{i}. {req}" for i, req in enumerate([
                *self._extra_requirements(),
                *explore_extra_reqs,
                "Use proper Julia syntax",
                "Include a Julia docstring (`\"\"\"...\"\"\"`) immediately above the `function` line explaining its core idea, the steps it takes, and any heuristics or assumptions.",
                "Use inline comments as appropriate to explain the implementation of the function body.",
            ], start=1))
            + "\n\n"
            "## Output Format\n"
            "Return ONLY the new Julia function code (with the docstring above it and inline comments inside it), nothing else.\n"
            "The function should be named descriptively.\n"
            "Do not include markdown code blocks or prose outside the docstring.\n"
            + example_section
        )

    def build_refine_prompt(self, parent: JuliaOperator, reference: str) -> str:
        return (
            "You are an expert in symbolic regression, physics, and genetic programming.\n\n"
            f"Your task is to IMPROVE an existing custom {self.name} operator for PySR/SymbolicRegression.jl.\n"
            "Your proposal is being considered as part of a meta-evolutionary loop that samples\n"
            "and evaluates many proposed improvements to the PySR algorithm.\n"
            "Our goal is to improve the PySR symbolic regression algorithm to maximize the percent of tasks\n"
            "for which PySR discovers the correct ground truth expression.\n"
            "Example equations from this dataset include 0.5 sin(x - y) - sin(x) or q/(4*pi*epsilon*r*(1-v/c)).\n\n"
            f"## Parent {self.name} operator code\n```julia\n{parent.code}\n```\n\n"
            f"## Reference: relevant API\n{reference}\n\n"
            "## Requirements\n"
            + "\n".join(f"{i}. {req}" for i, req in enumerate([
                *self._extra_requirements(),
                "Keep the core idea but improve the implementation, or generate a variant that improves on the parent.",
                "Use proper Julia syntax",
                "Include a Julia docstring (`\"\"\"...\"\"\"`) immediately above the `function` line explaining the operator's core idea, steps, and heuristics, plus what changed vs. the parent.",
                "Use inline comments as appropriate to explain the implementation of the function body.",
            ], start=1))
            + "\n\n"
            "## Output Format\n"
            "Return ONLY the new Julia function code (with the docstring above it and inline comments inside it), nothing else.\n"
            "The function should be named descriptively.\n"
            "Do not include markdown code blocks or prose outside the docstring.\n"
        )

    def build_simplify_prompt(self, parent_code: str, reference: str) -> str:
        return (
            "You are an expert in symbolic regression, physics, and genetic programming.\n\n"
            f"Your task is to SIMPLIFY an existing custom {self.name} operator for PySR/SymbolicRegression.jl.\n"
            "Produce a streamlined version of the parent that keeps its core idea but removes complexity.\n"
            "For example, you might drop redundant branches, fold special cases into the common path,\n"
            "or trim heuristics. If the parent combines many factors (e.g. five distinct heuristics),\n"
            "you might keep only the most important three or four. The goal is to maintain performance\n"
            "(as measured by percent of tasks for which PySR discovers the correct ground truth expression)\n"
            "while simplifying the operator.\n"
            "Example equations from this dataset include 0.5 sin(x - y) - sin(x) or q/(4*pi*epsilon*r*(1-v/c)).\n\n"
            f"## Parent {self.name} operator code\n```julia\n{parent_code}\n```\n\n"
            f"## Reference: relevant API\n{reference}\n\n"
            "## Requirements\n"
            + "\n".join(f"{i}. {req}" for i, req in enumerate([
                *self._extra_requirements(),
                "Keep the same function signature shape as the parent",
                "Use proper Julia syntax",
                "Include a Julia docstring (`\"\"\"...\"\"\"`) immediately above the `function` line explaining the simplified operator's core idea, the steps it takes, and what was removed/merged from the parent (and why the simplification should still be sound).",
                "Use inline comments as appropriate to explain the implementation of the function body.",
            ], start=1))
            + "\n\n"
            "## Output Format\n"
            "Return ONLY the new Julia function code (with the docstring above it and inline comments inside it), nothing else.\n"
            "The function should be named descriptively.\n"
            "Do not include markdown code blocks or prose outside the docstring.\n"
        )

    def build_crossover_prompt(self, p1_code: str, p2_code: str, reference: str) -> str:
        return (
            "You are an expert in symbolic regression, physics, and genetic programming.\n\n"
            f"Your task is to COMBINE ideas from two {self.name} operators into a new one.\n"
            "Your proposal is being considered as part of a meta-evolutionary loop that samples\n"
            "and evaluates many proposed improvements to the PySR algorithm.\n"
            "Our goal is to improve the PySR symbolic regression algorithm to maximize the percent of tasks\n"
            "for which PySR discovers the correct ground truth expression.\n"
            "Example equations from this dataset include 0.5 sin(x - y) - sin(x) or q/(4*pi*epsilon*r*(1-v/c)).\n\n"
            f"## Parent {self.name} operator 1\n```julia\n{p1_code}\n```\n\n"
            f"## Parent {self.name} operator 2\n```julia\n{p2_code}\n```\n\n"
            f"## Reference: relevant API\n{reference}\n\n"
            "## Requirements\n"
            + "\n".join(f"{i}. {req}" for i, req in enumerate([
                *self._extra_requirements(),
                f"Create a NEW {self.name} operator that combines the best ideas from both parents",
                "Don't just concatenate — synthesize a coherent new approach",
                "Use proper Julia syntax",
                "Include a Julia docstring (`\"\"\"...\"\"\"`) immediately above the `function` line explaining the core idea synthesized from the parents, the steps it takes, and any heuristics or assumptions.",
                "Use inline comments as appropriate to explain the implementation of the function body.",
            ], start=1))
            + "\n\n"
            "## Output Format\n"
            "Return ONLY the new Julia function code (with the docstring above it and inline comments inside it), nothing else.\n"
            "The function should be named descriptively.\n"
            "Do not include markdown code blocks or prose outside the docstring.\n"
        )

    def load_default_baseline_operator(self) -> Optional[JuliaOperator]:
        """Return a JuliaOperator for this type's default PySR baseline.

        Reads `default_baseline_rel_path` from disk. Returns None if not set or
        the file is missing. The operator's `name` comes from the file's
        `function <name>(` definition; its code is the full file contents.
        Mutation operators receive a default weight of 0.5 (survival/selection
        don't use the weight field).
        """
        if not self.default_baseline_rel_path:
            return None
        path = (
            Path(__file__).resolve().parent
            / "SymbolicRegression.jl" / "src" / self.default_baseline_rel_path
        )
        if not path.exists():
            return None
        code = path.read_text()
        func_name = extract_function_name(code)
        if not func_name:
            return None
        weight = 0.5 if self.name == "mutation" else None
        return JuliaOperator(name=func_name, code=code, weight=weight)

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
    default_baseline_rel_path = "custom_mutations/add_constant_offset.jl"
    smoke_test_julia = """
    let
        using SymbolicRegression: Options, Node, AbstractExpressionNode, Dataset
        using SymbolicRegression.CustomMutationsModule: apply_custom_mutation
        using Random: Xoshiro
        options = Options(;
            binary_operators=[+, -, *, /],
            unary_operators=[sin, cos],
        )
        # Build a small tree: x1 + 0.5
        tree = Node(Float64; op=1, l=Node(Float64; feature=1), r=Node(Float64; val=0.5))
        # Small synthetic dataset so data-aware mutations have real X/y to read
        X = randn(Float64, 3, 10)
        y = randn(Float64, 10)
        dataset = Dataset(X, y)
        rng = Xoshiro(42)
        result = apply_custom_mutation(:{name}, tree, dataset, options, 3, rng)
        @assert result isa AbstractExpressionNode "Smoke test: mutation must return a Node, got $(typeof(result))"
    end
    """

    def load_reference(self) -> str:
        base = Path(__file__).resolve().parent / "SymbolicRegression.jl/src/custom_mutations"
        ref_path = base / "MUTATIONS_REFERENCE.md"
        if ref_path.exists():
            return ref_path.read_text()
        raise FileNotFoundError(f"Could not find MUTATIONS_REFERENCE.md")

    # Mutation has a per-explore data-aware vs structural toggle (chosen by
    # variation_seed parity) — the function signature itself differs (5-arg
    # with `dataset` vs 4-arg without), so the example template below changes
    # accordingly. Refine doesn't toggle: it inherits the parent's signature.
    def _explore_extras(self, variation_seed: int) -> Tuple[str, List[str], str]:
        data_aware = (variation_seed % 2 == 0)
        if data_aware:
            mode_note = (
                "For this proposal, write a **data-aware mutation**: consult `dataset.X` and/or `dataset.y`\n"
                "to make a data-driven decision (e.g. insert the feature most correlated with the residual,\n"
                "fit a constant by least squares, detect subtrees that produce NaN/Inf on the training data)."
            )
            signature_note = (
                "Use the **5-argument data-aware signature** with `dataset` included\n"
                "(the reference doc shows this form). Do NOT use the 4-argument form."
            )
            extra_req = (
                "Use `dataset.X` / `dataset.y` — this proposal is specifically a data-aware mutation."
            )
            example_signature = """function my_mutation_name(
    tree::N,
    dataset,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Implementation — may read dataset.X :: (nfeatures, n) and dataset.y :: (n,)
    return tree
end"""
        else:
            mode_note = (
                "For this proposal, write a **structural mutation**: operate purely on the tree (nodes,\n"
                "operators, constants, variables). Do not take a `dataset` argument."
            )
            signature_note = (
                "**Important signature override:** the reference doc shows a 5-argument signature with `dataset`.\n"
                "For this structural proposal, use the **4-argument form without `dataset`** shown below.\n"
                "The runtime adapts automatically based on the arity of your function."
            )
            extra_req = (
                "Your function MUST use the 4-argument signature below (no `dataset` parameter)."
            )
            example_signature = """function my_mutation_name(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Implementation — operates on tree structure only
    return tree
end"""

        intro_tail = f"\n\n{mode_note}\n\n{signature_note}"
        example_section = (
            "\nExample format (use this exact signature, with a docstring above and inline comments inside):\n"
            f'"""\n    my_{self.name}_name(tree, ...)\n\n'
            f"One- or two-paragraph explanation of how this {self.name} operator works.\n"
            f'"""\n'
            f"{example_signature}\n"
        )
        return intro_tail, [extra_req], example_section

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
    default_baseline_rel_path = "custom_survival/age_regularized_survival.jl"
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
    default_baseline_rel_path = "custom_selection/tournament_selection.jl"
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

class LossOperatorType(OperatorType):
    name = "loss"
    julia_module = "CustomLossModule"
    load_func = "load_loss_from_string!"
    clear_func = "clear_dynamic_losses!"
    list_func = "list_available_losses"
    default_baseline_rel_path = "custom_loss/mse_loss.jl"
    smoke_test_julia = """
    let
        using SymbolicRegression: Options, Node, Dataset, eval_loss
        using SymbolicRegression.CustomLossModule: apply_custom_loss
        options = Options(;
            binary_operators=[+, -, *, /],
            unary_operators=[sin, cos],
        )
        X = randn(Float64, 3, 30)
        y = X[1, :] .+ 0.5 .* X[2, :]
        dataset = Dataset(X, y)
        tree = Node(Float64; op=1, l=Node(Float64; feature=1), r=Node(Float64; val=0.5))
        # Direct dispatch (covers the eval_loss hook too).
        loss_val = apply_custom_loss(tree, dataset, options)
        @assert loss_val isa AbstractFloat "Smoke test: loss must return AbstractFloat, got $(typeof(loss_val))"
        @assert isfinite(loss_val) || isinf(loss_val) "Smoke test: loss must be finite or Inf, got $(loss_val)"
        # Confirm it also flows through eval_loss (the production code path).
        loss_via_eval = eval_loss(tree, dataset, options)
        @assert loss_via_eval isa AbstractFloat
    end
    """

    def load_reference(self) -> str:
        ref_path = Path(__file__).resolve().parent / "SymbolicRegression.jl/src/custom_loss/LOSS_REFERENCE.md"
        if ref_path.exists():
            return ref_path.read_text()
        raise FileNotFoundError(f"Could not find LOSS_REFERENCE.md at {ref_path}")

    # The constraint "DO NOT penalize complexity" must hold across explore,
    # refine, simplify, and crossover, so it lives in `_extra_requirements`
    # (which is consumed by all four prompt builders) rather than the
    # explore-only `_explore_extras` hook.
    def _extra_requirements(self) -> List[str]:
        return [
            "DO NOT penalize raw expression complexity in your loss. PySR already tracks "
            "a Pareto frontier of (loss, complexity) and adds a separate parsimony term to "
            "the cost. A complexity penalty inside the loss double-counts size and distorts "
            "the Pareto frontier.",
        ]

    def to_pysr_config(self, operator: JuliaOperator, pysr_kwargs: Dict) -> PySRConfig:
        mutation_weights = get_default_mutation_weights()
        return PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs=pysr_kwargs,
            custom_loss_code=operator.code,
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
    "loss": LossOperatorType(),
}

def validate_julia_code(name: str, code: str, op_type: OperatorType) -> Tuple[bool, str]:
    """Validate Julia operator code by attempting to load it and smoke-testing it."""
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
    for type_name in ("mutation", "survival", "selection", "loss"):
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
    variation_seed: int = 0,
    temperature: float = 0.0,
    use_cache: bool = True,
    task_info: Optional[Dict[str, str]] = None,
    log_prompt_dir: Optional[Path] = None,
    log_generation: int = -1,
) -> Tuple[str, str, str]:
    """Generate new Julia operator code using an LLM.

    Returns (code, func_name, selected_model).
    """
    if mode == "explore":
        prompt = op_type.build_explore_prompt(reference, variation_seed)
    elif mode == "refine":
        if parent is None:
            raise ValueError("refine mode requires a parent")
        prompt = op_type.build_refine_prompt(parent, reference)
    elif mode == "crossover":
        if parent is None or parent2 is None:
            raise ValueError("crossover mode requires two parents")
        prompt = op_type.build_crossover_prompt(parent.code, parent2.code, reference)
    elif mode == "simplify":
        if parent is None:
            raise ValueError("simplify mode requires a parent")
        prompt = op_type.build_simplify_prompt(parent.code, reference)
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
                max_tokens=128_000,
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
