"""
LLM-Meta-SR: Meta-evolution framework for evolving selection operators using LLMs
Based on the paper: "LLM-Meta-SR: In-Context Learning for Evolving Selection Operators
in Symbolic Regression"
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import traceback
from dataclasses import dataclass
import hashlib

# LLM imports (you'll need to set API keys)
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class SelectionOperator:
    """Represents a selection operator with its code and performance"""
    code: str
    fitness: float = 0.0
    scores: List[float] = None  # Performance on each dataset
    code_length: int = 0
    generation: int = 0
    id: str = ""

    def __post_init__(self):
        if self.scores is None:
            self.scores = []
        if not self.id:
            self.id = hashlib.md5(self.code.encode()).hexdigest()[:8]
        self.code_length = len([line for line in self.code.split('\n')
                               if line.strip() and not line.strip().startswith('#')])


class LLMMetaSR:
    """Meta-evolution framework for evolving selection operators using LLMs"""

    def __init__(self,
                 llm_provider='openai',  # 'openai' or 'anthropic'
                 model='gpt-4',
                 population_size=20,
                 n_generations=20,
                 n_crossover=19,
                 n_mutation=1,
                 max_code_lines=30,
                 results_dir='results_meta_sr',
                 use_domain_knowledge=True,
                 use_bloat_control=True,
                 use_semantic_selection=True):

        self.llm_provider = llm_provider
        self.model = model
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_crossover = n_crossover
        self.n_mutation = n_mutation
        self.max_code_lines = max_code_lines
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.use_domain_knowledge = use_domain_knowledge
        self.use_bloat_control = use_bloat_control
        self.use_semantic_selection = use_semantic_selection

        # Initialize LLM client
        if llm_provider == 'anthropic':
            if not HAS_ANTHROPIC:
                raise ValueError("Anthropic package not installed")
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.client = Anthropic(api_key=api_key)
        elif llm_provider == 'openai':
            if not HAS_OPENAI:
                raise ValueError("OpenAI package not installed")
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            openai.api_key = api_key
            self.client = None  # Use openai module directly
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")

    def call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call LLM API"""
        if self.llm_provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif self.llm_provider == 'openai':
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4096
            )
            return response.choices[0].message.content

    def extract_code(self, llm_response: str) -> str:
        """Extract Python code from LLM response"""
        # Try to find code between ```python and ```
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, llm_response, re.DOTALL)
        if matches:
            return matches[0]

        # Try to find code between ``` and ```
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, llm_response, re.DOTALL)
        if matches:
            return matches[0]

        # Return the whole response if no code blocks found
        return llm_response

    def get_system_prompt(self) -> str:
        """Get system prompt for code generation"""
        return f"""When writing code, prefer NumPy vectorized operations and avoid explicit Python for-loops unless absolutely necessary. Please implement code within {self.max_code_lines} lines."""

    def get_domain_knowledge_properties(self) -> str:
        """Get domain knowledge properties for selection operators"""
        if not self.use_domain_knowledge:
            return ""

        return """
1. Diverse & Specialized Selection
   - Choose a varied set of high-performing individuals.
   - Encourage specialization to maintain a diverse population.

2. Crossover-Aware Pairing
   - Promote complementarity between parents.

3. Stage-Specific Pressure
   - Vary selection pressure based on the current stage of evolution.

4. Interpretability
   - Prefer individuals with fewer nodes and lower tree height to improve model interpretability.

5. Efficiency & Scalability
   - Include clear stopping conditions to avoid infinite loops.

6. Code Simplicity
   - Favor clear, concise logic with minimal complexity.
"""

    def get_template(self) -> str:
        """Get function template for selection operator"""
        if self.use_domain_knowledge:
            template = '''def omni_selection(population, k=100, status={}):
    # Useful information about individuals:
    # squared_error_vector = individual.case_values
    # predicted_values = individual.predicted_values
    # residual = individual.y - individual.predicted_values
    # number_of_nodes = len(individual)
    # height = individual.height

    # Useful information about evolution:
    # status["evolutionary_stage"]: [0,1], where 0 is the first generation and 1 is the final generation

    parent_a = Select k//2 individuals based on performance over subsets of instances:
    - evaluate individuals on k//2 different random or structured subsets of the data
    - reward those that specialize or perform well on those subsets
    - may also consider overall fitness (e.g., full-dataset MSE)
    - optionally include structural complexity
    - an individual can be selected multiple times

    parent_b = For each parent_a, select a complementary individual:
    - low residual correlation with parent_a
    - may also consider complexity

    # Interleave parent_a and parent_b to form crossover pairs
    selected_individuals = [individual for pair in zip(parent_a, parent_b) for individual in pair]
    return selected_individuals'''
        else:
            template = '''def omni_selection(population, k=100, status={}):
    # Useful information about individuals:
    # squared_error_vector = individual.case_values
    # predicted_values = individual.predicted_values
    # residual = individual.y - individual.predicted_values
    # number_of_nodes = len(individual)
    # height = individual.height

    # Useful information about evolution:
    # status["evolutionary_stage"]: [0,1], where 0 is the first generation and 1 is the final generation

    # Implement selection logic here
    return selected_individuals'''

        return template

    def generate_initialization_prompt(self) -> str:
        """Generate prompt for initializing population"""
        system_prompt = self.get_system_prompt()
        properties = self.get_domain_knowledge_properties()
        template = self.get_template()

        prompt = f"""{system_prompt}

Your task is to develop an innovative and novel selection operator for symbolic regression using genetic programming in Python.

{properties if properties else ""}

Ensure that your newly designed function adheres to the following signature:
{template}

You do not need to provide a usage example.

Embrace creativity, novelty, and bold experimentation to push the boundaries of the state of the art in selection operators for genetic programming."""

        return prompt

    def generate_mutation_prompt(self, elite_operator: SelectionOperator) -> str:
        """Generate prompt for mutation"""
        system_prompt = self.get_system_prompt()
        properties = self.get_domain_knowledge_properties()
        template = self.get_template()

        prompt = f"""{system_prompt}

Your task is to develop an innovative and novel selection operator for symbolic regression using genetic programming in Python.

Inspirational Example:
```python
{elite_operator.code}
```

Use this as inspiration to create a distinctly original and inventive selection operator.

{properties if properties else ""}

Ensure that your newly designed function adheres to the following signature:
{template}

You do not need to provide a usage example.

Embrace creativity, novelty, and bold experimentation to push the boundaries of the state of the art in selection operators for genetic programming."""

        return prompt

    def generate_crossover_prompt(self, parent1: SelectionOperator, parent2: SelectionOperator) -> str:
        """Generate prompt for crossover"""
        system_prompt = self.get_system_prompt()
        properties = self.get_domain_knowledge_properties()
        template = self.get_template()

        prompt = f"""{system_prompt}

You are tasked with designing a novel selection operator for symbolic regression using genetic programming.

Your goal is to synthesize a new operator that combines the strengths and mitigates the weaknesses of the two operators shown below:

— Operator A —
Score (Higher is Better): {parent1.fitness:.3f}, Lines of Code: {parent1.code_length}
Performance on datasets: {parent1.scores}

Code:
```python
{parent1.code}
```

— Operator B —
Score (Higher is Better): {parent2.fitness:.3f}, Lines of Code: {parent2.code_length}
Performance on datasets: {parent2.scores}

Code:
```python
{parent2.code}
```

{properties if properties else ""}

Ensure that your newly designed function adheres to the following signature:
{template}

You do not need to provide a usage example."""

        return prompt

    def validate_code(self, code: str) -> bool:
        """Validate that code can be executed"""
        try:
            # Try to compile the code
            compile(code, '<string>', 'exec')
            return True
        except Exception as e:
            print(f"Code validation failed: {e}")
            return False

    def semantic_selection(self, population: List[SelectionOperator]) -> Tuple[SelectionOperator, SelectionOperator]:
        """Select two complementary operators based on semantic awareness"""
        if not self.use_semantic_selection or len(population) < 2:
            # Random selection
            import random
            return random.choice(population), random.choice(population)

        # Select first parent randomly
        import random
        parent_a = random.choice(population)

        # Select second parent based on complementarity
        best_comp_score = -np.inf
        parent_b = None

        for candidate in population:
            if candidate.id == parent_a.id:
                continue

            # Compute complementarity score: max performance across all datasets
            scores_a = np.array(parent_a.scores)
            scores_b = np.array(candidate.scores)
            comp_score = np.mean(np.maximum(scores_a, scores_b))

            if comp_score > best_comp_score:
                best_comp_score = comp_score
                parent_b = candidate

        if parent_b is None:
            parent_b = random.choice([p for p in population if p.id != parent_a.id])

        return parent_a, parent_b

    def bloat_control_selection(self, population: List[SelectionOperator],
                                offspring: List[SelectionOperator]) -> List[SelectionOperator]:
        """Select survivors using multi-objective selection based on fitness and code length"""
        if not self.use_bloat_control:
            # Simple elitism: keep best by fitness
            combined = population + offspring
            combined.sort(key=lambda x: x.fitness, reverse=True)
            return combined[:self.population_size]

        combined = population + offspring

        # Compute dominance scores
        dominance_scores = []
        for i, op_i in enumerate(combined):
            score = 0
            for j, op_j in enumerate(combined):
                if i == j:
                    continue
                # op_i dominates op_j if better fitness and shorter code
                if op_i.fitness >= op_j.fitness and op_i.code_length <= op_j.code_length:
                    # Compute code similarity (simplified version)
                    similarity = 1.0 if op_i.code == op_j.code else 0.5
                    score -= similarity
            dominance_scores.append(score)

        # Select top N by dominance score
        indices = np.argsort(dominance_scores)[::-1][:self.population_size]
        return [combined[i] for i in indices]

    def generate_fallback_operator(self) -> str:
        """Generate fallback tournament selection operator"""
        return '''import numpy as np
import random

def omni_selection(population, k=100, status={}, tour_size=3):
    mean_errors = [np.mean(ind.case_values) for ind in population]
    selected = []
    for _ in range(k):
        competitors = random.sample(range(len(population)), min(tour_size, len(population)))
        best_idx = min(competitors, key=lambda idx: mean_errors[idx])
        selected.append(population[best_idx])
    return selected'''

    def save_checkpoint(self, generation: int, population: List[SelectionOperator]):
        """Save checkpoint of current population"""
        checkpoint = {
            'generation': generation,
            'population': [
                {
                    'code': op.code,
                    'fitness': op.fitness,
                    'scores': op.scores,
                    'code_length': op.code_length,
                    'id': op.id
                }
                for op in population
            ]
        }

        checkpoint_file = self.results_dir / f'checkpoint_gen_{generation}.json'
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def run(self, train_datasets: List[str], n_datasets: int = 4):
        """
        Run meta-evolution to evolve selection operators

        Args:
            train_datasets: List of dataset paths to use for training
            n_datasets: Number of datasets to use (default: 4)
        """
        print("="*80)
        print("LLM-Meta-SR: Meta-Evolution for Selection Operator Design")
        print("="*80)
        print(f"LLM: {self.llm_provider} - {self.model}")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.n_generations}")
        print(f"Domain knowledge: {self.use_domain_knowledge}")
        print(f"Bloat control: {self.use_bloat_control}")
        print(f"Semantic selection: {self.use_semantic_selection}")
        print("="*80)

        # Select training datasets
        train_datasets = train_datasets[:n_datasets]
        print(f"\nTraining on {len(train_datasets)} datasets")

        # Initialize population
        print("\nInitializing population...")
        population = []
        for i in range(self.population_size):
            print(f"Generating operator {i+1}/{self.population_size}...")
            prompt = self.generate_initialization_prompt()
            response = self.call_llm(prompt)
            code = self.extract_code(response)

            if not self.validate_code(code):
                print(f"  Invalid code generated, using fallback")
                code = self.generate_fallback_operator()

            operator = SelectionOperator(code=code, generation=0)
            population.append(operator)
            print(f"  Generated operator with {operator.code_length} lines")

        print(f"\nInitialized {len(population)} operators")

        # TODO: Evaluate initial population on train datasets
        # This requires integrating with the symbolic regression loop

        # Save initial population
        self.save_checkpoint(0, population)

        print("\nMeta-evolution complete!")
        print(f"Results saved to: {self.results_dir}")

        return population


if __name__ == "__main__":
    # Example usage
    meta_sr = LLMMetaSR(
        llm_provider='openai',
        model='gpt-4',
        population_size=5,  # Small for testing
        n_generations=3,
        results_dir='results_meta_sr_test'
    )

    # You would normally provide actual dataset paths
    train_datasets = ['dataset1', 'dataset2', 'dataset3', 'dataset4']

    # Note: This won't fully work without API keys and actual evaluation
    # population = meta_sr.run(train_datasets)

    print("LLM-Meta-SR framework implemented!")
    print("To use:")
    print("1. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
    print("2. Provide training datasets")
    print("3. Run meta_sr.run(train_datasets)")
