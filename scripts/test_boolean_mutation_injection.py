"""De-risk the evolved-condition path: inject a custom Julia mutation operator
into a Boolean-PySR fit and confirm it runs without error.

Uses the repo's default baseline mutation (add_constant_offset.jl) as a stand-in
for an evolved operator, exercising _load_dynamic_mutations + weight_custom_mutation_1.
"""
import sys
sys.path.insert(0, "/home/sca63/meta_sr")

from boolean_tasks import generate_synthetic_task
from boolean_pysr import run_boolean_pysr
from operator_types import MutationOperatorType


def main():
    op = MutationOperatorType().load_default_baseline_operator()
    assert op is not None, "could not load default baseline mutation"
    print(f"injecting mutation: {op.name}", flush=True)
    code = {op.name: op.code}

    task = generate_synthetic_task("parity3")
    res = run_boolean_pysr(
        task, eval_task=task, custom_mutation_code=code,
        custom_mutation_weight=5.0, pysr_kwargs={"niterations": 20}, seed=0,
    )
    print("train_acc:", res.train_acc, "solved:", res.train_solved, flush=True)
    print("eval_acc:", res.eval_acc, flush=True)
    print("best:", res.best_equation, flush=True)
    print("error:", res.error, flush=True)
    print("INJECTION_OK" if res.error is None else "INJECTION_FAILED", flush=True)


if __name__ == "__main__":
    main()
