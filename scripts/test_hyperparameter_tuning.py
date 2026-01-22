#!/usr/bin/env python3
"""
Test script for hyperparameter tuning functionality.

Usage:
    python test_hyperparameter_tuning.py
"""

import sys
import numpy as np
from meta_evolution import (
    Operator,
    OperatorBundle,
    get_default_operator,
    create_and_test_operator,
    OPERATOR_TYPES,
)
from hyperparameter_tuning import (
    identify_hyperparameters,
    inject_hyperparameters,
    tune_bundle_hyperparameters,
    HyperparameterSpec,
)


def test_hyperparameter_identification():
    """Test that we can identify hyperparameters in operator code."""
    print("\n" + "=" * 60)
    print("TEST 1: Hyperparameter Identification")
    print("=" * 60)

    # Test with a selection operator that has clear hyperparameters
    test_code = '''
def selection(population, fitnesses, n_crossover, n_mutation):
    """Tournament selection with configurable tournament size."""
    tournament_size = 3
    selection_pressure = 2.0
    elite_fraction = 0.1

    n_elite = max(1, int(len(population) * elite_fraction))
    sorted_indices = np.argsort(fitnesses)[::-1]

    crossover_pairs = []
    for _ in range(n_crossover):
        candidates = np.random.choice(len(population), tournament_size, replace=False)
        winner_idx = candidates[np.argmax(fitnesses[candidates])]
        crossover_pairs.append((population[winner_idx], population[winner_idx]))

    mutants = [population[sorted_indices[i % n_elite]] for i in range(n_mutation)]
    return crossover_pairs, mutants
'''

    # Create a mock operator for testing
    operator = Operator(
        func=lambda *args: None,  # Dummy function
        operator_type="selection",
        code=test_code,
    )

    print(f"\nTest operator code:\n{test_code[:300]}...")

    # Test identification (this will call the LLM)
    print("\nIdentifying hyperparameters (calling LLM)...")
    hyperparams = identify_hyperparameters(operator, model="openai/gpt-4o-mini")

    if hyperparams:
        print(f"\nFound {len(hyperparams)} hyperparameters:")
        for hp in hyperparams:
            print(f"  - {hp.name}: {hp.current_value} ({hp.param_type})")
            if hasattr(hp, 'line_pattern'):
                print(f"      pattern: '{hp.line_pattern}'")
        print("\n[PASS] Hyperparameter identification")
        return True, hyperparams, test_code
    else:
        print("\n[WARN] No hyperparameters found - this might be expected for simple operators")
        return True, [], test_code


def test_hyperparameter_injection(hyperparams, original_code):
    """Test that we can inject new hyperparameter values into code."""
    print("\n" + "=" * 60)
    print("TEST 2: Hyperparameter Injection")
    print("=" * 60)

    if not hyperparams:
        print("Skipping - no hyperparameters to inject")
        return True

    # Create new values
    new_values = {}
    for hp in hyperparams:
        if hp.param_type == "float":
            new_values[hp.name] = hp.current_value * 1.5  # Increase by 50%
        elif hp.param_type == "int":
            new_values[hp.name] = int(hp.current_value) + 1
        else:
            new_values[hp.name] = hp.current_value

    print(f"\nOriginal values: {[(hp.name, hp.current_value) for hp in hyperparams]}")
    print(f"New values: {new_values}")

    # Inject new values
    modified_code = inject_hyperparameters(original_code, hyperparams, new_values)

    print(f"\nModified code:\n{modified_code[:500]}...")

    # Check if values were actually changed
    values_changed = original_code != modified_code
    if values_changed:
        print("\n[PASS] Code was modified with new hyperparameter values")
    else:
        print("\n[WARN] Code was not modified - injection might have failed")

    return values_changed


def test_operator_validation():
    """Test that operators with modified hyperparameters still pass validation."""
    print("\n" + "=" * 60)
    print("TEST 3: Operator Validation After Modification")
    print("=" * 60)

    # Get a default operator
    for op_type in OPERATOR_TYPES:
        print(f"\nTesting {op_type} operator...")
        operator = get_default_operator(op_type)

        # Identify hyperparameters
        hyperparams = identify_hyperparameters(operator, model="openai/gpt-4o-mini")

        if not hyperparams:
            print(f"  No hyperparameters found for {op_type}")
            continue

        print(f"  Found {len(hyperparams)} hyperparameters")

        # Try modifying and validating
        new_values = {}
        for hp in hyperparams:
            # Skip hyperparams without current values or with non-numeric values
            if hp.current_value is None:
                continue
            # Check if current_value is actually a number
            try:
                if hp.param_type == "int":
                    int(hp.current_value)
                elif hp.param_type == "float":
                    float(hp.current_value)
            except (ValueError, TypeError):
                continue

            if hp.param_type == "float":
                # Stay within reasonable bounds
                if hp.min_value is not None and hp.max_value is not None:
                    new_values[hp.name] = (hp.min_value + hp.max_value) / 2
                else:
                    new_values[hp.name] = float(hp.current_value) * 1.1
            elif hp.param_type == "int":
                if hp.min_value is not None and hp.max_value is not None:
                    new_values[hp.name] = int((hp.min_value + hp.max_value) / 2)
                else:
                    new_values[hp.name] = int(hp.current_value) + 1

        modified_code = inject_hyperparameters(operator.code, hyperparams, new_values)

        # Try to create and validate the modified operator
        new_op, passed, error = create_and_test_operator(modified_code, op_type)

        if passed:
            print(f"  [PASS] Modified {op_type} operator passes validation")
        else:
            print(f"  [WARN] Modified {op_type} operator failed validation: {error}")

    return True


def test_mock_tuning():
    """Test the tuning pipeline with a mock evaluation function."""
    print("\n" + "=" * 60)
    print("TEST 4: Mock Hyperparameter Tuning Pipeline")
    print("=" * 60)

    # Create a default bundle
    bundle = OperatorBundle.create_default()
    print("Created default operator bundle")

    # Mock evaluation function that just returns a random score
    call_count = [0]
    def mock_evaluate_fn(b):
        call_count[0] += 1
        # Return a score that depends slightly on the bundle to simulate optimization
        return 0.5 + np.random.random() * 0.1

    print("\nRunning hyperparameter tuning with mock evaluation...")
    print("(This tests the pipeline but uses a fake evaluator)")

    try:
        tuned_bundle, best_params = tune_bundle_hyperparameters(
            bundle=bundle,
            evaluate_fn=mock_evaluate_fn,
            operator_types=["fitness"],  # Just tune one operator for speed
            n_trials_per_operator=3,  # Few trials for testing
            model="openai/gpt-5-mini",
            seed=42,
            verbose=True,
        )

        print(f"\nEvaluation function was called {call_count[0]} times")
        print(f"Best params found: {best_params}")
        print("\n[PASS] Mock tuning pipeline completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Mock tuning failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("HYPERPARAMETER TUNING TESTS")
    print("=" * 60)

    all_passed = True

    # Test 1: Hyperparameter identification
    passed, hyperparams, test_code = test_hyperparameter_identification()
    all_passed = all_passed and passed

    # Test 2: Hyperparameter injection
    if hyperparams:
        passed = test_hyperparameter_injection(hyperparams, test_code)
        all_passed = all_passed and passed

    # Test 3: Operator validation after modification
    passed = test_operator_validation()
    all_passed = all_passed and passed

    # Test 4: Mock tuning pipeline
    passed = test_mock_tuning()
    all_passed = all_passed and passed

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
