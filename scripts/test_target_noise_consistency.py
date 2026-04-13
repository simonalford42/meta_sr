#!/usr/bin/env python3
"""Test that target noise map logic is identical across evolve_pysr, hpo_pysr, and openevolve_pysr."""

import hashlib
import sys
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _reference_stable_target_noise(dataset_name, seed, noise_levels):
    """Reference implementation from evolve_pysr.py."""
    digest = hashlib.sha256(f"{seed}:{dataset_name}".encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "little") % len(noise_levels)
    return noise_levels[idx]


def _reference_build_map(dataset_names, seed, noise_levels):
    return {name: _reference_stable_target_noise(name, seed, noise_levels) for name in dataset_names}


NOISE_LEVELS = [0.0, 0.001, 0.01, 0.1]
DATASETS = ["192_vineyard", "210_cloud", "503_wind", "529_pollen", "1027_ESL", "feynman_I_6_20a"]
SEED = 42


def test_evolve_pysr_matches_reference():
    from evolve_pysr import _stable_target_noise, _build_target_noise_map, TARGET_NOISE_LEVELS

    assert TARGET_NOISE_LEVELS == NOISE_LEVELS

    ref_map = _reference_build_map(DATASETS, SEED, NOISE_LEVELS)
    actual_map = _build_target_noise_map(DATASETS, SEED, TARGET_NOISE_LEVELS)
    assert actual_map == ref_map, f"evolve_pysr mismatch: {actual_map} != {ref_map}"

    for ds in DATASETS:
        assert _stable_target_noise(ds, SEED, NOISE_LEVELS) == ref_map[ds]


def test_hpo_pysr_matches_reference():
    from hpo_pysr import _stable_target_noise, _build_target_noise_map, TARGET_NOISE_LEVELS

    assert TARGET_NOISE_LEVELS == NOISE_LEVELS

    ref_map = _reference_build_map(DATASETS, SEED, NOISE_LEVELS)
    actual_map = _build_target_noise_map(DATASETS, SEED, TARGET_NOISE_LEVELS)
    assert actual_map == ref_map, f"hpo_pysr mismatch: {actual_map} != {ref_map}"

    for ds in DATASETS:
        assert _stable_target_noise(ds, SEED, NOISE_LEVELS) == ref_map[ds]


def test_openevolve_evaluator_matches_reference():
    from openevolve_pysr.evaluator import _stable_target_noise, _build_target_noise_map, TARGET_NOISE_LEVELS

    assert TARGET_NOISE_LEVELS == NOISE_LEVELS

    ref_map = _reference_build_map(DATASETS, SEED, NOISE_LEVELS)
    actual_map = _build_target_noise_map(DATASETS, SEED, TARGET_NOISE_LEVELS)
    assert actual_map == ref_map, f"openevolve evaluator mismatch: {actual_map} != {ref_map}"

    for ds in DATASETS:
        assert _stable_target_noise(ds, SEED, NOISE_LEVELS) == ref_map[ds]


def test_evolve_basic_sr_matches_reference():
    from evolve_basic_sr import _stable_target_noise, _build_target_noise_map, TARGET_NOISE_LEVELS

    assert TARGET_NOISE_LEVELS == NOISE_LEVELS

    ref_map = _reference_build_map(DATASETS, SEED, NOISE_LEVELS)
    actual_map = _build_target_noise_map(DATASETS, SEED, TARGET_NOISE_LEVELS)
    assert actual_map == ref_map, f"evolve_basic_sr mismatch: {actual_map} != {ref_map}"

    for ds in DATASETS:
        assert _stable_target_noise(ds, SEED, NOISE_LEVELS) == ref_map[ds]


def test_noise_distribution_is_reasonable():
    """Verify that noise levels are distributed across datasets, not all the same."""
    ref_map = _reference_build_map(DATASETS, SEED, NOISE_LEVELS)
    unique_values = set(ref_map.values())
    # With 6 datasets and 4 noise levels, we expect at least 2 distinct values
    assert len(unique_values) >= 2, f"Only got {unique_values} — noise assignment may be degenerate"
    # All values must be from the allowed set
    assert unique_values <= set(NOISE_LEVELS), f"Got unexpected noise values: {unique_values - set(NOISE_LEVELS)}"


def test_deterministic_across_calls():
    """Same inputs always produce the same map."""
    map1 = _reference_build_map(DATASETS, SEED, NOISE_LEVELS)
    map2 = _reference_build_map(DATASETS, SEED, NOISE_LEVELS)
    assert map1 == map2


def test_different_seeds_produce_different_maps():
    """Different seeds should (usually) produce different assignments."""
    map_42 = _reference_build_map(DATASETS, 42, NOISE_LEVELS)
    map_99 = _reference_build_map(DATASETS, 99, NOISE_LEVELS)
    # Not guaranteed to differ for every dataset, but at least one should differ
    assert map_42 != map_99, "Different seeds produced identical maps — suspicious"


if __name__ == "__main__":
    test_evolve_pysr_matches_reference()
    print("PASS: evolve_pysr")
    test_hpo_pysr_matches_reference()
    print("PASS: hpo_pysr")
    test_openevolve_evaluator_matches_reference()
    print("PASS: openevolve_pysr/evaluator")
    test_evolve_basic_sr_matches_reference()
    print("PASS: evolve_basic_sr")
    test_noise_distribution_is_reasonable()
    print("PASS: noise distribution")
    test_deterministic_across_calls()
    print("PASS: deterministic")
    test_different_seeds_produce_different_maps()
    print("PASS: seed sensitivity")
    print("\nAll tests passed!")
