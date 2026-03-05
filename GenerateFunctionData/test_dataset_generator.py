"""
Test the dataset generator. Full set is 500 function types × 20 parameterizations = 10,000;
tests run on a subset of 50 examples only.
"""

import csv
import math
import tempfile
from pathlib import Path

import pytest

from generate_function_data import generate_dataset, generate_dataset_from_range, meshgrid_2d, meshgrid_3d
from sample_functions import FUNCTIONS, FUNCTIONS_TEST_SUBSET, TOTAL_FUNCTIONS


def _inputs_for_dims(num_dims: int) -> list:
    """Return a small grid of input points for the given dimension."""
    if num_dims == 2:
        return meshgrid_2d(-1.0, 1.0, 3, -1.0, 1.0, 3)  # 9 points
    if num_dims == 3:
        return meshgrid_3d(-1.0, 1.0, 2, -1.0, 1.0, 2, -1.0, 1.0, 2)  # 8 points
    raise ValueError(f"Unsupported num_dims: {num_dims}")


def test_total_functions_count():
    """Full FUNCTIONS list has 10,000 entries (500 types × 20 parameterizations)."""
    assert len(FUNCTIONS) == TOTAL_FUNCTIONS
    assert TOTAL_FUNCTIONS == 10_000


def test_fifty_functions_each_write_to_individual_file():
    """Run generator on 50 examples only; all outputs recorded in individual files."""
    subset = FUNCTIONS_TEST_SUBSET
    assert len(subset) == 50
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "datasets"
        out_dir.mkdir()

        written_paths = []
        for func, name, num_dims in subset:
            inputs = _inputs_for_dims(num_dims)
            output_path = out_dir / f"{name}_data.csv"
            path = generate_dataset(func, inputs, output_path, func_name=name)
            written_paths.append(path)
            assert path.exists()
            assert path.name == f"{name}_data.csv"

        assert len(written_paths) == 50

        for (func, name, num_dims), path in zip(subset, written_paths):
            inputs = _inputs_for_dims(num_dims)
            with open(path, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            data_header = next((r for r in rows if r and r[0] == "input_1"), None)
            assert data_header is not None
            data_start = rows.index(data_header) + 1
            data_rows = rows[data_start:]
            assert len(data_rows) == len(inputs)
            for i, row in enumerate(data_rows):
                point = tuple(float(row[j]) for j in range(num_dims))
                out_val = float(row[num_dims])
                expected = func(*point)
                assert math.isclose(out_val, expected, rel_tol=1e-9, abs_tol=1e-9) or (
                    math.isnan(out_val) and math.isnan(expected)
                ), f"{name}{point} = {out_val} vs expected {expected}"


def test_generate_dataset_from_range_creates_file():
    """generate_dataset_from_range produces correct number of rows (1D)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sin_range.csv"
        generate_dataset_from_range(
            math.sin, path, start=0, stop=math.pi, num_points=11, func_name="sin"
        )
        assert path.exists()
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        data = [r for r in rows if r and not r[0].startswith("#") and r[0] != "input_1"]
        assert len(data) == 11


def test_single_function_roundtrip():
    """Single function: inputs and outputs match when read back (2D)."""
    # First test entry is linear_2d with params (0.5, 0.5, 0) -> f(x,y) = 0.5*x + 0.5*y
    func, name, num_dims = FUNCTIONS_TEST_SUBSET[0]
    assert num_dims == 2
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"{name}.csv"
        inputs = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        generate_dataset(func, inputs, path, func_name=name)
        with open(path, newline="") as f:
            content = f.read()
        lines = content.strip().split("\n")
        data_lines = [l for l in lines if l and not l.startswith("#")][1:]
        for line, pt in zip(data_lines, inputs):
            parts = line.split(",")
            x, y = float(parts[0]), float(parts[1])
            out_val = float(parts[2])
            assert (x, y) == pt
            assert math.isclose(out_val, 0.5 * x + 0.5 * y, rel_tol=1e-9)
