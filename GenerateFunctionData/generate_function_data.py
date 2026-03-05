"""
Generate datasets by evaluating mathematical functions on a series of input values.
Supports multi-dimensional inputs; writes one file per function with (inputs..., output) rows.
"""

import csv
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence, Union


def generate_dataset(
    func: Callable[..., float],
    inputs: Iterable[Sequence[Union[int, float]]],
    output_path: Union[str, Path],
    *,
    func_name: str = None,
) -> Path:
    """
    Evaluate a function on each input point and write (input_1, ..., input_d, output) to a file.

    Args:
        func: A callable that takes d numeric arguments and returns a number.
        inputs: Iterable of input points; each point is a sequence of d numbers.
        output_path: Path for the output file (e.g. .csv).
        func_name: Optional name for the function (used in header comment).

    Returns:
        The path to the written file.

    Raises:
        ValueError: If evaluation fails for an input.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    num_dims = None
    for point in inputs:
        point = tuple(float(p) for p in point)
        if num_dims is None:
            num_dims = len(point)
        if len(point) != num_dims:
            raise ValueError(f"Input dimension mismatch: expected {num_dims}, got {len(point)}")
        try:
            y = func(*point)
            rows.append((*point, float(y)))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Evaluation failed at input {point}: {e}") from e

    if num_dims is None:
        num_dims = 0

    input_headers = [f"input_{i+1}" for i in range(num_dims)]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        if func_name:
            writer.writerow([f"# function: {func_name}"])
        writer.writerow([*input_headers, "output"])
        writer.writerows(rows)

    return output_path


def linspace(start: float, stop: float, num: int) -> list[float]:
    """Return num evenly spaced values from start to stop (inclusive endpoints)."""
    if num <= 0:
        return []
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def meshgrid_2d(
    x_start: float, x_stop: float, x_num: int,
    y_start: float, y_stop: float, y_num: int,
) -> list[tuple[float, float]]:
    """Return list of (x, y) points forming a 2D grid."""
    xs = linspace(x_start, x_stop, x_num)
    ys = linspace(y_start, y_stop, y_num)
    return [(x, y) for x in xs for y in ys]


def meshgrid_3d(
    x_start: float, x_stop: float, x_num: int,
    y_start: float, y_stop: float, y_num: int,
    z_start: float, z_stop: float, z_num: int,
) -> list[tuple[float, float, float]]:
    """Return list of (x, y, z) points forming a 3D grid."""
    xs = linspace(x_start, x_stop, x_num)
    ys = linspace(y_start, y_stop, y_num)
    zs = linspace(z_start, z_stop, z_num)
    return [(x, y, z) for x in xs for y in ys for z in zs]


def generate_dataset_from_range(
    func: Callable[[float], float],
    output_path: Union[str, Path],
    start: float = -10.0,
    stop: float = 10.0,
    num_points: int = 100,
    *,
    func_name: str = None,
) -> Path:
    """
    Evaluate a single-argument function on an evenly spaced range and write to a file.
    """
    inputs = [(x,) for x in linspace(start, stop, num_points)]
    return generate_dataset(func, inputs, output_path, func_name=func_name)
