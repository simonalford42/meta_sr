#!/usr/bin/env python3
"""
Run all 10,000 function parameterizations (500 types × 20 each) and write
their evaluation outputs to ./datasets/, one CSV file per function.
"""

from pathlib import Path

from generate_function_data import generate_dataset, meshgrid_2d, meshgrid_3d
from sample_functions import FUNCTIONS

OUTPUT_DIR = Path("datasets")
# Grids for 2D and 3D: small step for reasonable file size
INPUTS_2D = meshgrid_2d(-2.0, 2.0, 5, -2.0, 2.0, 5)   # 25 points
INPUTS_3D = meshgrid_3d(-1.0, 1.0, 3, -1.0, 1.0, 3, -1.0, 1.0, 3)  # 27 points


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for func, name, num_dims in FUNCTIONS:
        inputs = INPUTS_2D if num_dims == 2 else INPUTS_3D
        path = OUTPUT_DIR / f"{name}_data.csv"
        generate_dataset(func, inputs, path, func_name=name)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
