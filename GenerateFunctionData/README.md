# Generate Function Data

A Python project that takes mathematical functions (written in Python) and generates datasets by evaluating each function on a series of input values, writing one file per function with `(input, output)` pairs.

## Usage

```python
from generate_function_data import generate_dataset, generate_dataset_from_range
import math

# Option 1: Provide your own list of inputs
def my_func(x):
    return x ** 2 + 1

inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]
generate_dataset(my_func, inputs, "my_func_data.csv", func_name="my_func")

# Option 2: Use an evenly spaced range
generate_dataset_from_range(
    math.sin,
    "sin_data.csv",
    start=0,
    stop=2 * math.pi,
    num_points=50,
    func_name="sin",
)
```

Output files are CSV with an optional comment line (`# function: name`), a header row `input,output`, then one row per `(input, output)` pair.

## Running Tests

Ten different mathematical functions are tested; each is evaluated on a series of inputs and written to an individual file. Run:

```bash
pip install -r requirements.txt
pytest test_dataset_generator.py -v
```

## Requirements

- Python 3.10+
- pytest (for tests)
