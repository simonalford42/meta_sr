"""
Utility functions for meta-evolution
"""
import signal
import time
import traceback
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import json
import pickle


# Path to PMLB datasets
PMLB_PATH = Path(__file__).parent / 'pmlb' / 'datasets'


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved {filename}')


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


def load_srbench_dataset(dataset_name: str, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load a single SRBench dataset by name.

    Args:
        dataset_name: Name of the dataset (e.g., 'feynman_I_29_16')
        max_samples: Maximum number of samples to load (subsampling). If None, loads all.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        formula: Ground truth formula string (or empty string if not available)
    """
    dataset_path = PMLB_PATH / dataset_name / f"{dataset_name}.tsv.gz"
    metadata_path = PMLB_PATH / dataset_name / "metadata.yaml"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load data
    df = pd.read_csv(dataset_path, sep='\t', compression='gzip')
    X = df.drop('target', axis=1).values
    y = df['target'].values

    # Subsample if requested
    if max_samples is not None and len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]

    # Try to extract formula from metadata
    formula = ""
    if metadata_path.exists():
        try:
            import yaml
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            if 'description' in metadata:
                desc = metadata['description']
                lines = desc.split('\n')
                for line in lines:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        # Skip lines that are variable ranges like "x1 in [1.0,5.0]"
                        if ' in [' not in line and ' in (' not in line:
                            formula = line
                            break
        except Exception:
            pass

    return X, y, formula


def load_datasets_from_split(split_file: str, max_samples: Optional[int] = None, data_seed: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load all datasets listed in a split file.

    Args:
        split_file: Path to split file (one dataset name per line)
        max_samples: Maximum samples per dataset (subsampling). If None, loads all.
        data_seed: Seed for RNG before each dataset load (for reproducible subsampling).
                   If None, uses current RNG state.

    Returns:
        Dictionary mapping dataset names to (X, y, formula) tuples
    """
    datasets = {}

    with open(split_file, 'r') as f:
        dataset_names = [line.strip() for line in f if line.strip()]

    for name in dataset_names:
        try:
            if data_seed is not None:
                np.random.seed(data_seed)
            X, y, formula = load_srbench_dataset(name, max_samples=max_samples)
            datasets[name] = (X, y, formula)
            # print(f"  Loaded {name}: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"  Warning: Could not load {name}: {e}")

    return datasets


def load_datasets_from_list(dataset_names: List[str], max_samples: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load datasets by a list of names.

    Args:
        dataset_names: List of dataset names
        max_samples: Maximum samples per dataset (subsampling). If None, loads all.

    Returns:
        Dictionary mapping dataset names to (X, y, formula) tuples
    """
    datasets = {}

    for name in dataset_names:
        try:
            X, y, formula = load_srbench_dataset(name, max_samples=max_samples)
            datasets[name] = (X, y, formula)
            print(f"  Loaded {name}: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"  Warning: Could not load {name}: {e}")

    return datasets


class TimeoutError(Exception):
    """Raised when execution times out"""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def run_with_timeout(func, args=(), kwargs=None, name="function",
                     soft_timeout=0.326, hard_timeout=1):
    """
    Run a function with soft and hard timeouts.

    Args:
        func: The function to run
        args: Positional arguments for func
        kwargs: Keyword arguments for func
        name: Name for error messages (e.g., "selection operator", "test 'random_fitness'")
        soft_timeout: Soft timeout in seconds (returns failure if exceeded, but only checked after completion)
        hard_timeout: Hard timeout in seconds (kills execution via signal, catches infinite loops)

    Returns:
        (success, elapsed_time, error_message)
        - success: True if completed within soft_timeout, False otherwise
        - elapsed_time: Time taken in seconds (None if hard timeout hit)
        - error_message: None if success, otherwise description of failure
    """
    if kwargs is None:
        kwargs = {}

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    try:
        signal.alarm(hard_timeout)

        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        signal.alarm(0)

        if elapsed > soft_timeout:
            return False, elapsed, f"{name} timed out: {elapsed:.4f}s > {soft_timeout:.4f}s (soft timeout)"

        return True, elapsed, None

    except TimeoutError:
        signal.alarm(0)
        return False, None, f"{name} killed: infinite loop or exceeded {hard_timeout}s hard timeout"

    except Exception as e:
        signal.alarm(0)
        return False, None, (e, f"{name} failed: {e}")

    finally:
        signal.signal(signal.SIGALRM, old_handler)


def _find_lineno_in_generated_code(exc):
    """
    Find the line number in generated code (<string>) from an exception traceback.

    Walks the traceback to find the last frame that's in <string> (the exec'd code),
    rather than the innermost frame which might be in library code.
    """
    if exc is None or exc.__traceback__ is None:
        return None

    error_lineno = None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<string>":
            error_lineno = tb.tb_lineno
        tb = tb.tb_next

    return error_lineno


def print_code_with_error(code, exc=None, title="Generated code"):
    """
    Print code with the error line highlighted (if exception provided).

    Args:
        code: The source code string
        exc: Optional exception with traceback to highlight the error line
        title: Title for the code block
    """
    error_lineno = _find_lineno_in_generated_code(exc)

    if error_lineno is not None:
        print(f"\n--- {title} with error ---")
        for i, line in enumerate(code.split('\n'), 1):
            marker = "-->" if i == error_lineno else "   "
            print(f"{marker} {i:3d} | {line}")
        print("-" * (len(title) + 20) + "\n")
    else:
        print(f"\n--- {title} ---")
        for i, line in enumerate(code.split('\n'), 1):
            print(f"    {i:3d} | {line}")
        print("-" * (len(title) + 8) + "\n")


def print_error_in_generated_code(code, exc, name="function"):
    """
    Print full error information for LLM-generated code.

    Args:
        code: The source code string
        exc: The exception that was raised
        name: Name of what failed (for error message)
    """
    print(f"{name} failed: {exc}")
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    print_code_with_error(code, exc, title="Generated code")
