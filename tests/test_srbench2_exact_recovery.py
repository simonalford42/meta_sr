from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domains import get_domain
from operator_types import OperatorBundle
from srbench2_full_eval import _with_srbench2_defaults
from srbench_eval_source import apply_srbench2_exact_recovery_protocol


def test_srbench2_ground_truth_defaults_select_exact_recovery_protocol():
    args = _with_srbench2_defaults(["--ground-truth"])

    assert args[-1] == "--srbench2-exact-recovery"
    assert args[args.index("--n-runs") + 1] == "10"
    assert args[args.index("--max-evals") + 1] == "1000000000"
    assert args[args.index("--timeout") + 1] == "3600"
    assert args[args.index("--pysr-wall-limit") + 1] == "3900"


def test_srbench2_black_box_does_not_use_exact_recovery_protocol():
    args = _with_srbench2_defaults(["--black-box"])

    assert "--srbench2-exact-recovery" not in args


def test_srbench2_defaults_preserve_explicit_aliases_and_equals_values():
    args = _with_srbench2_defaults([
        "--ground-truth",
        "--n-trials-per-dataset", "3",
        "--timeout=120",
    ])

    assert "--n-runs" not in args
    assert args[args.index("--n-trials-per-dataset") + 1] == "3"
    assert "--timeout=120" in args
    assert "--timeout" not in args


def test_srbench2_combined_protocols_require_separate_commands():
    with pytest.raises(SystemExit, match="separate commands"):
        _with_srbench2_defaults(["--ground-truth", "--black-box"])


def test_exact_recovery_config_is_serial_float64_without_early_stop():
    config = OperatorBundle.create_default().to_pysr_config({
        "early_stop_condition": 1e-8,
        "elementwise_loss": "L1DistLoss()",
        "timeout_in_seconds": 3600,
        "max_evals": 1_000_000_000,
    })

    configured = apply_srbench2_exact_recovery_protocol(config)
    kwargs = configured.pysr_kwargs

    assert "early_stop_condition" not in kwargs
    assert "elementwise_loss" not in kwargs
    assert kwargs["timeout_in_seconds"] == 3600
    assert kwargs["max_evals"] == 1_000_000_000
    assert kwargs["niterations"] == 1_000_000_000
    assert kwargs["maxsize"] == 30
    assert kwargs["maxdepth"] == 20
    assert kwargs["populations"] == 15
    assert kwargs["population_size"] == 33
    assert kwargs["precision"] == 64
    assert kwargs["parallelism"] == "serial"
    assert kwargs["procs"] == 0
    assert kwargs["batching"] is False
    assert kwargs["unary_operators"] == ["square", "cube", "exp", "log", "sqrt"]


def test_exact_recovery_domain_reuses_every_row(monkeypatch):
    domain = get_domain("srbench2_exact")
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float)
    monkeypatch.setattr(domain, "load_dataset", lambda *args, **kwargs: (X, y, "y=x"))

    X_train, y_train, X_val, y_val, target = domain.load_train_validation("kepler")

    assert X_train is X_val is X
    assert y_train is y_val is y
    assert target == "y=x"
