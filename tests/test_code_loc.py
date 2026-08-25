"""Tests for `code_loc` — the docstring/comment-free LOC counter behind `_bundle_loc`."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evolution_helpers import _bundle_loc, code_loc  # noqa: E402


@pytest.mark.parametrize(
    "name,source,expected",
    [
        ("empty", "", 0),
        ("blank lines ignored", "\n\n\nx = 1\n\n", 1),
        ("line comment ignored", "# hi\nx = 1\n", 1),
        ("trailing comment still counts as code", "x = 1  # note\n", 1),
        (
            "multiline docstring ignored",
            '"""\n    foo(x)\n\nblah blah\n"""\nfunction foo(x)\n    x\nend\n',
            3,
        ),
        ("one-line docstring ignored", '"""foo(x)"""\nfunction foo(x)\n    x\nend\n', 3),
        ("code after docstring close on same line", '"""d\nmore"""\nx = 1\n', 1),
        ("block comment ignored", "#=\nnope\nnope\n=#\nx = 1\n", 1),
        ("nested block comment ignored", "#= a #= b =# c =#\nx = 1\n", 1),
        ("code after block comment close", "#= a =# x = 1\n", 1),
        ("hash inside a string is not a comment", 'x = "# not a comment"\n', 1),
    ],
)
def test_code_loc(name, source, expected):
    assert code_loc(source) == expected, name


class _Op:
    def __init__(self, code):
        self.code = code


class _Bundle:
    def __init__(self, operators):
        self.operators = operators


class _RawBundle:
    def __init__(self, raw_module_body):
        self.raw_module_body = raw_module_body


DOCUMENTED_OP = '''"""
    my_op(x)

A docstring the simplify prompt requires.
"""
function my_op(x)
    # an inline comment
    return x + 1
end
'''


def test_bundle_loc_excludes_prose():
    bundle = _Bundle({"mutation": _Op(DOCUMENTED_OP), "loss": _Op(DOCUMENTED_OP)})
    # 3 code lines per operator: `function`, `return`, `end`.
    assert _bundle_loc(bundle) == 6


def test_bundle_loc_skips_missing_operators():
    bundle = _Bundle({"mutation": _Op(DOCUMENTED_OP), "survival": None, "loss": _Op("")})
    assert _bundle_loc(bundle) == 3


def test_bundle_loc_handles_fullsr_raw_module_body():
    assert _bundle_loc(_RawBundle(DOCUMENTED_OP)) == 3
