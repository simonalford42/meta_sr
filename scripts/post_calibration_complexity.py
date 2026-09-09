"""Count a*f+b after bounded affine rewrites and constant folding.

All constants, variables, binary operators and unary functions cost one node,
matching the saved PySR frontiers. This is not a global symbolic minimizer.
Only exact algebraic identities are used; no tolerance-based coefficient removal.
"""
import ast
import math


def const(v):
    return ("constant", float(v))


def is_const(t):
    return t[0] == "constant"


def count(t):
    return 1 if t[0] in {"constant", "variable"} else 1+sum(count(x) for x in t[1:])


def parse(expression):
    def visit(n):
        if isinstance(n, ast.Constant):
            return const(n.value)
        if isinstance(n, ast.Name) and n.id == "x0":
            return ("variable", "x0")
        if isinstance(n, ast.UnaryOp):
            value = visit(n.operand)
            if isinstance(n.op, ast.UAdd):
                return value
            assert isinstance(n.op, ast.USub) and is_const(value)
            return const(-value[1])
        if isinstance(n, ast.BinOp):
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(n.op)]
            return (op, visit(n.left), visit(n.right))
        if isinstance(n, ast.Call):
            assert n.func.id in {"square", "cube", "sqrt", "log", "exp"} and len(n.args) == 1
            return (n.func.id, visit(n.args[0]))
        raise ValueError(ast.dump(n))
    return visit(ast.parse(expression, mode="eval").body)


def binary(op, u, v):
    if is_const(u) and is_const(v):
        try:
            value = {"+": lambda: u[1]+v[1], "-": lambda: u[1]-v[1],
                     "*": lambda: u[1]*v[1], "/": lambda: u[1]/v[1]}[op]()
            if math.isfinite(value):
                return const(value)
        except (ZeroDivisionError, OverflowError):
            pass
    if op == "+":
        if is_const(u) and u[1] == 0: return v
        if is_const(v) and v[1] == 0: return u
    if op == "-" and is_const(v) and v[1] == 0: return u
    if op == "*":
        if (is_const(u) and u[1] == 0) or (is_const(v) and v[1] == 0): return const(0)
        if is_const(u) and u[1] == 1: return v
        if is_const(v) and v[1] == 1: return u
    if op == "/" and is_const(v) and v[1] == 1: return u
    return (op, u, v)


def fold(t):
    if t[0] in {"constant", "variable"}: return t
    if len(t) == 3: return binary(t[0], fold(t[1]), fold(t[2]))
    u = fold(t[1])
    if is_const(u):
        try:
            value = {"square": lambda x:x*x, "cube": lambda x:x*x*x,
                     "sqrt": math.sqrt, "log": math.log, "exp": math.exp}[t[0]](u[1])
            if math.isfinite(value): return const(value)
        except (ValueError, OverflowError):
            pass
    return (t[0], u)


def scale(t, a):
    options = [binary("*", const(a), t)]
    if len(t) == 3 and t[0] not in {"constant", "variable"}:
        op, u, v = t
        if op == "*":
            options += [binary("*", scale(u, a), v), binary("*", u, scale(v, a))]
        elif op == "/":
            options.append(binary("/", scale(u, a), v))
            if a != 0 and math.isfinite(1/a):
                options.append(binary("/", u, scale(v, 1/a)))
        elif op in {"+", "-"}:
            options.append(binary(op, scale(u, a), scale(v, a)))
    return min(options, key=count)


def offset(t, b):
    options = [binary("+", t, const(b))]
    if len(t) == 3 and t[0] not in {"constant", "variable"}:
        op, u, v = t
        if op == "+":
            options += [binary("+", offset(u, b), v), binary("+", u, offset(v, b))]
        elif op == "-":
            options += [binary("-", offset(u, b), v), binary("-", u, offset(v, -b))]
        elif op == "*":
            if is_const(u) and u[1] != 0:
                options.append(binary("*", u, offset(v, b/u[1])))
            if is_const(v) and v[1] != 0:
                options.append(binary("*", offset(u, b/v[1]), v))
        elif op == "/" and is_const(v):
            options.append(binary("/", offset(u, b*v[1]), v))
    return min(options, key=count)


def format_tree(t):
    if is_const(t): return repr(t[1])
    if t[0] == "variable": return t[1]
    if len(t) == 3: return f"({format_tree(t[1])} {t[0]} {format_tree(t[2])})"
    return f"{t[0]}({format_tree(t[1])})"


def calibrate(expression, a, b):
    t = offset(scale(fold(parse(expression)), a), b)
    return count(t), format_tree(t)
