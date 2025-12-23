import importlib
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
DOWHY_ENV = ROOT / "packages" / "dowhy_env"


def _load_dowhy():
    for name in list(sys.modules):
        if name == "dowhy" or name.startswith("dowhy."):
            sys.modules.pop(name, None)
    sys.path.insert(0, str(DOWHY_ENV))
    try:
        return importlib.import_module("dowhy")
    finally:
        sys.path.pop(0)


def test_do_string_forms():
    _load_dowhy()
    prob = importlib.import_module("dowhy.probability")
    x = sp.Symbol("X")
    assert str(prob.Do(x)) == "do(X)"
    assert str(prob.Do(x, 1)) == "do(X=1)"


def test_parse_basic_probability():
    _load_dowhy()
    prob = importlib.import_module("dowhy.probability")
    expr = "P(Y|Z,do(X))"
    assert str(prob.CausalProbability.parse(expr)) == "P(Y | do(X), Z)"


def test_rule1_application():
    _load_dowhy()
    prob = importlib.import_module("dowhy.probability")
    ce = importlib.import_module("dowhy.causal_equiv")
    x = sp.Symbol("X")
    y = sp.Symbol("Y")
    z = sp.Symbol("Z")
    expr = prob.CausalProbability(y, prob.Do(x), z)
    graph = {x: [y]}
    outs = [str(o) for o in ce.CausalExpr(expr, graph).apply_rule_1_all()]
    assert outs


def test_find_proof_identity():
    _load_dowhy()
    prob = importlib.import_module("dowhy.probability")
    fp = importlib.import_module("dowhy.find_proof")
    x = sp.Symbol("X")
    y = sp.Symbol("Y")
    expr = prob.CausalProbability(y, prob.Do(x))
    proof = fp.CausalProofFinder({x: [y]}).find_proof(expr, expr)
    assert proof == []
