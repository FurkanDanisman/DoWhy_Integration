import importlib
import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
DOWHY_ENV = ROOT / "packages" / "dowhy_env"


def _safe_call(fn):
    try:
        return fn()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _swap_modules(temp_modules):
    saved = {}
    for name, mod in temp_modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    return saved


def _restore_modules(saved):
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def load_dowhy_env(env_path):
    for name in list(sys.modules):
        if name == "dowhy" or name.startswith("dowhy."):
            sys.modules.pop(name, None)
    sys.path.insert(0, str(env_path))
    try:
        dw_prob = importlib.import_module("dowhy.probability")
        dw_ce = importlib.import_module("dowhy.causal_equiv")
        dw_fp = importlib.import_module("dowhy.find_proof")
    finally:
        sys.path.pop(0)
    return dw_prob, dw_ce, dw_fp


def run_in_dowhy_env(env_path, fn):
    dw_prob, dw_ce, dw_fp = load_dowhy_env(env_path)
    saved = _swap_modules({"probability": dw_prob, "causal_equiv": dw_ce, "find_proof": dw_fp})
    try:
        return fn(dw_prob, dw_ce, dw_fp)
    finally:
        _restore_modules(saved)


def add_case(groups, category, name, input_repr, dowhy_out):
    case = {
        "id": name,
        "input": input_repr,
        "dowhy_output": dowhy_out,
        "expected": dowhy_out,
    }
    groups.setdefault(category, []).append(case)


def build_cases():
    groups = {}

    X, Y, Z, W, V, V2, A, B, C, U = sp.symbols("X Y Z W V V2 A B C U")

    do_cases = [
        ("do_1", (X, None)),
        ("do_2", (X, 1)),
        ("do_3", (X, 0)),
        ("do_4", (X, sp.Symbol("x"))),
        ("do_5", (X, -1)),
    ]

    mult_exprs = [
        ("mult_1", ["P(Y)", "P(Z)"]),
        ("mult_2", ["P(Y|X)", "P(Z)"]),
        ("mult_3", ["P(Y|do(X))", "P(Z)"]),
        ("mult_4", ["P(Y|X,Z)", "P(Z)"]),
        ("mult_5", ["P(A|B)", "P(B|C)"]),
    ]

    sum_cases = [
        ("sum_1", Z, "P(Y|X)"),
        ("sum_2", (Z, W), "P(Y)"),
        ("sum_3", (Z, W, V), "P(Y|do(X))"),
        ("sum_4", Z, "P(A|B)*P(B)"),
        ("sum_5", (V2,), "P(Y|Z,do(X))"),
    ]

    parse_basic = [
        "P(Y)",
        "P(Y|X)",
        "P(Y|X,Z)",
        "P(Y|Z,X)",
        "P(Y|do(X))",
        "P(Y|do(X),Z)",
        "P(Y|Z,do(X))",
        "P(Y|do(X=1),Z)",
        "P(Y|do(X=1),Z=2)",
        "P(Y|X=1)",
        "P(Y|X=1,Z=2)",
        "P(Y=1)",
        "P(Y=0|X)",
        "P(Y=1|X=0)",
        "P(Y|do(X),Z=2)",
        "P(Y|do(X=0),Z=2)",
        "P(Y|do(X),Z,W)",
        "P(Y|do(X),Z,W=1)",
        "P(Y|Z,W=1)",
        "P(Y|do(X),do(Z))",
    ]

    parse_subscript = [
        "P(Y_{X=1})",
        "P(Y_{X=0})",
        "P(Y_{X})",
        "P(Y_{X=1,Z=2})",
        "P(Y_{X=1,Z})",
        "P(Y_{X=1,Z=2,V=3})",
        "P(Y_{X=0,V2=1})",
        "P(Y_{X=0,V2=0})",
        "P(Y_{X=1,V2=0})",
        "P(Y_{X=1,V2=0,Z=3})",
    ]

    parse_products = [
        "P(A|B)*P(B)",
        "P(A)*P(B)",
        "P(A|do(B))*P(B)",
        "P(A|B)*P(B|C)",
        "P(A|do(B),C)*P(B)",
        "P(A|B)*P(B|do(C))",
        "P(A|B)*P(B)*P(C)",
        "P(A|do(B))*P(B|do(C))",
        "P(A|B)*P(B|C)*P(C)",
        "P(A)*P(B)*P(C)",
    ]

    parse_arith = [
        "P(Y)-P(Z)",
        "P(Y)+P(Z)",
        "P(Y|X)-P(Y|Z)",
        "P(Y|X)+P(Z|X)",
        "P(Y|do(X)) - P(Y|do(Z))",
        "P(Y_{X=1})-P(Y_{X=0})",
        "2*P(Y|X)",
        "P(Y|X)/P(Z)",
        "P(Y|X)+P(Z)",
        "P(Y|X)-P(Z)",
        "P(Y|X)*P(Z)",
        "P(Y|X)-1",
        "1-P(Y|X)",
        "P(Y|X)+2",
        "P(Y|X)*3+P(Z)",
    ]

    sort_cases = [
        ("sort_1", ("Y", ["do(X)", "Z"], ["Z", "do(X)"])),
        ("sort_2", ("Y", ["do(X)", "Eq(Z,1)"], ["Eq(Z,1)", "do(X)"])),
        ("sort_3", ("Eq(Y,1)", ["do(X)", "Z"], ["Z", "do(X)"])),
        ("sort_4", ("Y", ["Z", "W", "do(X)"], ["do(X)", "W", "Z"])),
        ("sort_5", ("Y", ["do(X=1)", "Z"], ["Z", "do(X=1)"])),
    ]

    rule1_cases = [
        ("rule1_1", "P(Y | do(X), Z)", {"X": ["Y"]}),
        ("rule1_2", "P(Y | do(X), Z)", {"Z": ["Y"]}),
        ("rule1_3", "P(Y | do(X), Z, W)", {"X": ["Y"], "Z": ["Y"]}),
        ("rule1_u1", "P(Y | do(X), Z)", {"U": ["X", "Y"], "Z": ["Y"]}),
        ("rule1_u2", "P(Y | do(X), Z, W)", {"U": ["X", "Y"], "Z": ["Y"]}),
    ]

    rule2_cases = [
        ("rule2_1", "P(Y | do(X), do(Z))", {"X": ["Y"]}),
        ("rule2_2", "P(Y | do(X), do(Z), W)", {"X": ["Y"], "Z": ["Y"]}),
        ("rule2_u1", "P(Y | do(X), do(Z))", {"U": ["X", "Y"], "Z": ["Y"]}),
        ("rule2_u2", "P(Y | do(X), do(Z), W)", {"U": ["X", "Y"], "Z": ["Y"]}),
        ("rule2_5", "P(Y | do(X), do(Z))", {}),
    ]

    rule3_cases = [
        ("rule3_1", "P(Y | do(X), do(Z), W)", {"X": ["Y"]}),
        ("rule3_2", "P(Y | do(X), do(Z), W)", {"Z": ["W"]}),
        ("rule3_3", "P(Y | do(X), do(Z), W=1)", {"X": ["Y"], "Z": ["Y"]}),
        ("rule3_u1", "P(Y | do(X), do(Z), W)", {"U": ["X", "Y"], "Z": ["Y"]}),
        ("rule3_5", "P(Y | do(X), do(Z), W)", {}),
    ]

    suggest_cases = [
        ("suggest_1", "P(Y | Z)", {"Z": ["Y"]}),
        ("suggest_2", "P(Y | Z)", {"X": ["Y"]}),
        ("suggest_3", "P(Y | do(X), Z)", {"X": ["Y"], "Z": ["Y"]}),
        ("suggest_4", "P(Y | Z, W)", {"Z": ["Y"], "W": ["Z"]}),
        ("suggest_u1", "P(Y | Z)", {"U": ["X", "Y"], "Z": ["Y"]}),
    ]

    find_cases = [
        ("proof_1", "P(Y | do(X))", "P(Y | do(X))"),
        ("proof_2", "P(Y | do(X), Z)", "P(Y | do(X), Z)"),
        ("proof_3", "P(Y | Z)", "P(Y | Z)"),
        ("proof_4", "P(Y | do(X))", "P(Y | do(X))"),
        ("proof_5", "P(Y | do(X), Z)", "P(Y | do(X), Z)"),
    ]

    def _graph_symbols(graph):
        sym_graph = {}
        for k, vals in graph.items():
            sym_k = sp.Symbol(k)
            sym_graph[sym_k] = [sp.Symbol(v) for v in vals]
        return sym_graph

    def run_env(prob, ce, fp):
        local_groups = {}

        for name, (var, val) in do_cases:
            if val is None:
                new_out = _safe_call(lambda: str(prob.Do(var)))
                input_repr = f"Do({var})"
            else:
                new_out = _safe_call(lambda: str(prob.Do(var, val)))
                input_repr = f"Do({var}={val})"
            local_groups.setdefault("do_operator", []).append((name, input_repr, new_out))

        for name, parts in mult_exprs:
            new_out = _safe_call(
                lambda: str(prob.Mult(*[prob.CausalProbability.parse(p) for p in parts]))
            )
            input_repr = " * ".join(parts)
            local_groups.setdefault("mult", []).append((name, input_repr, new_out))

        for name, vars_, expr in sum_cases:
            new_out = _safe_call(
                lambda: str(prob.SumOver(vars_, prob.CausalProbability.parse(expr)))
            )
            input_repr = f"SumOver({vars_}, {expr})"
            local_groups.setdefault("sum_over", []).append((name, input_repr, new_out))

        for i, expr in enumerate(parse_basic, 1):
            new_out = _safe_call(lambda: str(prob.CausalProbability.parse(expr)))
            local_groups.setdefault("parse_basic", []).append((f"parse_basic_{i}", expr, new_out))

        for i, expr in enumerate(parse_subscript, 1):
            new_out = _safe_call(lambda: str(prob.CausalProbability.parse(expr)))
            local_groups.setdefault("parse_subscript", []).append(
                (f"parse_subscript_{i}", expr, new_out)
            )

        for i, expr in enumerate(parse_products, 1):
            new_out = _safe_call(lambda: str(prob.CausalProbability.parse(expr)))
            local_groups.setdefault("parse_products", []).append(
                (f"parse_products_{i}", expr, new_out)
            )

        for i, expr in enumerate(parse_arith, 1):
            new_out = _safe_call(lambda: str(prob.CausalProbability.parse(expr)))
            local_groups.setdefault("parse_arithmetic", []).append(
                (f"parse_arithmetic_{i}", expr, new_out)
            )

        for name, (outcome, conds1, conds2) in sort_cases:

            def build_expr(outcome_str, conds):
                out = (
                    prob.CausalProbability._parse_variable_assignment(outcome_str)
                    if isinstance(outcome_str, str)
                    else outcome_str
                )
                cond_objs = []
                for c in conds:
                    if c.startswith("do("):
                        if c == "do(X)":
                            cond_objs.append(prob.Do(X))
                        elif c == "do(X=1)":
                            cond_objs.append(prob.Do(X, 1))
                    elif c.startswith("Eq("):
                        cond_objs.append(sp.Eq(Z, 1))
                    else:
                        cond_objs.append(Z if c == "Z" else W)
                return prob.CausalProbability(out, *cond_objs)

            expr1 = build_expr(outcome, conds1)
            expr2 = build_expr(outcome, conds2)
            new_out = _safe_call(
                lambda: {"str1": str(expr1), "str2": str(expr2), "equal": expr1 == expr2}
            )
            input_repr = {"outcome": str(outcome), "conds1": conds1, "conds2": conds2}
            local_groups.setdefault("sorting", []).append((name, input_repr, new_out))

        for name, expr_str, graph in rule1_cases:
            expr = prob.CausalProbability.parse(expr_str)
            new_out = _safe_call(
                lambda: [
                    str(o) for o in ce.CausalExpr(expr, _graph_symbols(graph)).apply_rule_1_all()
                ]
            )
            local_groups.setdefault("rule1", []).append(
                (name, {"expr": expr_str, "graph": graph}, new_out)
            )

        for name, expr_str, graph in rule2_cases:
            expr = prob.CausalProbability.parse(expr_str)
            new_out = _safe_call(
                lambda: [
                    str(o) for o in ce.CausalExpr(expr, _graph_symbols(graph)).apply_rule_2_all()
                ]
            )
            local_groups.setdefault("rule2", []).append(
                (name, {"expr": expr_str, "graph": graph}, new_out)
            )

        for name, expr_str, graph in rule3_cases:
            expr = prob.CausalProbability.parse(expr_str)
            new_out = _safe_call(
                lambda: [
                    str(o) for o in ce.CausalExpr(expr, _graph_symbols(graph)).apply_rule_3_all()
                ]
            )
            local_groups.setdefault("rule3", []).append(
                (name, {"expr": expr_str, "graph": graph}, new_out)
            )

        for name, expr_str, graph in suggest_cases:
            expr = prob.CausalProbability.parse(expr_str)
            new_out = _safe_call(lambda: ce.CausalExpr(expr, _graph_symbols(graph)).suggest_fix())
            local_groups.setdefault("suggest_fix", []).append(
                (name, {"expr": expr_str, "graph": graph}, new_out)
            )

        for name, start_str, target_str in find_cases:
            start = prob.CausalProbability.parse(start_str)
            target = prob.CausalProbability.parse(target_str)
            new_out = _safe_call(
                lambda: fp.CausalProofFinder(_graph_symbols({"X": ["Y"]})).find_proof(start, target)
            )
            local_groups.setdefault("find_proof", []).append(
                (name, {"start": start_str, "target": target_str}, new_out)
            )

        return local_groups

    dowhy_groups = run_in_dowhy_env(DOWHY_ENV, run_env)

    for group, cases in dowhy_groups.items():
        for name, input_repr, dowhy_out in cases:
            add_case(groups, group, name, input_repr, dowhy_out)

    total = sum(len(v) for v in groups.values())
    if total != 100:
        raise RuntimeError(f"Expected 100 cases, got {total}")

    return groups


def main():
    groups = build_cases()
    out_path = ROOT / "results" / "Test_results_dowhy_ready.json"
    out_path.write_text(json.dumps(groups, indent=2, sort_keys=True), encoding="ascii")

    total = sum(len(v) for v in groups.values())
    print(f"Wrote {total} cases to {out_path}")


if __name__ == "__main__":
    main()
