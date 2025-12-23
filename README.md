# doverifier

doverifier is a focused toolkit for reasoning about causal expressions using
do-calculus. It provides parsing utilities, rewrite rules, and proof search for
causal probability expressions while preserving the exact semantics of the
original implementation.

Core modules
- `DoWhy_ready/packages/dowhy_env/dowhy/probability.py`
  Defines the causal probability language. This includes:
  - `Do` for do-operator terms (e.g., `do(X)` or `do(X=1)`).
  - `CausalProbability` for expressions like `P(Y | do(X), Z)`.
  - Parsing helpers that accept the string grammar used throughout the project.
- `DoWhy_ready/packages/dowhy_env/dowhy/causal_equiv.py`
  Implements do-calculus rewrite rules and expression equivalence utilities.
  This is where transformations such as Rules 1–3 are defined and applied to
  expressions.
- `DoWhy_ready/packages/dowhy_env/dowhy/find_proof.py`
  Provides BFS-based proof search over do-calculus rewrites, allowing you to
  search for a sequence of valid rewrite steps between two expressions.

Tests
- Generator: `DoWhy_ready/tests/generate_test_results_dowhy_ready.py`
- Results: `DoWhy_ready/results/Test_results_dowhy_ready.json`

Quick run
```bash
python3 DoWhy_ready/tests/generate_test_results_dowhy_ready.py
```
