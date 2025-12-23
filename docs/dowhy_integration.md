# DoWhy Integration Notes

This project keeps do-calculus utilities in a DoWhy-compatible layout under
`dowhy_env/dowhy/`.

Planned integration steps:

1) Copy the module files into the DoWhy package:
   - dowhy/probability.py
   - dowhy/causal_equiv.py
   - dowhy/find_proof.py

2) Ensure relative imports inside the DoWhy package:
   - from .probability import ...
   - from .causal_equiv import ...

3) Expose public symbols in dowhy/__init__.py if needed.

4) Add tests under DoWhy's test suite that compare behavior against
   expected outputs (see tests/ and results/).

5) Run CI-like checks locally using scripts/ci_check.sh.
