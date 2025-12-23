#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

python3 -m pytest "${ROOT_DIR}/tests"

if command -v ruff >/dev/null 2>&1; then
  ruff check "${ROOT_DIR}/packages/dowhy_env/dowhy/probability.py" \
    "${ROOT_DIR}/packages/dowhy_env/dowhy/causal_equiv.py" \
    "${ROOT_DIR}/packages/dowhy_env/dowhy/find_proof.py" \
    "${ROOT_DIR}/tests"
else
  echo "ruff not installed; skipping"
fi

if command -v black >/dev/null 2>&1; then
  black --check "${ROOT_DIR}/packages/dowhy_env/dowhy/probability.py" \
    "${ROOT_DIR}/packages/dowhy_env/dowhy/causal_equiv.py" \
    "${ROOT_DIR}/packages/dowhy_env/dowhy/find_proof.py" \
    "${ROOT_DIR}/tests"
else
  echo "black not installed; skipping"
fi

if command -v mypy >/dev/null 2>&1; then
  mypy --config-file "${ROOT_DIR}/pyproject.toml" \
    "${ROOT_DIR}/packages/dowhy_env/dowhy/probability.py" \
    "${ROOT_DIR}/packages/dowhy_env/dowhy/causal_equiv.py" \
    "${ROOT_DIR}/packages/dowhy_env/dowhy/find_proof.py"
else
  echo "mypy not installed; skipping"
fi
