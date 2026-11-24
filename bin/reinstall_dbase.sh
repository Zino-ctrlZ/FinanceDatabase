#!/usr/bin/env bash
set -euo pipefail

# Prefer openbb_new_use; do not inherit whatever CONDA_PREFIX happens to be.
ENV_PYTHON="${HOME}/miniconda3/envs/openbb_new_use/bin/python"
REPO="$(cd "$(dirname "$0")" && pwd)"

echo "Using: $ENV_PYTHON"
"$ENV_PYTHON" -m pip --version

echo "Uninstalling old/new distribution names (ignore if missing)..."
"$ENV_PYTHON" -m pip uninstall -y FinanceDatabase dbase financedatabase || true

# Leftover egg-info can make pip think the local tree is already "financedatabase".
rm -rf "${REPO}/FinanceDatabase.egg-info" "${REPO}/dbase.egg-info" "${REPO}/financedatabase.egg-info"

echo "Installing local repo editable (distribution name: dbase)..."
"$ENV_PYTHON" -m pip install -e "$REPO"

echo "Installing distributed financedatabase from PyPI (>=2, not the local 0.1)..."
"$ENV_PYTHON" -m pip install --upgrade "financedatabase>=2.0.0"

echo "Verify:"
"$ENV_PYTHON" -m pip show dbase financedatabase
"$ENV_PYTHON" -c "import dbase; print('dbase OK', getattr(dbase, '__file__', dbase))"
"$ENV_PYTHON" -c "import financedatabase; print('financedatabase OK', financedatabase.__file__)"
