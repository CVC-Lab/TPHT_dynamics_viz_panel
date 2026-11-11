#!/usr/bin/env bash
# cvcpkg/recipes/tpht-lab-cp312/build.sh — build the tpht-lab wheel from this repository with
# the prefix's CPython 3.12 and install it into the staging prefix. The wheel is built from a
# copy of just the package files, so the checkout gets no build/ or egg-info litter.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
_CVC_ENV="${SCRIPT_DIR}/../_common/env-${CVC_PLATFORM:-linux}.sh"
# shellcheck disable=SC1090
[ -f "${_CVC_ENV}" ] && . "${_CVC_ENV}"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/../_common/python-wheel.sh"   # cvc_python_exe, cvc_python_check

: "${CVC_PYTHON_ABI:=cp312}"
: "${CVC_PYTHON_INTERPRETER:=python312}"
PY_EXE="$(cvc_python_exe)"
echo "tpht-lab-cp312: building with ${PY_EXE}"

# The PEP-517 backend (setuptools-cp312) sits in the build prefix; bridge it onto the path.
if [ -n "${CVC_BUILD_PREFIX:-}" ]; then
  export PYTHONPATH="${CVC_BUILD_PREFIX}/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
fi

WORK="${CVC_BUILD_DIR:-$(mktemp -d)}/tpht-lab-src"
rm -rf "${WORK}"
mkdir -p "${WORK}"
cp -R "${CVC_SOURCE_DIR}/tpht_lab" "${WORK}/tpht_lab"
cp "${CVC_SOURCE_DIR}/pyproject.toml" "${CVC_SOURCE_DIR}/README.md" "${CVC_SOURCE_DIR}/LICENSE" "${WORK}/"
find "${WORK}" -name __pycache__ -prune -exec rm -rf {} +

WHEELHOUSE="${WORK}/../wheelhouse"
rm -rf "${WHEELHOUSE}"
mkdir -p "${WHEELHOUSE}"
"${PY_EXE}" -m pip wheel --no-build-isolation --no-deps --no-index --no-cache-dir \
  --wheel-dir "${WHEELHOUSE}" "${WORK}"
WHEEL=""
for WHEEL in "${WHEELHOUSE}"/*.whl; do [ -e "${WHEEL}" ] && break; WHEEL=""; done
[ -n "${WHEEL}" ] || { echo "tpht-lab-cp312: no wheel produced" >&2; exit 1; }
echo "tpht-lab-cp312: built $(basename "${WHEEL}")"

"${PY_EXE}" -m pip install --no-index --no-deps --no-compile --ignore-installed \
  --prefix "${CVC_INSTALL_DIR}" "${WHEEL}"

unset PYTHONPATH
cvc_python_check "import tpht_lab, tpht_lab.cli, tpht_lab.serve, tpht_lab.export, tpht_lab.dm_control_src; assert tpht_lab.__version__ == '0.1.0'"
