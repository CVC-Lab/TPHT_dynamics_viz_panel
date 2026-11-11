#!/usr/bin/env bash
# cvcpkg/recipes/tpht-backend/build.sh — the C++ backend natively (Linux, macOS): libtpht_core,
# its headers and tpht-sim. The googletest suite runs before anything is installed, against
# the site's assets and fixtures in this checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1090
. "${SCRIPT_DIR}/../_common/env-${CVC_PLATFORM:-linux}.sh"

: "${CVC_SOURCE_DIR:?}" "${CVC_BUILD_DIR:?}" "${CVC_INSTALL_DIR:?}"
cmake -G Ninja \
    -S "${CVC_SOURCE_DIR}/cpp" \
    -B "${CVC_BUILD_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${CVC_INSTALL_DIR}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    -DTPHT_BUILD_TESTS=ON
cmake --build "${CVC_BUILD_DIR}" -j "${CVC_JOBS:-4}"

# The test binary finds libmujoco through its build-tree RPATH; the prefixes' lib dirs are a
# fallback for loaders that ignore it.
_libs=""
for _p in "${CVC_DEPS_PREFIX:-}" "${CVC_BUILD_PREFIX:-}"; do
    [ -n "${_p}" ] && _libs="${_libs:+${_libs}:}${_p}/lib"
done
LD_LIBRARY_PATH="${_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
DYLD_LIBRARY_PATH="${_libs}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
    ctest --test-dir "${CVC_BUILD_DIR}" --output-on-failure

cmake --install "${CVC_BUILD_DIR}"
mkdir -p "${CVC_INSTALL_DIR}/share/licenses/tpht-backend"
cp "${CVC_SOURCE_DIR}/LICENSE" "${CVC_INSTALL_DIR}/share/licenses/tpht-backend/LICENSE"

LD_LIBRARY_PATH="${_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
DYLD_LIBRARY_PATH="${_libs}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
    "${CVC_INSTALL_DIR}/bin/tpht-sim" version
echo "cvcpkg: installed tpht-backend (native) -> ${CVC_INSTALL_DIR}"
