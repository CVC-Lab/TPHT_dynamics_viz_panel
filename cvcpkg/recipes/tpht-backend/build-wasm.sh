#!/usr/bin/env bash
# cvcpkg/recipes/tpht-backend/build-wasm.sh — the C++ backend as the page's Embind module
# (single-threaded wasm32): share/tpht-backend/wasm/tpht_backend.{js,wasm} and build-info.json.
# Needs CVC_EMSDK_DIR (cvcpkg's emsdk) and the prefix's wasm32 mujoco and nlohmann-json.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/../_common/env-wasm.sh"

: "${CVC_SOURCE_DIR:?}" "${CVC_BUILD_DIR:?}" "${CVC_INSTALL_DIR:?}"
_roots=""
for _p in "${CVC_DEPS_PREFIX:-}" "${CVC_BUILD_PREFIX:-}"; do
    [ -n "${_p}" ] && _roots="${_roots:+${_roots};}${_p}"
done
cmake -G Ninja \
    -S "${CVC_SOURCE_DIR}/cpp" \
    -B "${CVC_BUILD_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${CVC_INSTALL_DIR}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    -DCMAKE_TOOLCHAIN_FILE="${EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake" \
    -DCMAKE_FIND_ROOT_PATH="${_roots}" \
    -DCMAKE_PREFIX_PATH="${_roots}" \
    -DTPHT_BUILD_TESTS=OFF
cmake --build "${CVC_BUILD_DIR}" -j "${CVC_JOBS:-4}"
cmake --install "${CVC_BUILD_DIR}"

# The module carries MuJoCo and nlohmann-json inside it, so the bundle carries their texts.
_lic="${CVC_INSTALL_DIR}/share/licenses/tpht-backend"
mkdir -p "${_lic}"
cp "${CVC_SOURCE_DIR}/LICENSE" "${_lic}/LICENSE"
for _pkg in mujoco3 mujoco nlohmann-json; do
    for _p in "${CVC_DEPS_PREFIX:-}" "${CVC_BUILD_PREFIX:-}"; do
        if [ -n "${_p}" ] && [ -d "${_p}/share/licenses/${_pkg}" ]; then
            mkdir -p "${_lic}/${_pkg}"
            cp -R "${_p}/share/licenses/${_pkg}/." "${_lic}/${_pkg}/"
            break
        fi
    done
done

for _f in tpht_backend.js tpht_backend.wasm build-info.json; do
    [ -f "${CVC_INSTALL_DIR}/share/tpht-backend/wasm/${_f}" ] || { echo "tpht-backend: ${_f} was not installed" >&2; exit 1; }
done
echo "cvcpkg: installed tpht-backend (wasm32) -> ${CVC_INSTALL_DIR}/share/tpht-backend/wasm"
