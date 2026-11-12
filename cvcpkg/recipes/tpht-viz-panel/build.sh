#!/usr/bin/env bash
# cvcpkg/recipes/tpht-viz-panel/build.sh — assemble the static site into
# share/tpht-viz-panel/ with tools/build_site.py, taking the C++ backend's wasm module,
# MuJoCo's wasm build and ONNX Runtime Web from the prefix (see recipe.yaml).
set -euo pipefail

: "${CVC_SOURCE_DIR:?CVC_SOURCE_DIR must be set}"
: "${CVC_INSTALL_DIR:?CVC_INSTALL_DIR must be set}"

# Any Python 3.9+ runs build_site.py (stdlib only). Prefer cvcpkg's python312 in a prefix,
# then python3 on PATH.
PY=""
for d in "${CVC_BUILD_PREFIX:-}" "${CVC_DEPS_PREFIX:-}" "${CVC_PREFIX:-}"; do
  [ -n "${d}" ] || continue
  for c in "${d}/bin/python3" "${d}/python.exe" "${d}/bin/python3.12"; do
    if [ -x "${c}" ]; then PY="${c}"; break 2; fi
  done
done
[ -n "${PY}" ] || PY="$(command -v python3 || command -v python || true)"
[ -n "${PY}" ] || { echo "tpht-viz-panel: no Python found (cvcpkg install python312 --prefix <prefix>)" >&2; exit 1; }

PREFIXES=()
for d in "${CVC_DEPS_PREFIX:-}" "${CVC_BUILD_PREFIX:-}" "${CVC_PREFIX:-}"; do
  [ -n "${d}" ] && PREFIXES+=(--prefix "${d}")
done

"${PY}" "${CVC_SOURCE_DIR}/tools/build_site.py" \
  --src "${CVC_SOURCE_DIR}" \
  --out "${CVC_INSTALL_DIR}/share/tpht-viz-panel" \
  --with-tests \
  "${PREFIXES[@]}"

for f in index.html vendor/tpht/tpht_backend.wasm vendor/mujoco/mujoco.wasm vendor/ort/ort-wasm-simd-threaded.wasm tests/index.html; do
  [ -f "${CVC_INSTALL_DIR}/share/tpht-viz-panel/${f}" ] || { echo "tpht-viz-panel: ${f} missing from the site" >&2; exit 1; }
done
echo "cvcpkg: installed the TPHT viz panel site -> ${CVC_INSTALL_DIR}/share/tpht-viz-panel"
