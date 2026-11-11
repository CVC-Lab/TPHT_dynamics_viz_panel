#!/usr/bin/env bash
# cvcpkg/recipes/onnxruntime-web/build.sh — stage ONNX Runtime Web 1.20.1's WebAssembly-only
# build from the npm package into share/onnxruntime-web/, and its MIT licence plus third-party
# notices into share/licenses/onnxruntime-web/, cvcpkg's place for a bundle's licence texts.
# Nothing to compile: the package carries the built files. The npm tarball has no LICENSE, so
# the recipe keeps the texts from the onnxruntime repository at tag v1.20.1 beside this script
# (and installs them itself, which cvcpkg then leaves alone):
#   LICENSE                sha256 2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c
#   ThirdPartyNotices.txt  sha256 cf7342f7ba482ef715ae58f5f497a8d3564fa255164175aea324cd293c5701a0
set -euo pipefail

: "${CVC_SOURCE_DIR:?CVC_SOURCE_DIR must be set}"
: "${CVC_INSTALL_DIR:?CVC_INSTALL_DIR must be set}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DEST="${CVC_INSTALL_DIR}/share/onnxruntime-web"
DOC="${CVC_INSTALL_DIR}/share/licenses/onnxruntime-web"
mkdir -p "${DEST}" "${DOC}"
for f in ort.wasm.min.mjs ort-wasm-simd-threaded.mjs ort-wasm-simd-threaded.wasm; do
  [ -f "${CVC_SOURCE_DIR}/dist/${f}" ] || { echo "onnxruntime-web: dist/${f} missing from the tarball" >&2; exit 1; }
  cp "${CVC_SOURCE_DIR}/dist/${f}" "${DEST}/${f}"
done
cp "${CVC_SOURCE_DIR}/package.json" "${DEST}/package.json"
cp "${SCRIPT_DIR}/LICENSE" "${SCRIPT_DIR}/ThirdPartyNotices.txt" "${DOC}/"

# The package.json must be the version this recipe pins.
grep -q '"version": "1.20.1"' "${DEST}/package.json" || { echo "onnxruntime-web: package.json is not 1.20.1" >&2; exit 1; }
echo "cvcpkg: installed onnxruntime-web 1.20.1 (wasm EP) -> ${DEST}"
