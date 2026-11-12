# cvcpkg/recipes/tpht-backend/build-wasm.ps1 — Windows-host counterpart of build-wasm.sh.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\..\_common\env-wasm.ps1"   # emsdk activation, Invoke-CvcWasmCMakeBuild

$roots = (@($env:CVC_DEPS_PREFIX, $env:CVC_BUILD_PREFIX) | Where-Object { $_ }) -join ';'
Invoke-CvcWasmCMakeBuild -SourceDir (Join-Path $env:CVC_SOURCE_DIR 'cpp') -ExtraArgs @(
    "-DCMAKE_FIND_ROOT_PATH=$roots",
    "-DCMAKE_PREFIX_PATH=$roots",
    '-DTPHT_BUILD_TESTS=OFF')

# The module carries MuJoCo and nlohmann-json inside it, so the bundle carries their texts.
$lic = Join-Path $env:CVC_INSTALL_DIR 'share\licenses\tpht-backend'
New-Item -ItemType Directory -Force -Path $lic | Out-Null
Copy-Item (Join-Path $env:CVC_SOURCE_DIR 'LICENSE') (Join-Path $lic 'LICENSE')
foreach ($pkg in 'mujoco', 'nlohmann-json') {
    foreach ($p in @($env:CVC_DEPS_PREFIX, $env:CVC_BUILD_PREFIX) | Where-Object { $_ }) {
        $from = Join-Path $p "share\licenses\$pkg"
        if (Test-Path $from) {
            Copy-Item -Recurse -Force $from (Join-Path $lic $pkg)
            break
        }
    }
}

foreach ($f in 'tpht_backend.js', 'tpht_backend.wasm', 'build-info.json') {
    if (-not (Test-Path (Join-Path $env:CVC_INSTALL_DIR "share\tpht-backend\wasm\$f"))) { throw "tpht-backend: $f was not installed" }
}
Write-Output "cvcpkg: installed tpht-backend (wasm32) -> $env:CVC_INSTALL_DIR\share\tpht-backend\wasm"
