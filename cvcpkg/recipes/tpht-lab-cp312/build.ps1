# cvcpkg/recipes/tpht-lab-cp312/build.ps1 — Windows counterpart of build.sh (same contract).
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\..\_common\python-wheel.ps1"

$py = Get-CvcPythonExe
Write-Output "tpht-lab-cp312: building with $py"

if ($env:CVC_BUILD_PREFIX) {
    $sp = Join-Path $env:CVC_BUILD_PREFIX 'Lib\site-packages'
    $env:PYTHONPATH = if ($env:PYTHONPATH) { "$sp;$env:PYTHONPATH" } else { $sp }
}

# Build from a copy of just the package files, so the checkout gets no build/ or egg-info.
$root = if ($env:CVC_BUILD_DIR) { $env:CVC_BUILD_DIR } else { Join-Path ([IO.Path]::GetTempPath()) "tpht-lab-$PID" }
$work = Join-Path $root 'tpht-lab-src'
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Force -Path $work | Out-Null
Copy-Item -Recurse (Join-Path $env:CVC_SOURCE_DIR 'tpht_lab') (Join-Path $work 'tpht_lab')
foreach ($f in 'pyproject.toml', 'README.md', 'LICENSE') { Copy-Item (Join-Path $env:CVC_SOURCE_DIR $f) $work }
Get-ChildItem -Path $work -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force

$wheelhouse = Join-Path $root 'wheelhouse'
if (Test-Path $wheelhouse) { Remove-Item -Recurse -Force $wheelhouse }
New-Item -ItemType Directory -Force -Path $wheelhouse | Out-Null
& $py -m pip wheel --no-build-isolation --no-deps --no-index --no-cache-dir --wheel-dir $wheelhouse $work
if ($LASTEXITCODE -ne 0) { throw "tpht-lab-cp312: pip wheel failed ($LASTEXITCODE)" }
$wheel = Get-ChildItem -Path $wheelhouse -Filter '*.whl' -File | Select-Object -First 1
if (-not $wheel) { throw "tpht-lab-cp312: no wheel produced under $wheelhouse" }
Write-Output "tpht-lab-cp312: built $($wheel.Name)"

& $py -m pip install --no-index --no-deps --no-compile --ignore-installed --prefix $env:CVC_INSTALL_DIR $wheel.FullName
if ($LASTEXITCODE -ne 0) { throw "tpht-lab-cp312: pip install failed ($LASTEXITCODE)" }

$env:PYTHONPATH = ''
Invoke-CvcPythonCheck 'import tpht_lab, tpht_lab.cli, tpht_lab.serve, tpht_lab.export, tpht_lab.dm_control_src; assert tpht_lab.__version__ == ''0.1.0'''
