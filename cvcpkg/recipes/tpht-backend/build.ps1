# cvcpkg/recipes/tpht-backend/build.ps1 — Windows counterpart of build.sh: MSVC, Ninja, the
# googletest suite, then install.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\..\_common\env-windows.ps1"   # MSVC environment, $cmakeBuildType

$src = Join-Path $env:CVC_SOURCE_DIR 'cpp'
# The dynamic CRT, like the mujoco and googletest DLLs this links against.
& cmake -G Ninja -S $src -B $env:CVC_BUILD_DIR `
    "-DCMAKE_INSTALL_PREFIX=$env:CVC_INSTALL_DIR" `
    "-DCMAKE_BUILD_TYPE=$cmakeBuildType" `
    '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>DLL' `
    '-DCMAKE_C_COMPILER=cl' '-DCMAKE_CXX_COMPILER=cl' `
    '-DTPHT_BUILD_TESTS=ON'
if ($LASTEXITCODE -ne 0) { throw 'cmake configure failed' }
& cmake --build $env:CVC_BUILD_DIR -j $env:CVC_JOBS
if ($LASTEXITCODE -ne 0) { throw 'cmake build failed' }

# mujoco.dll and the gtest DLLs live in the prefixes' bin directories.
$bins = @($env:CVC_DEPS_PREFIX, $env:CVC_BUILD_PREFIX) | Where-Object { $_ } | ForEach-Object { Join-Path $_ 'bin' }
$env:PATH = (($bins + @($env:PATH)) -join ';')
& ctest --test-dir $env:CVC_BUILD_DIR --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'tests failed' }

& cmake --install $env:CVC_BUILD_DIR
if ($LASTEXITCODE -ne 0) { throw 'cmake install failed' }
$lic = Join-Path $env:CVC_INSTALL_DIR 'share\licenses\tpht-backend'
New-Item -ItemType Directory -Force -Path $lic | Out-Null
Copy-Item (Join-Path $env:CVC_SOURCE_DIR 'LICENSE') (Join-Path $lic 'LICENSE')

& (Join-Path $env:CVC_INSTALL_DIR 'bin\tpht-sim.exe') version
if ($LASTEXITCODE -ne 0) { throw 'tpht-sim does not run' }
Write-Output "cvcpkg: installed tpht-backend (native) -> $env:CVC_INSTALL_DIR"
