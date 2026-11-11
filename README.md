# Cheetah baseline lab

A lab-internal browser tool for debugging the TPHT cheetah baselines. The simulation and the reference controllers are a C++ core (`cpp/`) on MuJoCo's C API. It runs as WebAssembly in a Web Worker in the page, and natively as `tpht-sim`. You can push the robot, change friction and target speed, step one decision at a time, and overlay the recorded Python run as a dashed ghost.

The page has three tabs:

- **Reference controllers** runs ports of `mujoco_ls`, `mujoco_fd_ls` and `mujoco_mpc` from `mujoco_reference/controller.py`. The learned arms are ghost-only until their models are exported.
- **CT-RL baselines** runs CT-SAC/SAC actors exported to ONNX on the dm_control cheetah. Until a policy is exported it offers a built-in open-loop sine gait, so you can see MuJoCo running. That gait is not a baseline.
- **Explainers** embeds archify diagrams of the algorithms and code.

Live numbers are for debugging. Benchmark results come from the Python pipeline; see "Known differences" below.

## Build and run

Everything comes from [cvcpkg](https://cvcpkg.org) packages. There is no node, npm or bundler: the app is plain ES modules.

- **From the cvcpkg.org catalog:**
  - MuJoCo 4.0, as its wasm32 static build and natively;
  - the C++ toolchain pieces: emsdk 5.0.7 (Emscripten), cmake, ninja, nlohmann-json and googletest;
  - CPython 3.12, numpy and MuJoCo's Python package;
  - a headless Chromium for the tests.
- **From this repository's recipes (`cvcpkg/recipes/`):**
  - `tpht-backend`: the C++ core, built as the page's WebAssembly module and natively as `tpht-sim`;
  - `onnxruntime-web`: ONNX Runtime Web 1.20.1, from the pinned npm tarball, which cvcpkg fetches and hash-checks itself;
  - `tpht-viz-panel`: the site;
  - `tpht-lab-cp312`: the native host.

Emscripten's `emcc` runs the node that ships inside cvcpkg's emsdk package for some of its own link steps. Nothing here installs node or npm, or runs a JavaScript package manager.

### 1. Install the cvcpkg CLI

Use the installer that cvcpkg.org serves; you don't need a checkout of the cvcpkg repository. It downloads the newest `cvcpkg` for your OS and checks it against the sha256 the registry recorded. Give it the directory to install the `cvcpkg` binary into:

Linux and macOS:

```bash
curl -fsSL https://cvcpkg.org/install.sh | sh -s -- --install-dir "$HOME/.local/bin"
```

Windows (PowerShell):

```powershell
& ([scriptblock]::Create((irm https://cvcpkg.org/install.ps1))) -InstallDir "$env:LOCALAPPDATA\cvcpkg"
```

The steps below run `cvcpkg` from `PATH`. The Windows installer adds the directory to your user `PATH`. On Linux and macOS, if the directory is not on `PATH`, the installer prints the `export PATH=...` line to add. Check the install with `cvcpkg --version`. To pin a release, add `--version 2.5.13` (`-Version 2.5.13` on Windows).

### 2. Install the dependencies into a prefix

Run this from the repository root. Any directory works as a prefix; `deps*` is git-ignored.

```bash
cvcpkg install python312 numpy-cp312 mujoco-core-cp312 setuptools-cp312 nlohmann-json cmake ninja --prefix deps
cvcpkg install mujoco --platform wasm --arch wasm32 --link static --prefix deps
cvcpkg install emsdk --prefix deps-emsdk
```

Add `playwright-chromium-headless-shell` to the first line on Linux x86_64 for `tpht-lab test`. On Windows also add `powershell`, because cvcpkg runs the Windows build scripts with PowerShell 7.

emsdk gets a prefix of its own because activating it removes every `PATH` entry under its root, and that would drop cmake and ninja.

### 3. Build the recipes

First the C++ backend's WebAssembly module, then the site and the host:

```bash
CVC_EMSDK_DIR="$PWD/deps-emsdk" PATH="$PWD/deps/bin:$PATH" \
  cvcpkg build tpht-backend --platform wasm --local --recipes-dir cvcpkg/recipes --prefix deps --build-prefix deps
cvcpkg build onnxruntime-web --platform any --local --recipes-dir cvcpkg/recipes --prefix deps --build-prefix deps
cvcpkg build tpht-viz-panel  --platform any --local --recipes-dir cvcpkg/recipes --prefix deps --build-prefix deps
cvcpkg build tpht-lab-cp312  --local --recipes-dir cvcpkg/recipes --prefix deps --build-prefix deps
```

On Windows, build the module from PowerShell, with `--host-platform windows` so cvcpkg picks the Windows build script:

```powershell
$env:CVC_EMSDK_DIR = "$PWD\deps-emsdk"; $env:PATH = "$PWD\deps\bin;$env:PATH"
cvcpkg build tpht-backend --platform wasm --host-platform windows --local --recipes-dir cvcpkg/recipes --prefix deps --build-prefix deps
```

What this gives you:

- **The site:** `deps/share/tpht-viz-panel/`, the deployable static site, with the C++ module in `vendor/tpht/`.
- **`tpht-lab`:** the native host, in `deps/bin/` (`deps\Scripts\tpht-lab.exe` on Windows).

The native C++ tools are optional; see "The C++ backend" below.

### 4. Run it

```bash
deps/bin/tpht-lab serve --open     # the page at http://127.0.0.1:8000/
deps/bin/tpht-lab check            # the page's binding self-tests, run natively on the same assets
deps/bin/tpht-lab test             # tests/index.html in headless Chromium
```

`tpht-lab serve` finds the installed site by itself. `--dir` serves another one, and `--isolate` adds the COOP/COEP headers that threaded WebAssembly would need.

To run the tests in your own browser instead, open `/tests/` on the served site. Elsewhere than Linux x86_64, point `tpht-lab test` at a Chromium-family browser:

```bash
tpht-lab test --browser "/path/to/chrome"     # or set TPHT_BROWSER
```

If the browser stops with "No usable sandbox", add `--no-sandbox`. That happens on Ubuntu 23.10 and later when AppArmor restricts unprivileged user namespaces, as on GitHub's ubuntu-24.04 runners, and the Pages workflow passes it for that reason.

### Working on the code

`src/` is served as it is. After an edit, rebuild the site, which is just a copy (about a second), and reload the page:

```bash
deps/bin/python3 tools/build_site.py --prefix deps --out build/site --with-tests
deps/bin/tpht-lab serve --dir build/site
```

On Windows the interpreter is `deps\python.exe`. From a checkout you can also run the host without installing it: `deps/bin/python3 -m tpht_lab ...`.

After a C++ edit, rebuild the module with plain CMake (emsdk's activation script, then the same toolchain file the recipe uses), install it next to the build, and point `build_site.py` at it:

```bash
source deps-emsdk/emsdk_env.sh
deps/bin/cmake -G Ninja -S cpp -B build/cpp-wasm -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake" \
  -DCMAKE_FIND_ROOT_PATH="$PWD/deps" -DCMAKE_PREFIX_PATH="$PWD/deps"
deps/bin/cmake --build build/cpp-wasm && deps/bin/cmake --install build/cpp-wasm --prefix build/cpp-wasm-install
deps/bin/python3 tools/build_site.py --prefix deps --prefix build/cpp-wasm-install --out build/site --with-tests
```

On Windows, run `deps-emsdk\emsdk_env.bat` in `cmd` instead of sourcing the script.

## The C++ backend

`cpp/` holds one implementation of everything the page simulates. It uses MuJoCo's C API and the nlohmann-json headers, and nothing else. It is built two ways from the same sources:

- **In the browser:** `tpht_backend.{js,wasm}`, an Embind module (4 MB; MuJoCo's own Embind build is 10 MB). The page's worker runs it by default. It is single-threaded, so GitHub Pages can serve it without cross-origin isolation headers.
- **Natively:** `libtpht_core`, its headers, and the `tpht-sim` tool.

What it contains, each piece ported from the JavaScript of the same name:

| C++ | Ported from | What it is |
|---|---|---|
| `sim.h` | `src/backend/mujoco_env.js` | Model loading (XML through a VFS, or `.mjb`), the legacy split step, pushes, friction, mass matrix, segments, self-tests |
| `native_controller.h` | `src/controllers/native.js` | `mujoco_ls`, `mujoco_fd_ls`, CEM `mujoco_mpc` |
| `bvls.h`, `gait.h`, `linalg.h` | `bvls.js`, `gait.js`, `linalg.js` | Bounded least squares, gait references and acceleration target, privileged scan |
| `rng.h` | `makeNormal` in `native.js` | The seeded MPC noise, in exact integer arithmetic, so both backends draw the same candidates |
| `backend.h` | `src/backend/sim.worker.js` | One model, its state and the active controller: what the worker and `tpht-sim` drive |

The page keeps the JavaScript port behind the same interface (`src/backend/backends.js`) for comparison. Pick it with the **Backend** menu in the Run panel, or with `?backend=js` in the URL. The first self-test names the backend that is running. If the C++ module is missing, the page falls back to the JavaScript port, and that self-test fails to say so. ONNX policies still run in ONNX Runtime Web in either case: the worker reads the state from the backend and steps the backend with the policy's action.

### Native build and `tpht-sim`

The native MuJoCo goes in a prefix of its own, because the native and wasm `mujoco` bundles both install `lib/cmake/mujoco`:

```bash
cvcpkg install mujoco cmake ninja nlohmann-json googletest --prefix deps-native
cvcpkg build tpht-backend --local --recipes-dir cvcpkg/recipes --prefix deps-native --build-prefix deps-native
```

On Linux, add `libglvnd` to the install line: MuJoCo's CMake package looks for OpenGL, and a server or CI image may have none. On Windows, add `powershell`. The recipe runs the googletest suite before it installs anything. Then:

```bash
deps-native/bin/tpht-sim check --assets public/assets          # binding self-tests on every model
deps-native/bin/tpht-sim run --assets public/assets --section ctrl --seconds 12
deps-native/bin/tpht-sim run --section reference --case F --method mujoco_mpc --seconds 3
deps-native/bin/tpht-sim golden --case F --clock fixed         # replay a reference sample
```

Every command takes `--json`. Without `--assets`, `tpht-sim` uses `public/assets` in the current directory, else `share/tpht-viz-panel/assets` in its own prefix. The `reference` and `golden` commands need the benchmark's models and `gait.json`, which are git-ignored (see "Exporting assets").

For iteration without the recipe:

```bash
deps-native/bin/cmake -G Ninja -S cpp -B build/cpp-native -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/deps-native"
deps-native/bin/cmake --build build/cpp-native && deps-native/bin/ctest --test-dir build/cpp-native
```

On Windows, run these from a Visual Studio x64 developer prompt, with `deps-native\bin` on `PATH` so the tests find `mujoco.dll`.

### How the backends were checked against each other

- **Native C++ against MuJoCo's Python package.** The C++ demo-gait rollout moves 5.134876807099926 m in 5 s. The same rollout through `import mujoco` (4.0.0) moves the same distance to 1e-15, when it steps as the page does: float32 controls, `mj_step2` then `mj_step1`.
- **C++ WebAssembly against the JavaScript port.** In the browser test page, both run 60 decisions of `mujoco_ls` and of `mujoco_fd_ls`, and 25 of `mujoco_mpc`, on the cheetah with `tests/fixtures/cheetah_gait.json`. Their actions are identical, with a difference of exactly 0, and so is the demo gait.
- **The numerics against SciPy and the Python formulas.** The scipy BVLS cases and the gait cases in `tests/fixtures/` pass natively (googletest) at the same tolerances as in the browser.
- **The MPC noise.** The seeded generator matches an independent Python port: exactly for the uniforms, and to 1e-15 for the normals, in both backends.

The reference cases themselves (models F and J) have not been run in either backend, because their models are not in the repository.

## Why the MuJoCo backend failed, and what changed

Four problems stopped the page. All were in `src/backend/mujoco_env.js`:

- **The mass-matrix buffer accessor.** The code called `buf.getView()`, but the Embind accessor is `GetView()`. It also built buffers with `new DoubleBuffer(array)`, but the constructor takes a length: `new DoubleBuffer(n)`, or `DoubleBuffer.FromArray`. The `mj_mulM` fallback also allocated n×n for an n-vector. So every mass-matrix path threw, which took down the mass-matrix self-test and every reference controller, since all of them need M.
- **The binary loaders.** `MjModel.from_binary_path` and `MjModel.mj_loadModel` take a virtual file system as their second argument (`mujoco.d.ts`). Without it they throw. `mj_loadModel` was also being called on the module instead of on `MjModel`.
- **Binary models are version-locked.** An `.mjb` only loads in the MuJoCo version that wrote it. The site now runs cvcpkg's MuJoCo 4.0.0, and the benchmark's `.mjb` files come from 3.17. `tpht-lab export ref-model` now writes flattened XML when the bundle has `*_transition.xml`, and XML loads in any version.
- **Absolute paths.** `index.html` loaded `/src/...` absolutely, which breaks a Pages project site. Paths are now relative to `src/base.js`, so the same build works at any base path.

The CT-RL tab had no model to load. It now ships dm_control 1.0.51's public cheetah: Apache-2.0, fetched by `tpht-lab export ctrl-model` from the dm_control repository at a pinned tag, with each file sha256-checked.

With those fixes, these all agree:

- the page's self-tests in WebAssembly;
- `tpht-lab check` with native MuJoCo, to the same digits (force balance 2.07e-16);
- the demo gait's distance: 14.7 m in 12 s, both natively and in the page.

## Deploying to GitHub Pages

The app is fully static: the C++ core, MuJoCo and ONNX Runtime run as WebAssembly in the browser, and navigation uses `#hash` links, so it needs no server and no 404 fallback page.

1. In the repository, open Settings, then Pages, and set Source to **GitHub Actions**.
2. Push to `main`. `.github/workflows/pages.yml` does the following on an Ubuntu runner, then deploys `share/tpht-viz-panel/`:
   - installs cvcpkg 2.5.13 with the step 1 installer;
   - runs steps 2 and 3 above, the C++ module included;
   - runs `tpht-lab check` and `tpht-lab test`.

   A second job builds the C++ backend natively on Ubuntu and Windows (the recipe runs the googletest suite), then runs `tpht-sim check` and a 12 s demo run. Pull requests run every build and test without deploying.

The site uses only relative URLs, so it works at `/<repo>/` without a base-path setting.

### What ends up public

A Pages site is public even when the repository is private, unless your organization uses GitHub Enterprise Cloud with private Pages. Everything committed under `public/` is published. By default `.gitignore` keeps these out of the repository, and therefore off the site:

- `public/assets/reference/models/`: the compiled benchmark models.
- `public/assets/reference/gait/`: derived from `final.pt`.
- `public/assets/ctrl/policies/`: the CT-RL policies.

The dm_control cheetah in `public/assets/ctrl/` is committed and published. It is public upstream, and its licence is beside it.

Lab members load the private files from their own machine with **Use your exported assets folder** in the Run panel. They pick their local `public/assets` folder, and the page reads it in the tab; nothing is uploaded. Local files take precedence over deployed ones, including the manifests.

If your group decides some of these are fine to publish, remove the matching lines from `.gitignore`. The workflow log lists the published `assets` folders on every run, so you can check what went out.

### Static-hosting limits that apply

- **No custom headers.** Pages cannot send the cross-origin isolation headers that multithreaded WebAssembly needs, so the app uses single-threaded builds: the C++ module, MuJoCo and ONNX Runtime.
- **Size.** Published sites are limited to 1 GB, and individual files in a repository to 100 MB. The site is about 33 MB: ONNX Runtime's `.wasm` (11 MB), MuJoCo's Embind build for the JavaScript port (10 MB) and the C++ module (4 MB). The page downloads only the backend it runs. A binary model adds about 12 MB; an XML model adds a few KB.
- **Explainers are public too.** archify HTML in `public/explain/` describes your code structure.

## Exporting assets

Traces, samples, results and the dm_control cheetah are already in `public/assets/`. The remaining steps read the benchmark bundle. `B` is the path to `TPHT_CT_RL_Benchmark/`.

You can run the exporter two ways:

- **Installed:** `tpht-lab export <step>`.
- **From a checkout:** `deps/bin/python3 tools/export_web_assets.py <step>`, which writes into the checkout's `public/assets`.

```bash
# MuJoCo (cvcpkg's, 4.0): models + ids + ground profile (via mj_ray).
# Writes flattened XML when the bundle has models/*_transition.xml (recommended: loads in any
# MuJoCo version); an .mjb only loads in the version that wrote it, and 4.0 cannot read 3.17's.
tpht-lab export ref-model --bundle $B

# Bundle code + torch (the benchmark's own environment): saved gait, joint limits, CEM bounds
# and options. Native controllers need this, because even mujoco_ls tracks the saved gait.
python3 tools/export_web_assets.py gait --bundle $B --case F --checkpoint /path/to/sensor/.../F/final.pt
python3 tools/export_web_assets.py gait --bundle $B --case J --checkpoint /path/to/sensor/.../J/final.pt

# dm_control cheetah for the CT-RL tab: --xml for your own copy, else an installed dm_control,
# else the pinned dm_control 1.0.51 files (downloaded once, sha256-checked, cached in
# ~/.cache/tpht-lab or $TPHT_LAB_CACHE).
tpht-lab export ctrl-model

# torch (+ onnxruntime for the numerical check): one per algorithm
python3 tools/export_web_assets.py ctrl-policy --algo ct_sac --label CT-SAC \
  --checkpoint /path/to/ct_sac.pt --ctrl-repo $B/../ct-rl-master
```

Re-run `traces`, `samples` or `results` with `--bundle $B` after a new benchmark run. Each step updates the section's `manifest.json`. Then rebuild the site.

`ctrl-policy` supports `StochasticActor` checkpoints (CT-SAC, SAC). It infers the layer sizes from the state dict, loads them with `strict=True` so a wrong structure fails loudly, and compares ONNX output with PyTorch when `onnxruntime` is installed. TD3-style deterministic actors need a small wrapper first.

The controller-test fixtures (`tests/fixtures/`) come from `tools/make_test_fixtures.py`, which needs SciPy:

```bash
cvcpkg install scipy-cp312 --prefix deps
deps/bin/python3 tools/make_test_fixtures.py
```

## First run: what to check

1. **Binding self-tests** (bottom left). The first line names the backend. The rest confirm the model dimensions, the Euler integrator, the gear map, and that the mass matrix balances forces. In the JavaScript port a failure names the binding call that differs in your MuJoCo build; all such calls live in `src/backend/mujoco_env.js`. `tpht-lab check` (Python) and `tpht-sim check` (C++) run the same checks natively, which separates "the asset is wrong" (everything fails) from "the browser build is wrong" (only the page fails).
2. **Golden check**. This replays `reference_samples/<clock>/<case>_transition.json`:
   - The recorded action is stepped through the browser model and compared with the recorded next state. This tests the model and the split-step convention, and needs no checkpoint.
   - With `gait.json` exported, it also compares the acceleration target, the fitted action and the full-contact acceleration with Python's values.

   The replay starts from q and v only, without the solver warm start, so expect small contact differences, not zero.
3. **Recorded Python episode** shows the matching `results.json` row next to the live run.

## Verification status

| Component | Status |
|---|---|
| `bvls.js` vs `scipy.optimize.lsq_linear(method='bvls')` | 400 problems, max difference 2.3e-13; passes in the browser test page |
| `gait.js` references and target vs NumPy transcription | 200 cases, below 1e-10; passes in the browser test page |
| Binary traces | Round-trip test passes; recorded panel matches `report.md` |
| MuJoCo WASM calls (CT-RL tab, dm_control cheetah) | All self-tests pass in the page, in both backends; `tpht-lab check` and `tpht-sim check` match natively (force balance 2.07e-16) |
| C++ core, native, vs MuJoCo's Python package | Demo gait identical to 1e-15 (5 s); 14 googletest tests, including the SciPy BVLS and gait fixtures |
| C++ WebAssembly vs the JavaScript port | Identical actions for `mujoco_ls`, `mujoco_fd_ls` and `mujoco_mpc` on the cheetah (difference exactly 0), and identical demo gait; in the browser test page |
| Demo gait in the page vs native | Same distance at 12 s (14.70 m natively; the page takes one decision more and shows 14.72 m) |
| cvcpkg build (catalog deps + the four recipes, wasm and native), `tpht-lab check`/`test` | Checked on Windows x86_64; the workflow runs the site on Ubuntu and the native build on Ubuntu and Windows |
| Reference tab with the benchmark models | Not run: the models are not in the repository (`.mjb` from MuJoCo 3.17 will not load in 4.0; export XML) |
| ONNX policies | Not run: no exported policy yet |

## Known differences from the Python benchmark

- **Terrain scan.** The live controllers get a privileged scan read from the exported ground profile. The causal environment observer (`environment_belief/observer.py`) is not ported yet.
- **MPC noise.** The CEM noise uses a seeded generator (mulberry32), not `torch.Generator`, so `mujoco_mpc` matches the Python arm in distribution, not candidate by candidate. The C++ core and the JavaScript port share the generator and draw the same candidates.
- **Pushes and friction changes are seen immediately by native controllers.** They read MuJoCo's current state and model, consistent with their "privileged" label.
- **Case J.** J draws its course per episode, so the J model in `reference_samples` may not be the course of the J traces. The ghost can drift from the terrain.
- **CT-RL clock and start pose.** The irregular clock draws holds uniformly in [2, 30] ms; the CT-RL time-grid generator is not ported. Episodes start from the model default pose rather than dm_control's randomized one.
- **MuJoCo version.** The benchmark ran MuJoCo 3.17; the page and `tpht-lab` run 4.0.0. Compare live numbers with that in mind.

## Explainers (archify)

Each module in `src/explain/modules.js` has a prompt naming real files and functions. To generate one:

1. Install the skill for your agent: `npx skills add tt-a1i/archify -g`. This is an authoring tool, separate from the build.
2. Open the benchmark repository, paste the prompt from the Explainers tab, and let archify validate and deliver.
3. Save the HTML as `public/explain/<id>.html`. The tab embeds it automatically.

Archify's source cards link to a commit in a public repository. Until your code is public, the diagrams work but without clickable source links.

## Next

- **An Ariadne host.** The core links only MuJoCo and nlohmann-json, so an Ariadne host can wrap the same code. The host would be a C++ program on libcvc's `cvc::cvc` / `cvc::cvcGL` from cvcpkg, with the simulation exposed to `.ari` documents through `register_function` and the robot drawn by a registered scene node. It would run natively (`arigl` style), and in the browser through libcvc's threaded wasm SDK. That build needs cross-origin isolation, which GitHub Pages cannot provide; `tpht-lab serve --isolate` can.
- **CT-RL policies natively.** `tpht-sim` runs the built-in gait only. An exported actor needs a small MLP evaluator in C++. libcvc's `coef_mlp` always applies softplus to its output, so it cannot evaluate a SAC actor as it is.
- **The benchmark's models.** Once XML models of cases F and J are exported, `tpht-sim golden` and the reference controllers can be compared with the Python run in both backends.

## Layout

```
index.html, src/main.js     page controller (tabs, controls, readouts)
src/base.js                 site base URL (from the module's own URL) and local-folder asset overrides
src/ui/viewport.js          side view and speed strip chart
src/backend/sim.worker.js   simulation loop, clocks, perturbations, golden check (drives a backend)
src/backend/backends.js     the backend interface; cpp_backend.js (the C++ module), js_backend.js (the port)
src/backend/mujoco_env.js   the JavaScript port's MuJoCo binding calls, fallbacks, self-tests
src/backend/traces.js       binary trace loader
src/backend/protocol.md     page/worker messages
src/controllers/            native.js, bvls.js, gait.js, onnx_policy.js, demo_gait.js, registry.js
src/explain/modules.js      archify module prompts
tests/                      in-browser test page (index.html, harness.js), suites and fixtures
cpp/                        the C++ core: include/tpht/, src/, wasm/bindings.cpp, tools/tpht_sim.cpp, tests/
tpht_lab/                   the native host: serve, check, test, export (pyproject.toml at the root)
tools/build_site.py         assembles the site from the repository and cvcpkg prefixes
tools/export_web_assets.py  the exporter, run from a checkout (code in tpht_lab/export.py)
tools/make_test_fixtures.py fixture generator (SciPy)
cvcpkg/recipes/             tpht-backend, onnxruntime-web, tpht-viz-panel, tpht-lab-cp312, and cvcpkg's shared _common/
.github/workflows/pages.yml cvcpkg builds (site, C++ native), tests, GitHub Pages deploy
public/assets/              exported assets (benchmark models, gait and policies are git-ignored)
```
