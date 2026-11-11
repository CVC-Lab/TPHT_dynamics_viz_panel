# Cheetah baseline lab

A lab-internal browser tool for debugging the TPHT cheetah baselines. MuJoCo runs as WebAssembly in a Web Worker. You can push the robot, change friction and target speed, step one decision at a time, and overlay the recorded Python run as a dashed ghost.

The page has three tabs:

- **Reference controllers** runs ports of `mujoco_ls`, `mujoco_fd_ls` and `mujoco_mpc` from `mujoco_reference/controller.py`. The learned arms are ghost-only until their models are exported.
- **CT-RL baselines** runs CT-SAC/SAC actors exported to ONNX on the dm_control cheetah. Until a policy is exported it offers a built-in open-loop sine gait, so you can see MuJoCo running. That gait is not a baseline.
- **Explainers** embeds archify diagrams of the algorithms and code.

Live numbers are for debugging. Benchmark results come from the Python pipeline; see "Known differences" below.

## Build and run

Everything comes from [cvcpkg](https://cvcpkg.org) packages. There is no node, npm or bundler: the app is plain ES modules.

- **From the cvcpkg.org catalog:**
  - MuJoCo 4.0's WebAssembly build;
  - CPython 3.12, numpy and MuJoCo's Python package;
  - a headless Chromium for the tests.
- **From this repository's recipes (`cvcpkg/recipes/`):**
  - `onnxruntime-web`: ONNX Runtime Web 1.20.1, from the pinned npm tarball, which cvcpkg fetches and hash-checks itself;
  - `tpht-viz-panel`: the site;
  - `tpht-lab-cp312`: the native host.

### 1. Get the cvcpkg CLI

Download the release for your OS from <https://github.com/cy-pca/cvcpkg/releases> (2.5.6 or newer), check it against its `.sha256` file, and put it on `PATH` as `cvcpkg`.

### 2. Install the dependencies into a prefix

Run this from the repository root. Any directory works as the prefix; `deps` is git-ignored.

```bash
cvcpkg install python312 numpy-cp312 mujoco-core-cp312 setuptools-cp312 --prefix deps
cvcpkg install mujoco --platform wasm --arch wasm32 --link static --prefix deps
```

Add `playwright-chromium-headless-shell` to the first line on Linux x86_64 for `tpht-lab test`. On Windows also add `powershell`, because cvcpkg runs the Windows build script with PowerShell 7.

### 3. Build the recipes

```bash
cvcpkg build onnxruntime-web --platform any --local --recipes-dir cvcpkg/recipes --prefix deps --build-prefix deps
cvcpkg build tpht-viz-panel  --platform any --local --recipes-dir cvcpkg/recipes --prefix deps --build-prefix deps
cvcpkg build tpht-lab-cp312  --local --recipes-dir cvcpkg/recipes --prefix deps --build-prefix deps
```

What this gives you:

- **The site:** `deps/share/tpht-viz-panel/`, the deployable static site.
- **`tpht-lab`:** the native host, in `deps/bin/` (`deps\Scripts\tpht-lab.exe` on Windows).

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

The app is fully static: MuJoCo and ONNX Runtime run as WebAssembly in the browser, and navigation uses `#hash` links, so it needs no server and no 404 fallback page.

1. In the repository, open Settings, then Pages, and set Source to **GitHub Actions**.
2. Push to `main`. `.github/workflows/pages.yml` does the following on an Ubuntu runner, then deploys `share/tpht-viz-panel/`:
   - downloads the pinned cvcpkg CLI and checks its sha256;
   - runs steps 2 and 3 above;
   - runs `tpht-lab check` and `tpht-lab test`.

   Pull requests run the same build and tests without deploying.

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

- **No custom headers.** Pages cannot send the cross-origin isolation headers that multithreaded WebAssembly needs, so the app uses the single-threaded MuJoCo and ONNX Runtime builds.
- **Size.** Published sites are limited to 1 GB, and individual files in a repository to 100 MB. The site is about 28 MB, mostly MuJoCo's `.wasm` (10 MB) and ONNX Runtime's (11 MB). A binary model adds about 12 MB; an XML model adds a few KB.
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

1. **Binding self-tests** (bottom left). These confirm the model dimensions, the Euler integrator, the gear map, and that the mass matrix balances forces. A failure names the binding call that differs in your MuJoCo build. All such calls live in `src/backend/mujoco_env.js`. `tpht-lab check` runs the same checks natively, which separates "the asset is wrong" (both fail) from "the wasm binding call is wrong" (only the page fails).
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
| MuJoCo WASM calls (CT-RL tab, dm_control cheetah) | All self-tests pass in the page; `tpht-lab check` matches natively |
| Demo gait in the page vs native | Same distance at 12 s (about 14.7 m) |
| cvcpkg build (catalog deps + the three recipes), `tpht-lab check`/`test` | Checked on Windows x86_64; the Pages workflow runs it on Ubuntu |
| Reference tab with the benchmark models | Not run: the models are not in the repository (`.mjb` from MuJoCo 3.17 will not load in 4.0; export XML) |
| ONNX policies | Not run: no exported policy yet |

## Known differences from the Python benchmark

- **Terrain scan.** The live controllers get a privileged scan read from the exported ground profile. The causal environment observer (`environment_belief/observer.py`) is not ported yet.
- **MPC noise.** The CEM noise uses a seeded JS generator, so `mujoco_mpc` matches the Python arm in distribution, not candidate by candidate.
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

## Next: a C++ backend

The plan is a C++ simulation core on the cvcpkg toolchain and the Ariadne SDK, built both natively and to WebAssembly. It would replace `mujoco_env.js` and the controller ports with one implementation that the page and native tools share. This page and `tpht-lab` are the reference it has to match.

## Layout

```
index.html, src/main.js     page controller (tabs, controls, readouts)
src/base.js                 site base URL (from the module's own URL) and local-folder asset overrides
src/ui/viewport.js          side view and speed strip chart
src/backend/sim.worker.js   simulation loop, clocks, perturbations, golden check
src/backend/mujoco_env.js   all MuJoCo binding calls, fallbacks, self-tests
src/backend/traces.js       binary trace loader
src/backend/protocol.md     page/worker messages
src/controllers/            native.js, bvls.js, gait.js, onnx_policy.js, demo_gait.js, registry.js
src/explain/modules.js      archify module prompts
tests/                      in-browser test page (index.html, harness.js), suites and fixtures
tpht_lab/                   the native host: serve, check, test, export (pyproject.toml at the root)
tools/build_site.py         assembles the site from the repository and cvcpkg prefixes
tools/export_web_assets.py  the exporter, run from a checkout (code in tpht_lab/export.py)
tools/make_test_fixtures.py fixture generator (SciPy)
cvcpkg/recipes/             onnxruntime-web, tpht-viz-panel, tpht-lab-cp312, and cvcpkg's shared _common/
.github/workflows/pages.yml cvcpkg build, tests, GitHub Pages deploy
public/assets/              exported assets (benchmark models, gait and policies are git-ignored)
```
