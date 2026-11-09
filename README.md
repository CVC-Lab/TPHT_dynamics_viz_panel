# Cheetah baseline lab

A lab-internal browser tool for debugging the TPHT cheetah baselines. MuJoCo runs as WebAssembly in a Web Worker. You can push the robot, change friction and target speed, step one decision at a time, and overlay the recorded Python run as a dashed ghost.

The page has three tabs:

- **Reference controllers** runs ports of `mujoco_ls`, `mujoco_fd_ls` and `mujoco_mpc` from `mujoco_reference/controller.py`. The learned arms are ghost-only until their models are exported.
- **CT-RL baselines** runs CT-SAC/SAC actors exported to ONNX on the dm_control cheetah.
- **Explainers** embeds archify diagrams of the algorithms and code.

Live numbers are for debugging. Benchmark results come from the Python pipeline; see "Known differences" below.

## Setup

```bash
npm install          # also copies ONNX Runtime wasm files to public/ort
npm test             # controller ports vs SciPy/NumPy fixtures
npm run dev          # open the printed URL
```

`@mujoco/mujoco` is pinned to `3.17.0` because the bundle's `.mjb` models were compiled with MuJoCo 3.17.0, and binary models only load in the same version. If `ref-model` prints a different version, change the pin to match.

## Deploying to GitHub Pages

The app is fully static: MuJoCo and ONNX Runtime run as WebAssembly in the browser, and navigation uses `#hash` links, so it needs no server and no 404 fallback page.

1. Commit `package-lock.json` after your first `npm install`. The workflow uses `npm ci`.
2. In the repository, open Settings, then Pages, and set Source to **GitHub Actions**.
3. Push to `main`. `.github/workflows/pages.yml` installs, runs `npm test`, builds with the right base path, and deploys `dist/`.

Before pushing, you can test the exact Pages layout locally (replace the name with your repository's):

```bash
BASE_PATH=/rl-baseline-lab/ npm run build && npm run preview
```

### What ends up public

A Pages site is public even when the repository is private, unless your organization uses GitHub Enterprise Cloud with private Pages. Everything committed under `public/` is published. By default `.gitignore` keeps these out of the repository, and therefore off the site:

- `public/assets/reference/models/` (the compiled benchmark models)
- `public/assets/reference/gait/` (derived from `final.pt`)
- `public/assets/ctrl/` (CT-RL models and policies)

Lab members load those from their own machine with **Use your exported assets folder** in the Run panel. They pick their local `public/assets` folder; the page reads it in the tab and nothing is uploaded. Local files take precedence over deployed ones, including the manifests.

If your group decides some of these are fine to publish, remove the matching lines from `.gitignore`. The workflow log lists the published `assets` folders on every run, so you can check what went out.

### Static-hosting limits that apply

- **No custom headers.** Pages cannot send the cross-origin isolation headers that multithreaded WebAssembly needs, so the app uses the single-threaded MuJoCo and ONNX Runtime builds.
- **Size.** Published sites are limited to 1 GB, and individual files in a repository to 100 MB. The current assets are about 4 MB, plus about 12 MB per `.mjb` model.
- **Explainers are public too.** archify HTML in `public/explain/` describes your code structure.

## Exporting assets

Traces, samples and results are already in `public/assets/reference/`. The remaining steps need your Python environments. `B` is the path to `TPHT_CT_RL_Benchmark/`.

```bash
# MuJoCo 3.17.0: copies models, writes ids and the ground profile (via mj_ray)
python3 tools/export_web_assets.py ref-model --bundle $B

# Bundle code + torch: saved gait, joint limits, CEM bounds and options.
# Native controllers need this, because even mujoco_ls tracks the saved gait.
python3 tools/export_web_assets.py gait --bundle $B --case F --checkpoint /path/to/sensor/.../F/final.pt
python3 tools/export_web_assets.py gait --bundle $B --case J --checkpoint /path/to/sensor/.../J/final.pt

# dm_control: flattened cheetah XML for the CT-RL section
python3 tools/export_web_assets.py ctrl-model

# torch (+ onnxruntime for the numerical check): one per algorithm
python3 tools/export_web_assets.py ctrl-policy --algo ct_sac --label CT-SAC \
  --checkpoint /path/to/ct_sac.pt --ctrl-repo $B/../ct-rl-master
```

Re-run `traces`, `samples` or `results` with `--bundle $B` after a new benchmark run. Each step updates the section's `manifest.json`.

`ctrl-policy` supports `StochasticActor` checkpoints (CT-SAC, SAC). It infers the layer sizes from the state dict, loads them with `strict=True` so a wrong structure fails loudly, and compares ONNX output with PyTorch when `onnxruntime` is installed. TD3-style deterministic actors need a small wrapper first.

## First run: what to check

1. **Binding self-tests** (bottom left). These confirm the model dimensions, the Euler integrator, the gear map, and that the mass matrix balances forces. A failure names the binding call that differs in your `@mujoco/mujoco` version. All such calls live in `src/backend/mujoco_env.js`.
2. **Golden check**. This replays `reference_samples/<clock>/<case>_transition.json`:
   - The recorded action is stepped through the browser model and compared with the recorded next state. This tests the model and the split-step convention, and needs no checkpoint.
   - With `gait.json` exported, it also compares the acceleration target, the fitted action and the full-contact acceleration with Python's values.

   The replay starts from q and v only, without the solver warm start, so expect small contact differences, not zero.
3. **Recorded Python episode** shows the matching `results.json` row next to the live run.

## Verification status

| Component | Status |
|---|---|
| `bvls.js` vs `scipy.optimize.lsq_linear(method='bvls')` | 400 problems, max difference 2.3e-13 |
| `gait.js` references and target vs NumPy transcription | 200 cases, below 1e-10 |
| Binary traces | Round-trip test passes; recorded panel matches `report.md` |
| Page layout and error states | Checked in headless Chrome |
| Local assets folder overriding deployed files | Checked in headless Chrome |
| MuJoCo WASM calls, golden-check tolerances, ONNX export | Not yet run: needs `npm install` and your Python environments |
| Vite build with a base path, Pages workflow | Not yet run: needs `npm install`; try the local `BASE_PATH` build first |

## Known differences from the Python benchmark

- **Terrain scan.** The live controllers get a privileged scan read from the exported ground profile. The causal environment observer (`environment_belief/observer.py`) is not ported yet.
- **MPC noise.** The CEM noise uses a seeded JS generator, so `mujoco_mpc` matches the Python arm in distribution, not candidate by candidate.
- **Pushes and friction changes are seen immediately by native controllers.** They read MuJoCo's current state and model, consistent with their "privileged" label.
- **Case J.** J draws its course per episode, so the J model in `reference_samples` may not be the course of the J traces. The ghost can drift from the terrain.
- **CT-RL clock and start pose.** The irregular clock draws holds uniformly in [2, 30] ms; the CT-RL time-grid generator is not ported. Episodes start from the model default pose rather than dm_control's randomized one.

## Explainers (archify)

Each module in `src/explain/modules.js` has a prompt naming real files and functions. To generate one:

1. Install the skill for your agent: `npx skills add tt-a1i/archify -g`.
2. Open the benchmark repository, paste the prompt from the Explainers tab, and let archify validate and deliver.
3. Save the HTML as `public/explain/<id>.html`. The tab embeds it automatically.

Archify's source cards link to a commit in a public repository. Until your code is public, the diagrams work but without clickable source links.

## Layout

```
src/main.js                 page controller (tabs, controls, readouts)
src/base.js                 deploy base path and local-folder asset overrides
src/ui/viewport.js          side view and speed strip chart
src/backend/sim.worker.js   simulation loop, clocks, perturbations, golden check
src/backend/mujoco_env.js   all @mujoco/mujoco calls, fallbacks, self-tests
src/backend/traces.js       binary trace loader
src/backend/protocol.md     page/worker messages
src/controllers/            native.js, bvls.js, gait.js, onnx_policy.js, registry.js
src/explain/modules.js      archify module prompts
tools/                      exporter, fixture generator, ORT wasm copier
tests/                      node --test suites and fixtures
.github/workflows/pages.yml GitHub Pages build and deploy
public/assets/              exported assets (models, gait and ctrl are git-ignored)
```
