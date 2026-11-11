# Page ↔ worker messages

Page → worker
- `init {section: 'reference'|'ctrl', caseId, backend: 'cpp'|'js'}` load the backend (C++ WebAssembly by default), model, metadata, gait, trace index; runs self-tests
- `reset {method, clock, ghostMethod, initial: 'trace'|'sample'}` build controller or ONNX policy, set initial state
- `start`, `pause`, `step` run in real time (scaled), stop, or advance one control decision
- `set {target?, timeScale?, maxTime?, friction?, ctrlDt?}` live parameters
- `push {fx, fz, ty, duration}` world-frame force on the torso COM (N) and pitch torque (N m)
- `golden {clock}` replay `reference_samples/<clock>/<case>_transition.json`

Worker → page
- `status {text}`, `error {code, message}`, `running {running}`
- `ready {backend, selfTests, profile, planes, gaitLoaded, traces, policies, target, timestep}` (`backend`: the one running; the first self-test names it)
- `frame {t, segs, ghost, torsoX, speed, target, action, push, diag, stats}` at most ~30 Hz
- `golden {caseId, clock, method, results: [{name, value, tol}]}`

`segs` is a Float32Array of `[x1, z1, x2, z2, radius]` per robot geom, projected on the x–z plane.
