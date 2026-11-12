// Runs in a Web Worker so long MPC decisions never block the page.
// Protocol: see src/backend/protocol.md. The model, its state and the
// controllers live in a backend (backends.js): the C++ core compiled to
// WebAssembly by default, or the JavaScript port. This file keeps the clock,
// the messages, the recorded traces, the ONNX policies and the statistics.
import { BackendError } from './mujoco_env.js';
import { BACKENDS, backendKind } from './backends.js';
import { OnnxPolicy } from '../controllers/onnx_policy.js';
import { DEMO_POLICY } from '../controllers/demo_gait.js';
import { loadTraceIndex, loadTrace } from './traces.js';
import { fetchAsset, setLocalAssets } from '../base.js';

const traceGet = (path) => fetchAsset(`reference/traces/${path}`);
const S = {
  be: null, section: null, manifest: null, caseId: 'F', meta: {}, gait: null,
  policy: null, method: null, running: false, timeScale: 1, target: 3,
  clock: 'fixed', schedule: null, decision: 0, ghost: null, ghostMethod: null, traces: null,
  x0: 0, stats: null, wallStart: 0, simStart: 0, maxTime: 12, lastFrame: 0, frictionScale: 1,
  ctrlDt: { fixed: 0.01, min: 0.002, max: 0.03 }, loopToken: 0,
};

const post = (type, payload = {}) => self.postMessage({ type, ...payload });
const fail = (e) => post('error', { code: e.code ?? 'error', message: e.message });

async function fetchJSON(path, optional = false) {
  const r = await fetchAsset(path);
  if (!r.ok) {
    if (optional) return null;
    throw new BackendError('asset-missing', `Missing assets/${path}. Export it (tools/export_web_assets.py) or load your local assets folder.`);
  }
  return r.json();
}

async function fetchModel(section, spec) {
  const r = await fetchAsset(`${section}/${spec.file}`);
  if (!r.ok) throw new BackendError('asset-missing', `Missing model assets/${section}/${spec.file}. Export it or load your local assets folder.`);
  const data = spec.kind === 'xml' ? await r.text() : new Uint8Array(await r.arrayBuffer());
  return { kind: spec.kind, name: spec.file.split('/').pop(), data };
}

// The backend the page asked for, created once and reused. If the C++ module is
// missing, the JavaScript backend runs instead and the self-test panel says so.
async function selectBackend(kind) {
  kind = backendKind(kind);
  if (S.be?.kind === kind) return null;
  S.be?.dispose();
  S.be = null;
  try {
    S.be = await BACKENDS[kind].create();
    return null;
  } catch (e) {
    if (kind === 'js') throw e;
    S.be = await BACKENDS.js.create();
    return `${e.message} Running the JavaScript backend instead.`;
  }
}

// ---------------------------------------------------------------- init --------
async function init({ section, caseId, backend }) {
  stop();
  S.section = section;
  S.caseId = caseId ?? S.caseId;
  post('status', { text: backendKind(backend) === 'cpp' ? 'Loading the C++ backend (WebAssembly)…' : 'Loading MuJoCo WebAssembly…' });
  const fallback = await selectBackend(backend);
  const be = S.be;
  S.policy = null;
  S.manifest = await fetchJSON(`${section}/manifest.json`);
  let modelSpec, metaPath;
  if (section === 'reference') {
    const c = S.manifest.cases[S.caseId];
    if (!c) throw new BackendError('case-missing', `Case ${S.caseId} is not in reference/manifest.json`);
    modelSpec = c.model; metaPath = c.meta;
    S.gait = c.gait ? await fetchJSON(`reference/${c.gait}`, true) : null;
    S.target = S.manifest.target_speed ?? 3;
    S.traces = await loadTraceIndex(traceGet).catch(() => null);
  } else {
    modelSpec = S.manifest.model; metaPath = S.manifest.meta;
    S.gait = null; S.traces = null;
  }
  S.meta = metaPath ? (await fetchJSON(`${section}/${metaPath}`, true)) ?? {} : {};
  post('status', { text: 'Compiling model…' });
  const asset = await fetchModel(section, modelSpec);
  const { selfTests } = be.load(asset, S.meta, section === 'reference'
    ? { dims: [9, 9, 6], euler: true, timestep: S.meta.timestep }
    : { timestep: S.meta.timestep });
  be.setGait(S.gait);
  const tests = [
    { name: 'Backend', ok: !fallback, detail: `${be.name}, MuJoCo ${be.mujocoVersion}${fallback ? `. ${fallback}` : ''}` },
    ...selfTests,
  ];
  post('ready', {
    section, caseId: S.caseId,
    backend: be.kind,
    selfTests: tests,
    profile: S.meta.profile ?? null,
    planes: S.meta.planes ?? [0],
    gaitLoaded: !!S.gait,
    traces: S.traces?.list ?? [],
    // Exported CT-RL policies first, then the built-in demo gait (ctrl only).
    policies: section === 'ctrl' ? [...(S.manifest.policies ?? []), DEMO_POLICY] : (S.manifest.policies ?? []),
    target: S.target,
    timestep: be.timestep,
  });
}

// --------------------------------------------------------------- reset --------
async function reset({ method, clock, ghostMethod, initial }) {
  stop();
  const be = S.be;
  if (!be?.loaded) throw new BackendError('not-ready', 'Model is not loaded yet.');
  S.method = method; S.clock = clock ?? S.clock; S.decision = 0;
  S.ghostMethod = ghostMethod ?? null;
  S.ghost = S.ghostMethod && S.traces
    ? await loadTrace(traceGet, S.traces, S.caseId, S.clock, S.ghostMethod) : null;
  S.policy = null;

  // Initial state: first row of a recorded trace (reference) or model default (ctrl).
  if (S.section === 'reference') {
    const src = S.ghost ?? (S.traces && await loadTrace(traceGet, S.traces, S.caseId, S.clock,
      S.traces.list.find((t) => t.case === S.caseId && t.clock === S.clock)?.method));
    if (initial === 'sample' || !src) {
      const sample = await fetchJSON(`reference/${S.manifest.cases[S.caseId].sample}`);
      const p = sample.pipeline_input;
      be.setState(p.state_backend_q, p.state_backend_v, p.previous_action);
    } else {
      const row = src.row(0);
      be.setState(row.q, row.v, row.action);
    }
    // Row k+1's hold follows row k's state, so live decision k uses holds[k + 1].
    S.schedule = S.clock === 'irregular' && src ? src.holds.subarray(1) : null;
    if (!be.hasGait) throw new BackendError('gait-missing',
      'No gait.json for this case. Native controllers need the saved gait from final.pt (export_web_assets.py gait).');
    be.useNative(method);
  } else {
    // dm_control randomizes the start pose; here every reset uses the model's qpos0.
    be.resetData();
    if (method === DEMO_POLICY.id) {
      be.useSineGait();
    } else {
      be.clearController();
      const entry = (S.manifest.policies ?? []).find((p) => p.id === method);
      if (!entry) throw new BackendError('policy-missing', `No exported policy "${method}" in ctrl/manifest.json.`);
      const meta = await fetchJSON(`ctrl/${entry.meta}`);
      const onnx = await fetchAsset(`ctrl/${entry.onnx}`);
      if (!onnx.ok) throw new BackendError('asset-missing', `Missing assets/ctrl/${entry.onnx}.`);
      S.policy = await OnnxPolicy.create(new Uint8Array(await onnx.arrayBuffer()), meta);
    }
  }
  S.x0 = be.qpos()[0];
  S.stats = { n: 0, time: 0, sqErr: 0, sat: 0, decMs: 0, reward: 0, falls: 0, plans: 0 };
  S.lastAction = null;
  S.lastDiag = null;
  frame(true);
}

// ------------------------------------------------------------- stepping -------
function holdFor(decision) {
  if (S.section === 'reference') {
    if (S.clock === 'irregular' && S.schedule) return S.schedule[decision % S.schedule.length];
    return 0.01;
  }
  if (S.clock === 'irregular') return S.ctrlDt.min + Math.random() * (S.ctrlDt.max - S.ctrlDt.min);
  return S.ctrlDt.fixed;
}

const canStep = () => !!(S.be?.loaded && (S.be.hasController || S.policy));

async function decide() {
  const be = S.be;
  const hold = holdFor(S.decision);
  const dt = be.timestep;
  const substeps = Math.max(1, Math.round(hold / dt));
  let action, diag;
  if (S.policy) {
    action = await S.policy.act({ qpos: be.qpos(), qvel: be.qvel(), time: be.time });
    diag = S.policy.diagnostics;
  } else {
    ({ action, diag } = be.decide(S.target, hold));
  }
  be.stepHold(action, substeps);
  const st = be.torso();
  const h = substeps * dt;
  const s = S.stats;
  s.n++; s.time += h;
  s.sqErr += h * (st.linvelX - S.target) ** 2;
  s.sat += h * (diag.action_saturation ?? 0);
  s.decMs += diag.decision_ms ?? 0;
  s.reward += h * Math.min(1, Math.max(0, st.linvelX / 10)); // dm_control cheetah-run reward rate
  if (diag.planned) s.plans++;
  S.decision++;
  S.lastAction = action;
  S.lastDiag = diag;
}

function frame(force = false) {
  const now = performance.now();
  if (!force && now - S.lastFrame < 33) return;
  S.lastFrame = now;
  const be = S.be, t = be.time, st = be.torso(), s = S.stats;
  let ghost = null, ghostX = null;
  if (S.ghost) {
    const row = S.ghost.rowAt(S.ghost.t0 + t);
    if (row) { ghost = be.ghostSegments(row.q); ghostX = row.q[0]; }
  }
  const wall = (now - S.wallStart) / 1000;
  post('frame', {
    t,
    segs: be.segments(),
    ghost, ghostX,
    torsoX: st.x,
    speed: st.linvelX,
    target: S.target,
    action: S.lastAction ? Array.from(S.lastAction) : null,
    push: be.pushActive() ? be.pushForce() : null,
    diag: pickDiag(S.lastDiag),
    stats: s && s.time > 0 ? {
      decisions: s.n,
      distance: be.qpos()[0] - S.x0,
      speedRmse: Math.sqrt(s.sqErr / s.time),
      saturation: s.sat / s.time,
      decisionMs: s.decMs / s.n,
      reward: s.reward,
      plans: s.plans,
      realtime: S.running && wall > 0 ? (t - S.simStart) / wall : null,
    } : null,
  });
}

function pickDiag(d) {
  if (!d) return null;
  const keep = ['weighted_acceleration_residual', 'contact_linearization_error', 'action_saturation',
    'ls_iterations', 'planner_cost', 'planner_invalid_fraction', 'frequency_hz', 'phase', 'decision_ms'];
  return Object.fromEntries(keep.filter((k) => d[k] !== undefined).map((k) => [k, d[k]]));
}

async function loop(token) {
  while (S.running && token === S.loopToken) {
    const budgetEnd = performance.now() + 20;
    const wallTarget = S.simStart + ((performance.now() - S.wallStart) / 1000) * S.timeScale;
    try {
      while (S.be.time < wallTarget && performance.now() < budgetEnd) {
        await decide();
        if (S.be.time >= S.maxTime) { S.running = false; post('status', { text: `Episode ended at ${S.maxTime} s.` }); break; }
      }
    } catch (e) { S.running = false; fail(e); }
    frame();
    await new Promise((r) => setTimeout(r, 0));
  }
  frame(true);
  post('running', { running: false });
}

function start() {
  if (S.running || !canStep()) return;
  S.running = true;
  S.wallStart = performance.now();
  S.simStart = S.be.time;
  post('running', { running: true });
  loop(++S.loopToken);
}

function stop() { S.running = false; S.loopToken++; }

// --------------------------------------------------------- golden check -------
// Replays validation/reference_samples/<clock>/<case>_transition.json in the
// backend: the recorded action through the model, then (with gait.json) the
// acceleration target and the fitted action.
async function golden({ clock = 'fixed' }) {
  stop();
  if (!S.be?.loaded) throw new BackendError('not-ready', 'Model is not loaded yet.');
  const path = S.manifest.cases[S.caseId].sample.replace(/^samples\/(fixed|irregular)/, `samples/${clock}`);
  const sample = await fetchJSON(`reference/${path}`);
  const results = S.be.golden(sample);
  post('golden', { caseId: S.caseId, clock, method: sample.decision.method, results });
}

// ------------------------------------------------------------- messages -------
self.onmessage = async ({ data: msg }) => {
  try {
    switch (msg.type) {
      case 'local-assets': setLocalAssets(msg.entries); break;
      case 'init': await init(msg); break;
      case 'reset': await reset(msg); break;
      case 'start': start(); break;
      case 'pause': stop(); break;
      case 'step': if (!S.running && canStep()) { await decide(); frame(true); } break;
      case 'set':
        if (msg.target !== undefined) S.target = msg.target;
        if (msg.timeScale !== undefined) { S.timeScale = msg.timeScale; S.wallStart = performance.now(); S.simStart = S.be?.loaded ? S.be.time : 0; }
        if (msg.maxTime !== undefined) S.maxTime = msg.maxTime;
        if (msg.ctrlDt) Object.assign(S.ctrlDt, msg.ctrlDt);
        if (msg.friction !== undefined && S.be?.loaded) { S.frictionScale = msg.friction; S.be.setFrictionScale(msg.friction); }
        break;
      case 'push': if (S.be?.loaded) { S.be.push(msg.fx, msg.fz, msg.ty, msg.duration); frame(true); } break;
      case 'golden': await golden(msg); break;
      default: throw new Error(`Unknown message ${msg.type}`);
    }
  } catch (e) { fail(e); }
};
