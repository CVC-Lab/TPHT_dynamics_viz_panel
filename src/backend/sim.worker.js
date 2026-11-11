// Runs in a Web Worker so long MPC decisions never block the page.
// Protocol: see src/backend/protocol.md.
import { loadMujoco, loadModel, Sim, BackendError } from './mujoco_env.js';
import { NativeController } from '../controllers/native.js';
import { OnnxPolicy } from '../controllers/onnx_policy.js';
import { DEMO_POLICY, SineGait } from '../controllers/demo_gait.js';
import { privilegedScan } from '../controllers/gait.js';
import { maxAbsDiff } from '../controllers/linalg.js';
import { loadTraceIndex, loadTrace } from './traces.js';
import { fetchAsset, setLocalAssets } from '../base.js';

const traceGet = (path) => fetchAsset(`reference/traces/${path}`);
const S = {
  mj: null, sim: null, section: null, manifest: null, caseId: 'F', meta: {}, gait: null,
  controller: null, policy: null, method: null, running: false, timeScale: 1, target: 3,
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

// ---------------------------------------------------------------- init --------
async function init({ section, caseId }) {
  stop();
  S.section = section;
  S.caseId = caseId ?? S.caseId;
  post('status', { text: 'Loading MuJoCo WebAssembly…' });
  if (!S.mj) S.mj = await loadMujoco();
  S.sim?.dispose();
  S.sim = null;
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
  const { model, via } = loadModel(S.mj, asset);
  S.sim = new Sim(S.mj, model, S.meta);
  const tests = [{ name: 'Model loader', ok: true, detail: via }];
  tests.push(...S.sim.selfTest(section === 'reference'
    ? { dims: [9, 9, 6], euler: true, timestep: S.meta.timestep }
    : { timestep: S.meta.timestep }));
  post('ready', {
    section, caseId: S.caseId,
    selfTests: tests,
    profile: S.meta.profile ?? null,
    planes: S.meta.planes ?? [0],
    gaitLoaded: !!S.gait,
    traces: S.traces?.list ?? [],
    // Exported CT-RL policies first, then the built-in demo gait (ctrl only).
    policies: section === 'ctrl' ? [...(S.manifest.policies ?? []), DEMO_POLICY] : (S.manifest.policies ?? []),
    target: S.target,
    timestep: S.sim.model.opt.timestep,
  });
}

// --------------------------------------------------------------- reset --------
async function reset({ method, clock, ghostMethod, initial }) {
  stop();
  const sim = S.sim;
  if (!sim) throw new BackendError('not-ready', 'Model is not loaded yet.');
  S.method = method; S.clock = clock ?? S.clock; S.decision = 0;
  S.ghostMethod = ghostMethod ?? null;
  S.ghost = S.ghostMethod && S.traces
    ? await loadTrace(traceGet, S.traces, S.caseId, S.clock, S.ghostMethod) : null;

  // Initial state: first row of a recorded trace (reference) or model default (ctrl).
  if (S.section === 'reference') {
    const src = S.ghost ?? (S.traces && await loadTrace(traceGet, S.traces, S.caseId, S.clock,
      S.traces.list.find((t) => t.case === S.caseId && t.clock === S.clock)?.method));
    if (initial === 'sample' || !src) {
      const sample = await fetchJSON(`reference/${S.manifest.cases[S.caseId].sample}`);
      const p = sample.pipeline_input;
      sim.setState(p.state_backend_q, p.state_backend_v, p.previous_action);
    } else {
      const row = src.row(0);
      sim.setState(row.q, row.v, row.action);
    }
    // Row k+1's hold follows row k's state, so live decision k uses holds[k + 1].
    S.schedule = S.clock === 'irregular' && src ? src.holds.subarray(1) : null;
    if (!S.gait) throw new BackendError('gait-missing',
      'No gait.json for this case. Native controllers need the saved gait from final.pt (export_web_assets.py gait).');
    S.controller = new NativeController(sim, S.gait, method);
    S.policy = null;
  } else {
    // dm_control randomizes the start pose; here every reset uses the model's qpos0.
    S.mj.mj_resetData(sim.model, sim.data);
    sim.forward(sim.data);
    if (method === DEMO_POLICY.id) {
      S.policy = new SineGait(sim.nu);
    } else {
      const entry = (S.manifest.policies ?? []).find((p) => p.id === method);
      if (!entry) throw new BackendError('policy-missing', `No exported policy "${method}" in ctrl/manifest.json.`);
      const meta = await fetchJSON(`ctrl/${entry.meta}`);
      const onnx = await fetchAsset(`ctrl/${entry.onnx}`);
      if (!onnx.ok) throw new BackendError('asset-missing', `Missing assets/ctrl/${entry.onnx}.`);
      S.policy = await OnnxPolicy.create(new Uint8Array(await onnx.arrayBuffer()), meta);
    }
    S.controller = null;
  }
  S.x0 = sim.data.qpos[0];
  S.stats = { n: 0, time: 0, sqErr: 0, sat: 0, decMs: 0, reward: 0, falls: 0, plans: 0 };
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

async function decide() {
  const sim = S.sim, d = sim.data;
  const hold = holdFor(S.decision);
  const substeps = Math.max(1, Math.round(hold / sim.model.opt.timestep));
  let action, diag;
  if (S.controller) {
    const scan = privilegedScan(d.qpos[0], S.meta.profile ?? { x: [-1e3, 1e3], z: [0, 0] });
    action = S.controller.act(scan, S.target, hold, d.time);
    diag = S.controller.diagnostics;
  } else {
    action = await S.policy.act(d);
    diag = S.policy.diagnostics;
  }
  sim.stepHold(action, substeps);
  const st = sim.torsoState(d);
  const h = substeps * sim.model.opt.timestep;
  const s = S.stats;
  s.n++; s.time += h;
  s.sqErr += h * (st.torsoLinvelX - S.target) ** 2;
  s.sat += h * (diag.action_saturation ?? 0);
  s.decMs += diag.decision_ms ?? 0;
  s.reward += h * Math.min(1, Math.max(0, st.torsoLinvelX / 10)); // dm_control cheetah-run reward rate
  if (diag.planned) s.plans++;
  S.decision++;
  S.lastAction = action;
  S.lastDiag = diag;
}

function frame(force = false) {
  const now = performance.now();
  if (!force && now - S.lastFrame < 33) return;
  S.lastFrame = now;
  const sim = S.sim, d = sim.data, st = sim.torsoState(d), s = S.stats;
  let ghost = null, ghostX = null;
  if (S.ghost) {
    const row = S.ghost.rowAt(S.ghost.t0 + d.time);
    if (row) { ghost = sim.ghostSegments(row.q); ghostX = row.q[0]; }
  }
  const wall = (now - S.wallStart) / 1000;
  post('frame', {
    t: d.time,
    segs: sim.segments(d),
    ghost, ghostX,
    torsoX: st.torsoPos[0],
    speed: st.torsoLinvelX,
    target: S.target,
    action: S.lastAction ? Array.from(S.lastAction) : null,
    push: sim.pushActive() ? Array.from(d.xfrc_applied.slice(sim.torso * 6, sim.torso * 6 + 6)) : null,
    diag: pickDiag(S.lastDiag),
    stats: s && s.time > 0 ? {
      decisions: s.n,
      distance: d.qpos[0] - S.x0,
      speedRmse: Math.sqrt(s.sqErr / s.time),
      saturation: s.sat / s.time,
      decisionMs: s.decMs / s.n,
      reward: s.reward,
      plans: s.plans,
      realtime: S.running && wall > 0 ? (d.time - S.simStart) / wall : null,
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
      while (S.sim.data.time < wallTarget && performance.now() < budgetEnd) {
        await decide();
        if (S.sim.data.time >= S.maxTime) { S.running = false; post('status', { text: `Episode ended at ${S.maxTime} s.` }); break; }
      }
    } catch (e) { S.running = false; fail(e); }
    frame();
    await new Promise((r) => setTimeout(r, 0));
  }
  frame(true);
  post('running', { running: false });
}

function start() {
  if (S.running || !S.sim || (!S.controller && !S.policy)) return;
  S.running = true;
  S.wallStart = performance.now();
  S.simStart = S.sim.data.time;
  post('running', { running: true });
  loop(++S.loopToken);
}

function stop() { S.running = false; S.loopToken++; }

// --------------------------------------------------------- golden check -------
// Replays validation/reference_samples/<clock>/<case>_transition.json:
// 1) recorded action through the loaded model (model + stepping convention),
// 2) acceleration target and fitted action (needs gait.json).
async function golden({ clock = 'fixed' }) {
  stop();
  const sim = S.sim, mj = S.mj;
  const path = S.manifest.cases[S.caseId].sample.replace(/^samples\/(fixed|irregular)/, `samples/${clock}`);
  const sample = await fetchJSON(`reference/${path}`);
  const p = sample.pipeline_input, dec = sample.decision, rec = sample.received_transition;
  const out = [];
  sim.setState(p.state_backend_q, p.state_backend_v, p.previous_action);
  const action = Float64Array.from(dec.action_normalized);
  sim.stepHold(action, rec.physics_substeps);
  out.push({ name: 'Transition replay q (max abs)', value: maxAbsDiff(sim.data.qpos, rec.q), tol: 1e-6 });
  out.push({ name: 'Transition replay v (max abs)', value: maxAbsDiff(sim.data.qvel, rec.v), tol: 1e-4 });

  if (S.gait) {
    sim.setState(p.state_backend_q, p.state_backend_v, p.previous_action);
    const ctl = new NativeController(sim, S.gait, dec.method === 'mujoco_mpc' ? 'mujoco_ls' : dec.method);
    const d = sim.copyInto(sim.scratch[0], sim.data);
    const fit = ctl.fit(d, p.scan, ctl.mean, dec.diagnostics.phase, p.target_speed_m_per_s, dec.method === 'mujoco_fd_ls', true);
    out.push({ name: 'Acceleration target (max abs)', value: maxAbsDiff(fit.diagnostics.acceleration_target, dec.diagnostics.acceleration_target), tol: 1e-5 });
    out.push({ name: `Fitted action, ${dec.method} (max abs)`, value: maxAbsDiff(fit.action, dec.action_normalized), tol: 1e-4 });
    out.push({ name: 'Full-contact acceleration (max abs)', value: maxAbsDiff(fit.diagnostics.acceleration_full_contact, dec.diagnostics.acceleration_full_contact), tol: 1e-3 });
  } else {
    out.push({ name: 'Controller checks', value: null, tol: null, note: 'Skipped: export gait.json from final.pt first.' });
  }
  mj.mj_resetData(sim.model, sim.data);
  post('golden', { caseId: S.caseId, clock, method: dec.method, results: out });
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
      case 'step': if (!S.running && (S.controller || S.policy)) { await decide(); frame(true); } break;
      case 'set':
        if (msg.target !== undefined) S.target = msg.target;
        if (msg.timeScale !== undefined) { S.timeScale = msg.timeScale; S.wallStart = performance.now(); S.simStart = S.sim?.data.time ?? 0; }
        if (msg.maxTime !== undefined) S.maxTime = msg.maxTime;
        if (msg.ctrlDt) Object.assign(S.ctrlDt, msg.ctrlDt);
        if (msg.friction !== undefined && S.sim) { S.frictionScale = msg.friction; S.sim.setFrictionScale(msg.friction); }
        break;
      case 'push': S.sim?.push(msg.fx, msg.fz, msg.ty, msg.duration); frame(true); break;
      case 'golden': await golden(msg); break;
      default: throw new Error(`Unknown message ${msg.type}`);
    }
  } catch (e) { fail(e); }
};
