// The JavaScript backend: the controller ports in src/controllers/ over MuJoCo's
// own Embind build (vendor/mujoco/). Same interface as cpp_backend.js; the worker
// (sim.worker.js) drives whichever one the page picked. See backends.js.
import { loadMujoco, loadModel, Sim, BackendError } from './mujoco_env.js';
import { NativeController } from '../controllers/native.js';
import { SineGait } from '../controllers/demo_gait.js';
import { privilegedScan } from '../controllers/gait.js';
import { maxAbsDiff } from '../controllers/linalg.js';

const FLAT = { x: [-1e3, 1e3], z: [0, 0] };
let mujocoPromise = null;

export class JsBackend {
  static async create() {
    mujocoPromise ??= loadMujoco().catch((e) => { mujocoPromise = null; throw e; });
    return new JsBackend(await mujocoPromise);
  }

  constructor(mj) {
    this.mj = mj;
    this.kind = 'js';
    this.name = 'JavaScript port (MuJoCo Embind bindings)';
    this.sim = null;
    this.meta = {};
    this.gait = null;
    this.controller = null;
    this.sine = null;
  }

  get mujocoVersion() { return typeof this.mj.mj_versionString === 'function' ? this.mj.mj_versionString() : '?'; }
  get loaded() { return !!this.sim; }
  get hasGait() { return !!this.gait; }
  get hasController() { return !!(this.controller || this.sine); }

  // asset: { kind: 'xml' | 'mjb', name, data }; expect: see Sim.selfTest.
  load(asset, meta, expect) {
    this.clearController();
    this.sim?.dispose();
    this.sim = null;
    this.meta = meta ?? {};
    const { model, via } = loadModel(this.mj, asset);
    this.sim = new Sim(this.mj, model, this.meta);
    return { via, selfTests: [{ name: 'Model loader', ok: true, detail: via }, ...this.sim.selfTest(expect)] };
  }

  setGait(gait) { this.gait = gait ?? null; this.controller = null; }

  resetData() {
    this.mj.mj_resetData(this.sim.model, this.sim.data);
    this.sim.pushUntil = -Infinity;
    this.sim.forward(this.sim.data);
  }

  setState(q, v, ctrl) { this.sim.setState(q, v, ctrl); }

  useNative(method) {
    if (!this.gait) throw new BackendError('gait-missing',
      'No gait.json for this case. Native controllers need the saved gait from final.pt (export_web_assets.py gait).');
    this.sine = null;
    this.controller = new NativeController(this.sim, this.gait, method);
  }

  useSineGait() { this.controller = null; this.sine = new SineGait(this.sim.nu); }
  clearController() { this.controller = null; this.sine = null; }

  // The active controller's action for the current state; does not step.
  decide(target, hold) {
    const d = this.sim.data;
    if (this.controller) {
      const scan = privilegedScan(d.qpos[0], this.meta.profile ?? FLAT);
      const action = this.controller.act(scan, target, hold, d.time);
      return { action, diag: this.controller.diagnostics };
    }
    if (this.sine) return { action: this.sine.act(d), diag: this.sine.diagnostics };
    throw new BackendError('no-controller', 'No controller is selected; reset first.');
  }

  stepHold(action, substeps) { this.sim.stepHold(action, substeps); }

  get time() { return this.sim.data.time; }
  get timestep() { return this.sim.model.opt.timestep; }
  get nu() { return this.sim.nu; }
  qpos() { return this.sim.data.qpos; }
  qvel() { return this.sim.data.qvel; }
  torso() {
    const st = this.sim.torsoState(this.sim.data);
    return { x: st.torsoPos[0], z: st.torsoPos[2], linvelX: st.torsoLinvelX };
  }

  segments() { return this.sim.segments(this.sim.data); }
  ghostSegments(q) { return this.sim.ghostSegments(q); }

  push(fx, fz, ty, duration) { this.sim.push(fx, fz, ty, duration); }
  pushActive() { return this.sim.pushActive(); }
  pushForce() {
    const b = this.sim.torso * 6;
    return Array.from(this.sim.data.xfrc_applied.slice(b, b + 6));
  }
  setFrictionScale(s) { this.sim.setFrictionScale(s); }

  // Replays validation/reference_samples/<clock>/<case>_transition.json:
  // 1) recorded action through the loaded model (model + stepping convention),
  // 2) acceleration target and fitted action (needs gait.json).
  golden(sample) {
    const sim = this.sim;
    const p = sample.pipeline_input, dec = sample.decision, rec = sample.received_transition;
    const out = [];
    sim.setState(p.state_backend_q, p.state_backend_v, p.previous_action);
    sim.stepHold(Float64Array.from(dec.action_normalized), rec.physics_substeps);
    out.push({ name: 'Transition replay q (max abs)', value: maxAbsDiff(sim.data.qpos, rec.q), tol: 1e-6 });
    out.push({ name: 'Transition replay v (max abs)', value: maxAbsDiff(sim.data.qvel, rec.v), tol: 1e-4 });
    if (this.gait) {
      sim.setState(p.state_backend_q, p.state_backend_v, p.previous_action);
      const ctl = new NativeController(sim, this.gait, dec.method === 'mujoco_mpc' ? 'mujoco_ls' : dec.method);
      const d = sim.copyInto(sim.scratch[0], sim.data);
      const fit = ctl.fit(d, p.scan, ctl.mean, dec.diagnostics.phase, p.target_speed_m_per_s, dec.method === 'mujoco_fd_ls', true);
      out.push({ name: 'Acceleration target (max abs)', value: maxAbsDiff(fit.diagnostics.acceleration_target, dec.diagnostics.acceleration_target), tol: 1e-5 });
      out.push({ name: `Fitted action, ${dec.method} (max abs)`, value: maxAbsDiff(fit.action, dec.action_normalized), tol: 1e-4 });
      out.push({ name: 'Full-contact acceleration (max abs)', value: maxAbsDiff(fit.diagnostics.acceleration_full_contact, dec.diagnostics.acceleration_full_contact), tol: 1e-3 });
    } else {
      out.push({ name: 'Controller checks', value: null, tol: null, note: 'Skipped: export gait.json from final.pt first.' });
    }
    this.mj.mj_resetData(sim.model, sim.data);
    return out;
  }

  dispose() {
    this.clearController();
    this.sim?.dispose();
    this.sim = null;
  }
}
