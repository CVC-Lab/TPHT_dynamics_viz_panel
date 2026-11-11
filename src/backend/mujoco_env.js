// Every call into MuJoCo's JavaScript bindings goes through this file, so
// binding differences between MuJoCo versions get fixed in one place. Each
// fallback path is recorded and shown in the page's self-test panel.
//
// The bindings are MuJoCo's official Embind build (what the @mujoco/mujoco npm
// package ships), built from source by cvcpkg's `mujoco` wasm package and
// copied to vendor/mujoco/ by tools/build_site.py. See the README.

import { BASE } from '../base.js';

export class BackendError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}

export const MUJOCO_URL = `${BASE}vendor/mujoco/mujoco.js`;

export async function loadMujoco() {
  let mod;
  try {
    mod = await import(MUJOCO_URL);
  } catch (e) {
    throw new BackendError('mujoco-missing',
      `Could not import ${MUJOCO_URL} (${e.message}). Build the site with cvcpkg ` +
      '(see README: "Build and run") and open it through "tpht-lab serve".');
  }
  const load = mod.default ?? mod.loadMujoco;
  if (typeof load !== 'function') throw new BackendError('mujoco-api', 'MuJoCo module has no default loader export');
  return await load();
}

const has = (obj, name) => obj && typeof obj[name] === 'function';

function ensureDir(mj, path) {
  try { mj.FS.mkdir(path); } catch { /* exists */ }
}

// asset: { kind: 'xml' | 'mjb', name, data: string | Uint8Array }
export function loadModel(mj, asset) {
  ensureDir(mj, '/working');
  const path = `/working/${asset.name}`;
  mj.FS.writeFile(path, asset.data);
  const M = mj.MjModel;
  if (asset.kind === 'xml') {
    if (has(M, 'from_xml_path')) return { model: M.from_xml_path(path), via: 'MjModel.from_xml_path' };
    return { model: M.from_xml_string(asset.data), via: 'MjModel.from_xml_string' };
  }
  // The binary loaders take a virtual file system as their second argument
  // (MjModel.from_binary_path(path, vfs) and MjModel.mj_loadModel(path, vfs)
  // in mujoco.d.ts); calling them with the path alone throws. An empty VFS
  // makes them read the file written to MEMFS above. A .mjb only loads in the
  // MuJoCo version that wrote it, so prefer exporting XML.
  const viaVfs = (name, fn) => {
    const vfs = new mj.MjVFS();
    try { return { model: fn(vfs), via: name }; } finally { vfs.delete?.(); }
  };
  if (has(M, 'from_binary_path')) return viaVfs('MjModel.from_binary_path', (vfs) => M.from_binary_path(path, vfs));
  if (has(M, 'mj_loadModel')) return viaVfs('MjModel.mj_loadModel', (vfs) => M.mj_loadModel(path, vfs));
  const statics = Object.getOwnPropertyNames(M).filter((k) => typeof M[k] === 'function').join(', ');
  throw new BackendError('mjb-unsupported',
    `This MuJoCo wasm build exposes no binary loader (MjModel statics: ${statics}). Export XML instead (see README).`);
}

export class Sim {
  // meta: exported model metadata (ids, profile, timestep). Missing fields fall back to name lookups.
  constructor(mj, model, meta = {}) {
    this.mj = mj;
    this.model = model;
    this.meta = meta;
    this.data = new mj.MjData(model);
    this.scratch = Array.from({ length: 5 }, () => new mj.MjData(model));
    this.ghost = new mj.MjData(model);
    this.paths = {};
    this.nv = model.nv;
    this.nu = model.nu;
    this.torso = meta.torso_body_id ?? this.nameToId('body', 'torso');
    const gear = model.actuator_gear; // nu x 6
    this.gear = Array.from({ length: this.nu }, (_, i) => gear[i * 6]);
    // Generalized actuation map: 3 unactuated root coordinates, then diag(gear).
    this.B = new Float64Array(this.nv * this.nu);
    for (let i = 0; i < this.nu; i++) this.B[(this.nv - this.nu + i) * this.nu + i] = this.gear[i];
    this.frictionBase = Float64Array.from({ length: model.ngeom }, (_, g) => model.geom_friction[g * 3]);
    this.pushUntil = -Infinity;
    this.legacyStep = meta.legacy_step ?? true;
  }

  nameToId(kind, name) {
    const mj = this.mj;
    const objType = mj.mjtObj?.[`mjOBJ_${kind.toUpperCase()}`]?.value;
    if (has(mj, 'mj_name2id') && objType !== undefined) {
      const id = mj.mj_name2id(this.model, objType, name);
      if (id >= 0) return id;
    }
    if (has(this.model, kind)) return this.model[kind](name).id;
    throw new BackendError('name-lookup', `Cannot resolve ${kind} "${name}"; add it to the exported metadata.`);
  }

  dispose() {
    for (const d of [this.data, this.ghost, ...this.scratch]) d.delete?.();
    this.model.delete?.();
  }

  // ---- state ---------------------------------------------------------------
  forward(d, action) {
    if (action) d.ctrl.set(action);
    this.mj.mj_forward(this.model, d);
    this.mj.mj_subtreeVel(this.model, d);
  }

  copyInto(dst, src) {
    const mj = this.mj;
    if (this.paths.copy !== 'manual' && has(mj, 'mj_copyData')) {
      mj.mj_copyData(dst, this.model, src);
      this.paths.copy = 'mj_copyData';
      return dst;
    }
    // Fallback keeps every input that mj_forward reads, including the solver warm start.
    this.paths.copy = 'manual';
    dst.time = src.time;
    for (const k of ['qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'qfrc_applied', 'xfrc_applied', 'mocap_pos', 'mocap_quat']) {
      if (src[k] && dst[k] && src[k].length) dst[k].set(src[k]);
    }
    return dst;
  }

  setState(q, v, ctrl) {
    const d = this.data;
    this.mj.mj_resetData(this.model, d);
    d.qpos.set(q);
    d.qvel.set(v);
    if (ctrl) d.ctrl.set(ctrl);
    this.pushUntil = -Infinity;
    // After mj_forward the split-step invariant (step1 done) holds.
    this.forward(d);
  }

  // One control hold: dm_control legacy split step (mj_step2 then mj_step1).
  stepHold(action, substeps) {
    const { mj, model } = this;
    const d = this.data;
    d.ctrl.set(action);
    for (let k = 0; k < substeps; k++) {
      if (d.time >= this.pushUntil) this.clearPush();
      if (this.legacyStep) { mj.mj_step2(model, d); mj.mj_step1(model, d); }
      else mj.mj_step(model, d);
    }
    mj.mj_subtreeVel(model, d);
  }

  // ---- perturbations ------------------------------------------------------
  // Planar push on the torso COM: world-frame force (fx, fz) in N and pitch torque ty in N m.
  push(fx, fz, ty, duration) {
    const x = this.data.xfrc_applied, b = this.torso * 6;
    x[b] = fx; x[b + 1] = 0; x[b + 2] = fz; x[b + 3] = 0; x[b + 4] = ty; x[b + 5] = 0;
    this.pushUntil = this.data.time + duration;
  }

  clearPush() {
    const x = this.data.xfrc_applied, b = this.torso * 6;
    for (let i = 0; i < 6; i++) x[b + i] = 0;
    this.pushUntil = -Infinity;
  }

  pushActive() { return this.data.time < this.pushUntil; }

  // Scales the sliding friction of every geom. Scaling all geoms keeps the
  // pairwise result consistent under both max-mixing and geom priority.
  setFrictionScale(s) {
    const f = this.model.geom_friction;
    for (let g = 0; g < this.frictionBase.length; g++) f[g * 3] = this.frictionBase[g] * s;
  }

  // ---- mechanics ------------------------------------------------------------
  fullMass(d) {
    const { mj, model } = this;
    const n = this.nv;
    const tryPath = (name, fn) => {
      if (this.paths.mass && this.paths.mass !== name) return null;
      try {
        const out = fn();
        if (out) this.paths.mass = name;
        return out;
      } catch (e) {
        if (this.paths.mass === name) throw e;
        return null;
      }
    };
    // DoubleBuffer's constructor takes an element count (new DoubleBuffer(n),
    // or DoubleBuffer.FromArray(array)), and its accessor is GetView(), capital
    // G. The original new DoubleBuffer(array) / buf.getView() failed on every
    // path, so the mass-matrix self-test and every reference controller that
    // needs M (all of them) stopped here.
    const viaBuffer = (call, size = n * n) => {
      const buf = new mj.DoubleBuffer(size);
      try { call(buf); return Float64Array.from(buf.GetView()); } finally { buf.delete(); }
    };
    return (
      tryPath('mj_fullM(m, d, M)', () => has(mj, 'mj_fullM') && viaBuffer((b) => mj.mj_fullM(model, d, b))) ||
      tryPath('mj_fullM(m, M, qM)', () => has(mj, 'mj_fullM') && viaBuffer((b) => mj.mj_fullM(model, b, d.qM))) ||
      tryPath('mj_mulM columns', () => {
        if (!has(mj, 'mj_mulM')) return null;
        const M = new Float64Array(n * n);
        for (let j = 0; j < n; j++) {
          const e = new Array(n).fill(0); e[j] = 1;
          const col = viaBuffer((b) => mj.mj_mulM(model, d, b, e), n);
          for (let i = 0; i < n; i++) M[i * n + j] = col[i];
        }
        return M;
      }) ||
      (() => { throw new BackendError('mass-matrix', 'No working mass-matrix binding (mj_fullM / mj_mulM).'); })()
    );
  }

  torsoState(d) {
    const t = this.torso;
    return {
      qpos: d.qpos, qvel: d.qvel,
      torsoPos: [d.xpos[t * 3], d.xpos[t * 3 + 1], d.xpos[t * 3 + 2]],
      torsoLinvelX: d.subtree_linvel[t * 3],
      torsoComX: d.subtree_com[t * 3],
    };
  }

  // ---- drawing --------------------------------------------------------------
  // Robot geoms projected onto the x-z plane: [x1, z1, x2, z2, radius] per geom.
  segments(d) {
    const m = this.model, out = [];
    for (let g = 0; g < m.ngeom; g++) {
      if (m.geom_bodyid[g] === 0) continue;
      const type = m.geom_type[g];
      const px = d.geom_xpos[g * 3], pz = d.geom_xpos[g * 3 + 2];
      const R = d.geom_xmat, r = g * 9;
      const s0 = m.geom_size[g * 3], s1 = m.geom_size[g * 3 + 1];
      if (type === 3 || type === 5) { // capsule, cylinder: local z axis
        const ax = R[r + 2] * s1, az = R[r + 8] * s1;
        out.push(px - ax, pz - az, px + ax, pz + az, s0);
      } else if (type === 6) { // box: draw along local x
        const ax = R[r] * s0, az = R[r + 6] * s0;
        out.push(px - ax, pz - az, px + ax, pz + az, m.geom_size[g * 3 + 2]);
      } else { // sphere, ellipsoid, mesh: a dot
        out.push(px, pz, px, pz, s0);
      }
    }
    return Float32Array.from(out);
  }

  ghostSegments(q) {
    this.ghost.qpos.set(q);
    this.mj.mj_kinematics(this.model, this.ghost);
    return this.segments(this.ghost);
  }

  // ---- self-tests -----------------------------------------------------------
  selfTest(expect = {}) {
    const results = [];
    const add = (name, ok, detail) => results.push({ name, ok: !!ok, detail });
    const m = this.model;
    try {
      if (expect.dims) add('Model dimensions', m.nq === expect.dims[0] && m.nv === expect.dims[1] && m.nu === expect.dims[2],
        `nq=${m.nq} nv=${m.nv} nu=${m.nu}, expected ${expect.dims.join('/')}`);
      if (expect.timestep) add('Physics timestep', Math.abs(m.opt.timestep - expect.timestep) < 1e-12,
        `${m.opt.timestep} s, expected ${expect.timestep} s`);
      if (expect.euler) add('Euler integrator', m.opt.integrator === 0, `integrator=${m.opt.integrator}`);
      const cr = m.actuator_ctrlrange;
      add('Normalized actuator bounds', Array.from(cr).every((x, i) => Math.abs(x - (i % 2 ? 1 : -1)) < 1e-12),
        `ctrlrange=${Array.from(cr).map((x) => x.toFixed(2)).join(' ')}`);

      const probe = this.copyInto(this.scratch[0], this.data);
      const ctrl = Array.from({ length: this.nu }, (_, i) => -0.4 + (0.8 * i) / Math.max(1, this.nu - 1));
      this.forward(probe, ctrl);
      let gearErr = 0;
      for (let r = 0; r < this.nv; r++) {
        let s = 0;
        for (let c = 0; c < this.nu; c++) s += this.B[r * this.nu + c] * ctrl[c];
        gearErr = Math.max(gearErr, Math.abs(probe.qfrc_actuator[r] - s));
      }
      add('Linear gear map (qfrc_actuator = B ctrl)', gearErr < 1e-9, `max error ${gearErr.toExponential(2)}`);
      add('State copy', true, this.paths.copy);

      const M = this.fullMass(probe);
      let asym = 0, balance = 0, scale = 1;
      for (let i = 0; i < this.nv; i++) {
        let s = 0;
        for (let j = 0; j < this.nv; j++) {
          asym = Math.max(asym, Math.abs(M[i * this.nv + j] - M[j * this.nv + i]));
          s += M[i * this.nv + j] * probe.qacc[j];
        }
        const f = probe.qfrc_actuator[i] + probe.qfrc_passive[i] - probe.qfrc_bias[i]
          + probe.qfrc_constraint[i] + probe.qfrc_applied[i];
        scale = Math.max(scale, Math.abs(f));
        balance = Math.max(balance, Math.abs(s - f));
      }
      add(`Mass matrix via ${this.paths.mass}`, asym < 1e-9 && balance / scale < 1e-8,
        `asymmetry ${asym.toExponential(2)}, force balance ${(balance / scale).toExponential(2)} (relative)`);
    } catch (e) {
      add('Binding self-test', false, e.message);
    }
    return results;
  }
}
