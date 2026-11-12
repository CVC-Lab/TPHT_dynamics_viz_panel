// The C++ backend: cpp/ (MuJoCo's C API plus the reference controllers) compiled
// to WebAssembly by cvcpkg's tpht-backend recipe and copied to vendor/tpht/ by
// tools/build_site.py. Same interface as js_backend.js; see backends.js.
import { BASE } from '../base.js';
import { BackendError } from './mujoco_env.js';

export const TPHT_URL = `${BASE}vendor/tpht/tpht_backend.js`;
let modulePromise = null;

function loadModule() {
  modulePromise ??= import(TPHT_URL)
    .then((m) => (m.default ?? m.loadTphtBackend)())
    .catch((e) => { modulePromise = null; throw e; });
  return modulePromise;
}

// C++ errors arrive as Error objects with the BackendError code in .code.
function call(fn) {
  try {
    return fn();
  } catch (e) {
    if (e instanceof BackendError) throw e;
    throw new BackendError(e?.code ?? 'backend', e?.message ?? String(e));
  }
}

export class CppBackend {
  static async create() {
    let mod;
    try {
      mod = await loadModule();
    } catch (e) {
      throw new BackendError('cpp-missing',
        `Could not load ${TPHT_URL} (${e.message}). Build the site with the tpht-backend wasm package (see README: "Build and run").`);
    }
    return new CppBackend(mod);
  }

  constructor(mod) {
    this.mod = mod;
    this.b = new mod.TphtBackend();
    this.kind = 'cpp';
    this.name = 'C++ core (WebAssembly)';
    this.isLoaded = false;
  }

  get mujocoVersion() { return this.mod.TphtBackend.mujocoVersion(); }
  get loaded() { return this.isLoaded; }
  get hasGait() { return this.b.hasGait(); }
  get hasController() { return this.b.hasController(); }

  load(asset, meta, expect) {
    this.isLoaded = false;
    const bytes = typeof asset.data === 'string' ? new TextEncoder().encode(asset.data) : asset.data;
    const out = JSON.parse(call(() => this.b.load(asset.kind, asset.name, bytes,
      JSON.stringify(meta ?? {}), JSON.stringify(expect ?? {}))));
    this.isLoaded = true;
    return out;
  }

  setGait(gait) { call(() => this.b.setGait(gait ? JSON.stringify(gait) : '')); }
  resetData() { call(() => this.b.resetData()); }
  setState(q, v, ctrl) { call(() => this.b.setState(q, v, ctrl ?? null)); }
  useNative(method, seed = 100000) { call(() => this.b.useNative(method, seed)); }
  useSineGait() { call(() => this.b.useSineGait()); }
  clearController() { this.b.clearController(); }

  decide(target, hold) {
    const action = call(() => this.b.decide(target, hold));
    return { action, diag: JSON.parse(this.b.diagnosticsJson()) };
  }

  stepHold(action, substeps) { call(() => this.b.stepHold(action, substeps)); }

  get time() { return this.b.time(); }
  get timestep() { return this.b.timestep(); }
  get nu() { return this.b.nu(); }
  qpos() { return this.b.qpos(); }
  qvel() { return this.b.qvel(); }
  torso() {
    const t = this.b.torso();
    return { x: t[0], z: t[1], linvelX: t[2] };
  }

  segments() { return this.b.segments(); }
  ghostSegments(q) { return call(() => this.b.ghostSegments(q)); }

  push(fx, fz, ty, duration) { call(() => this.b.push(fx, fz, ty, duration)); }
  pushActive() { return this.b.pushActive(); }
  pushForce() { return Array.from(this.b.pushForce()); }
  setFrictionScale(s) { call(() => this.b.setFrictionScale(s)); }

  golden(sample) { return JSON.parse(call(() => this.b.goldenJson(JSON.stringify(sample)))); }

  dispose() {
    this.b?.delete();
    this.b = null;
    this.isLoaded = false;
  }
}
