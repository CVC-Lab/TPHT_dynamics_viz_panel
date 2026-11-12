// The two simulation backends behind one interface, so the worker, the tests and
// the page can switch between them:
//   cpp  the C++ core (cpp/) compiled to WebAssembly: the default, and the
//        implementation the native tool tpht-sim shares
//   js   the JavaScript port over MuJoCo's own Embind build, kept for comparison
//
// Interface (both classes):
//   static async create()
//   kind, name, mujocoVersion, loaded, hasGait, hasController
//   load(asset, meta, expect) -> { via, selfTests }
//   setGait(gait | null), resetData(), setState(q, v, ctrl)
//   useNative(method), useSineGait(), clearController()
//   decide(target, hold) -> { action, diag }      the action for the current state
//   stepHold(action, substeps)
//   time, timestep, nu, qpos(), qvel(), torso() -> { x, z, linvelX }
//   segments(), ghostSegments(q), push(fx, fz, ty, duration), pushActive(), pushForce()
//   setFrictionScale(s), golden(sample) -> [{ name, value, tol, note? }], dispose()
import { CppBackend } from './cpp_backend.js';
import { JsBackend } from './js_backend.js';

export const BACKENDS = { cpp: CppBackend, js: JsBackend };
export const DEFAULT_BACKEND = 'cpp';

export const backendKind = (kind) => (kind in BACKENDS ? kind : DEFAULT_BACKEND);
