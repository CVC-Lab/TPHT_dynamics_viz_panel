// Ports of pure_cpht/controller.py::Controller.references and
// mujoco_reference/controller.py::NativeController.desired.
import { interp } from './linalg.js';

// params: 19 coefficients [offset(6), sin(6), cos(6), frequency_hz].
// limits: 6 x [low, high] from the saved robot (exported by tools/export_web_assets.py gait).
export function references(params, phase, limits) {
  const omega = 2 * Math.PI * params[18];
  const s = Math.sin(phase), c = Math.cos(phase);
  const q = new Float64Array(6), v = new Float64Array(6), a = new Float64Array(6);
  for (let i = 0; i < 6; i++) {
    const wave = params[6 + i] * s + params[12 + i] * c;
    const qi = params[i] + wave;
    const lo = limits[i][0] + 0.02, hi = limits[i][1] - 0.02;
    const limited = qi > hi || qi < lo; // evaluated before clamping, as in Python
    q[i] = Math.min(hi, Math.max(lo, qi));
    v[i] = limited ? 0 : omega * (params[6 + i] * c - params[12 + i] * s);
    a[i] = limited ? 0 : -omega * omega * wave;
  }
  return { q, v, a };
}

// state: { qpos, qvel, torsoPos:[x,y,z], torsoLinvelX } read from MuJoCo after
// mj_forward + mj_subtreeVel. scan: rows [x, height, slope].
export function desiredAcceleration(state, scan, params, phase, target, options, limits) {
  const o = options;
  const { q: qr, v: vr, a: ar } = references(params, phase, limits);
  const xs = scan.map((r) => r[0]);
  const hs = scan.map((r) => r[1]);
  const height = interp(state.torsoPos[0], xs, hs);
  const out = new Float64Array(9);
  out[0] = o.speed_kd * (target - state.torsoLinvelX);
  out[1] = o.height_kp * (o.target_clearance - (state.torsoPos[2] - height)) - o.height_kd * state.qvel[1];
  out[2] = -o.pitch_kp * state.qpos[2] - o.pitch_kd * state.qvel[2];
  for (let i = 0; i < 6; i++) {
    out[3 + i] = ar[i] + o.joint_kp * (qr[i] - state.qpos[3 + i]) + o.joint_kd * (vr[i] - state.qvel[3 + i]);
  }
  return out;
}

// Scan offsets observed in validation/reference_samples (relative to q_rootx).
export const SCAN_OFFSETS = [-1, -0.5, 0, 0.5, 1, 2, 3];

// Privileged scan from the exported ground profile. The causal environment
// observer (environment_belief/observer.py) is not ported yet.
export function privilegedScan(rootX, profile) {
  return SCAN_OFFSETS.map((dx) => {
    const x = rootX + dx;
    const h = interp(x, profile.x, profile.z);
    const e = 0.05;
    const slope = (interp(x + e, profile.x, profile.z) - interp(x - e, profile.x, profile.z)) / (2 * e);
    return [x, h, slope];
  });
}
