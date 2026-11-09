// Port of mujoco_reference/controller.py::NativeController (plus the CEM loop
// from pure_cpht/controller.py::Controller.plan used by mujoco_mpc).
//
// Faithful parts: target construction, acceleration map, finite-difference
// Jacobian with trust box, bounded LS, rollout cost and validity rules, CEM
// update. Not bit-exact: the CEM noise uses a seeded JS generator instead of
// torch.Generator, so mujoco_mpc matches the Python arm in distribution only.
import { boundedLS } from './bvls.js';
import { desiredAcceleration } from './gait.js';
import { solveSPD, matVec, norm, interp } from './linalg.js';

// Seeded normal sampler (mulberry32 + Box-Muller).
export function makeNormal(seed) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => {
    const u = Math.max(uniform(), 1e-300), v = uniform();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

export class NativeController {
  // cfg: exported gait.json { gait, limits, options, low, high, scale }
  constructor(sim, cfg, method, { seed = 100000, fdEpsilon = 1e-3, trustRadius = 0.5 } = {}) {
    if (!['mujoco_ls', 'mujoco_fd_ls', 'mujoco_mpc'].includes(method)) throw new Error(`Unknown native method ${method}`);
    this.sim = sim;
    this.cfg = cfg;
    this.o = cfg.options;
    this.method = method;
    this.fdEpsilon = fdEpsilon;
    this.trustRadius = trustRadius;
    this.mean = cfg.gait.map((x, i) => Math.min(cfg.high[i], Math.max(cfg.low[i], x)));
    this.phase = 0;
    this.lastTime = null;
    this.lastPlan = -Infinity;
    this.firstHold = this.o.planning_dt;
    this.randn = makeNormal(seed);
    this.counters = { native_steps: 0, forward_calls: 0, feedback_solves: 0, plans: 0 };
    this.diagnostics = {};
  }

  forward(d, action) {
    this.sim.forward(d, action);
    this.counters.forward_calls++;
  }

  desired(d, scan, params, phase, target) {
    return desiredAcceleration(this.sim.torsoState(d), scan, params, phase, target, this.o, this.cfg.limits);
  }

  // Acceleration map with the measured current constraint force frozen.
  accelerationMap(d) {
    const { nv, nu, B } = this.sim;
    const M = this.sim.fullMass(d);
    const J = solveSPD(M, nv, B, nu);
    const Ju = matVec(J, nv, nu, d.ctrl);
    const offset = Float64Array.from(d.qacc, (a, i) => a - Ju[i]);
    return { J, offset };
  }

  fit(d, scan, params, phase, target, finiteDifference = false, diagnose = false) {
    const sim = this.sim;
    this.forward(d);
    const desired = this.desired(d, scan, params, phase, target);
    const center = Float64Array.from(d.ctrl);
    const base = Float64Array.from(d.qacc);
    let { J, offset } = this.accelerationMap(d);
    let low = -1, high = 1;
    if (finiteDifference) {
      J = new Float64Array(9 * 6);
      const [dm, dp] = [sim.scratch[2], sim.scratch[3]];
      for (let j = 0; j < 6; j++) {
        const minus = Float64Array.from(center), plus = Float64Array.from(center);
        minus[j] = Math.max(-1, center[j] - this.fdEpsilon);
        plus[j] = Math.min(1, center[j] + this.fdEpsilon);
        sim.copyInto(dm, d); sim.copyInto(dp, d);
        this.forward(dm, minus);
        this.forward(dp, plus);
        for (let i = 0; i < 9; i++) J[i * 6 + j] = (dp.qacc[i] - dm.qacc[i]) / (plus[j] - minus[j]);
      }
      const Jc = matVec(J, 9, 6, center);
      offset = Float64Array.from(base, (a, i) => a - Jc[i]);
      low = center.map((c) => Math.max(-1, c - this.trustRadius));
      high = center.map((c) => Math.min(1, c + this.trustRadius));
    }
    const { action, diagnostics } = boundedLS(J, offset, desired, this.o.feedback_ridge, low, high);
    this.counters.feedback_solves++;
    if (diagnose) {
      const actual = sim.copyInto(sim.scratch[4], d);
      this.forward(actual, action);
      const lin = matVec(J, 9, 6, action).map((x, i) => x + offset[i]);
      const err = Float64Array.from(actual.qacc, (a, i) => a - lin[i]);
      Object.assign(diagnostics, {
        acceleration_target: desired,
        acceleration_linear_prediction: lin,
        acceleration_full_contact: Float64Array.from(actual.qacc),
        contact_linearization_error: norm(err),
        action_saturation: action.reduce((s, x) => s + (Math.abs(x) > 0.99), 0) / 6,
      });
    }
    return { action, diagnostics };
  }

  rollout(scan, population, target, phase) {
    const sim = this.sim, o = this.o, m = sim.model;
    const dt = m.opt.timestep;
    const xs = scan.map((r) => r[0]), hs = scan.map((r) => r[1]);
    const jr = m.jnt_range;
    const costs = [], failures = [];
    const d = sim.scratch[1];
    for (const cand of population) {
      sim.copyInto(d, sim.data);
      this.forward(d);
      const start = sim.torsoState(d).torsoComX;
      let elapsed = 0, angle = phase, cost = 0, invalid = false;
      for (let interval = 0; interval < o.horizon_steps && !invalid; interval++) {
        const duration = interval === 0 ? this.firstHold : o.planning_dt;
        const count = Math.max(1, Math.round(duration / dt));
        const { action } = this.fit(d, scan, cand, angle, target);
        d.ctrl.set(action);
        sim.mj.mj_step1(m, d); // restore the split-step invariant after mj_forward
        const effort = action.reduce((s, x) => s + x * x, 0);
        for (let k = 0; k < count; k++) {
          if (sim.legacyStep) { sim.mj.mj_step2(m, d); sim.mj.mj_step1(m, d); } else sim.mj.mj_step(m, d);
          sim.mj.mj_subtreeVel(m, d);
          this.counters.native_steps++;
          const st = sim.torsoState(d);
          const clearance = st.torsoPos[2] - interp(st.torsoPos[0], xs, hs);
          const speed = st.torsoLinvelX;
          let stage = o.speed_weight * (speed - target) ** 2 + o.clearance_weight * (clearance - o.target_clearance) ** 2;
          stage += o.pitch_weight * d.qpos[2] ** 2 + o.effort_weight * effort;
          stage += o.uncertainty_weight * (st.torsoPos[0] < xs[0] || st.torsoPos[0] > xs[xs.length - 1] ? 1 : 0);
          cost += dt * stage;
          elapsed += dt;
          angle += 2 * Math.PI * cand[18] * dt;
          let bad = clearance < 0.15 || Math.abs(d.qpos[2]) > 1.6;
          for (let i = 0; i < 9 && !bad; i++) {
            if (!Number.isFinite(d.qpos[i]) || !Number.isFinite(d.qvel[i]) || Math.abs(d.qvel[i]) > 80) bad = true;
          }
          for (let j = 3; j < 9 && !bad; j++) {
            if (d.qpos[j] < jr[j * 2] - 0.3 || d.qpos[j] > jr[j * 2 + 1] + 0.3) bad = true;
          }
          if (bad) { invalid = true; break; }
        }
      }
      const st = sim.torsoState(d);
      cost += o.progress_weight * Math.max(0, target * elapsed - (st.torsoComX - start)) + 0.1 * (st.torsoLinvelX - target) ** 2;
      costs.push(cost + 1e6 * (invalid ? 1 : 0));
      failures.push(invalid);
    }
    return { costs, failures };
  }

  plan(scan, target) {
    const o = this.o, { low, high, scale } = this.cfg;
    let mean = this.mean.slice(), std = scale.slice(), best = mean.slice(), bestCost = Infinity, fraction = 0;
    const clampP = (p) => p.map((x, i) => Math.min(high[i], Math.max(low[i], x)));
    for (let it = 0; it < o.iterations; it++) {
      const population = Array.from({ length: o.candidates }, () => clampP(mean.map((m, i) => m + this.randn() * std[i])));
      population[0] = best.slice();
      const { costs, failures } = this.rollout(scan, population, target, this.phase);
      const order = costs.map((c, i) => [c, i]).sort((a, b) => a[0] - b[0]).slice(0, o.elites).map((x) => x[1]);
      if (costs[order[0]] < bestCost) { bestCost = costs[order[0]]; best = population[order[0]].slice(); }
      const eMean = mean.map((_, i) => order.reduce((s, k) => s + population[k][i], 0) / order.length);
      const eStd = mean.map((_, i) => Math.sqrt(order.reduce((s, k) => s + (population[k][i] - eMean[i]) ** 2, 0) / order.length));
      mean = mean.map((m, i) => 0.25 * m + 0.75 * eMean[i]);
      std = std.map((s, i) => Math.max(0.015, 0.25 * s + 0.75 * eStd[i]));
      fraction = failures.filter(Boolean).length / failures.length;
    }
    this.mean = best;
    this.counters.plans++;
    Object.assign(this.diagnostics, { planner_cost: bestCost, planner_invalid_fraction: fraction, frequency_hz: best[18] });
  }

  // Mirrors NativeController.act. scan: rows [x, height, slope].
  act(scan, target, hold, timeNow) {
    const started = performance.now();
    if (this.lastTime !== null) {
      this.phase = (this.phase + 2 * Math.PI * this.mean[18] * Math.max(0, timeNow - this.lastTime)) % (2 * Math.PI);
    }
    this.lastTime = timeNow;
    this.firstHold = hold;
    let planned = false;
    if (this.method === 'mujoco_mpc' && timeNow - this.lastPlan >= this.o.replan_seconds - 1e-9) {
      this.plan(scan, target);
      this.lastPlan = timeNow;
      planned = true;
    }
    const d = this.sim.copyInto(this.sim.scratch[0], this.sim.data);
    const { action, diagnostics } = this.fit(d, scan, this.mean, this.phase, target, this.method === 'mujoco_fd_ls', true);
    Object.assign(this.diagnostics, diagnostics, { phase: this.phase, planned, decision_ms: performance.now() - started });
    // The recorded environment used float32 controls.
    return Float32Array.from(action);
  }
}
