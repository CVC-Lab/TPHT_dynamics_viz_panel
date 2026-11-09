// Port of mujoco_reference/common.py::bounded_ls.
//
// Python solves   min || [sqrt(w) J; sqrt(ridge) I] x - [sqrt(w)(desired - offset); 0] ||^2
// subject to      low <= x <= high
// with scipy.optimize.lsq_linear(method='bvls'). The ridge makes the problem
// strictly convex, so the minimizer is unique and any exact method returns the
// same point. Here we use a primal active-set method on the normal equations,
// which terminates finitely for a 6-variable strictly convex box QP.
import { cholesky, cholSolveVec, norm } from './linalg.js';

export const ACCEL_WEIGHTS = [1, 4, 4, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15];

// Minimize 0.5 x'Qx - c'x on a box. Q is n x n SPD (row-major).
export function boxQP(Q, c, low, high, n, maxIter = 100, tol = 1e-12) {
  const x = new Float64Array(n);
  // 0: free, -1: fixed at low, +1: fixed at high
  const state = new Int8Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.min(high[i], Math.max(low[i], 0));
  for (let i = 0; i < n; i++) {
    if (x[i] === low[i] && low[i] === high[i]) state[i] = -1;
  }
  let iterations = 0;
  for (; iterations < maxIter; iterations++) {
    const free = [];
    for (let i = 0; i < n; i++) if (state[i] === 0) free.push(i);
    // Solve the equality-constrained subproblem on the free set.
    const target = new Float64Array(n);
    target.set(x);
    if (free.length) {
      const f = free.length;
      const qff = new Float64Array(f * f);
      const rhs = new Float64Array(f);
      for (let a = 0; a < f; a++) {
        const i = free[a];
        let r = c[i];
        for (let j = 0; j < n; j++) if (state[j] !== 0) r -= Q[i * n + j] * x[j];
        rhs[a] = r;
        for (let b = 0; b < f; b++) qff[a * f + b] = Q[i * n + free[b]];
      }
      const l = cholesky(qff, f);
      if (!l) throw new Error('boxQP: free-set Hessian is not positive definite');
      const sol = cholSolveVec(l, f, rhs);
      for (let a = 0; a < f; a++) target[free[a]] = sol[a];
    }
    // Step toward the subproblem solution, stopping at the first bound.
    let alpha = 1, block = -1, blockSide = 0;
    for (const i of free) {
      const d = target[i] - x[i];
      if (d > 0 && target[i] > high[i] + tol) {
        const a = (high[i] - x[i]) / d;
        if (a < alpha) { alpha = a; block = i; blockSide = 1; }
      } else if (d < 0 && target[i] < low[i] - tol) {
        const a = (low[i] - x[i]) / d;
        if (a < alpha) { alpha = a; block = i; blockSide = -1; }
      }
    }
    for (const i of free) x[i] += alpha * (target[i] - x[i]);
    if (block >= 0) {
      x[block] = blockSide > 0 ? high[block] : low[block];
      state[block] = blockSide;
      continue;
    }
    // Subproblem optimum is feasible: check multipliers of fixed variables.
    let worst = -1, worstVal = tol;
    for (let i = 0; i < n; i++) {
      if (state[i] === 0 || low[i] === high[i]) continue;
      let g = -c[i];
      for (let j = 0; j < n; j++) g += Q[i * n + j] * x[j];
      // At the lower bound the gradient must be >= 0; at the upper, <= 0.
      const violation = state[i] < 0 ? -g : g;
      if (violation > worstVal) { worstVal = violation; worst = i; }
    }
    if (worst < 0) return { x, iterations: iterations + 1, converged: true };
    state[worst] = 0;
  }
  return { x, iterations, converged: false };
}

// jacobian: 9x6 row-major, offset/desired: length 9.
export function boundedLS(jacobian, offset, desired, ridge, low = -1, high = 1) {
  const nr = 9, n = 6, w = ACCEL_WEIGHTS;
  const lo = typeof low === 'number' ? new Float64Array(n).fill(low) : Float64Array.from(low);
  const hi = typeof high === 'number' ? new Float64Array(n).fill(high) : Float64Array.from(high);
  const Q = new Float64Array(n * n);
  const c = new Float64Array(n);
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) {
      let s = a === b ? ridge : 0;
      for (let r = 0; r < nr; r++) s += w[r] * jacobian[r * n + a] * jacobian[r * n + b];
      Q[a * n + b] = s;
    }
    let s = 0;
    for (let r = 0; r < nr; r++) s += w[r] * jacobian[r * n + a] * (desired[r] - offset[r]);
    c[a] = s;
  }
  const { x, iterations, converged } = boxQP(Q, c, lo, hi, n);
  if (!converged) throw new Error('Bounded least squares did not converge');
  // Same diagnostics as the Python reference.
  const grad = new Float64Array(n);
  for (let a = 0; a < n; a++) {
    let s = -c[a];
    for (let b = 0; b < n; b++) s += Q[a * n + b] * x[b];
    grad[a] = s;
  }
  let projected = 0;
  for (let a = 0; a < n; a++) {
    const p = x[a] - Math.min(hi[a], Math.max(lo[a], x[a] - grad[a]));
    projected = Math.max(projected, Math.abs(p));
  }
  const res = new Float64Array(nr);
  for (let r = 0; r < nr; r++) {
    let s = offset[r] - desired[r];
    for (let a = 0; a < n; a++) s += jacobian[r * n + a] * x[a];
    res[r] = Math.sqrt(w[r]) * s;
  }
  return {
    action: x,
    diagnostics: {
      ls_projected_gradient_inf: projected,
      weighted_acceleration_residual: norm(res),
      ls_iterations: iterations,
    },
  };
}
