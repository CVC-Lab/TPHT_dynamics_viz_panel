#include "tpht/bvls.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tpht {

const std::array<double, kAccelRows> kAccelWeights = {1, 4, 4, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15};

BoxQPResult box_qp(const double* Q, const double* c, const double* low, const double* high, int n,
                   int max_iter, double tol) {
  BoxQPResult out;
  Vec& x = out.x;
  x.assign(n, 0.0);
  // 0: free, -1: fixed at low, +1: fixed at high
  std::vector<int> state(n, 0);
  for (int i = 0; i < n; i++) x[i] = std::min(high[i], std::max(low[i], 0.0));
  for (int i = 0; i < n; i++) {
    if (x[i] == low[i] && low[i] == high[i]) state[i] = -1;
  }
  std::vector<int> free;
  Vec target(n), qff, rhs, lff;
  int iterations = 0;
  for (; iterations < max_iter; iterations++) {
    free.clear();
    for (int i = 0; i < n; i++) if (state[i] == 0) free.push_back(i);
    // Solve the equality-constrained subproblem on the free set.
    target = x;
    if (!free.empty()) {
      const int f = static_cast<int>(free.size());
      qff.assign(static_cast<size_t>(f) * f, 0.0);
      rhs.assign(f, 0.0);
      lff.assign(static_cast<size_t>(f) * f, 0.0);
      for (int a = 0; a < f; a++) {
        const int i = free[a];
        double r = c[i];
        for (int j = 0; j < n; j++) if (state[j] != 0) r -= Q[i * n + j] * x[j];
        rhs[a] = r;
        for (int b = 0; b < f; b++) qff[a * f + b] = Q[i * n + free[b]];
      }
      if (!cholesky(qff.data(), f, lff.data())) {
        throw std::runtime_error("boxQP: free-set Hessian is not positive definite");
      }
      chol_solve(lff.data(), f, rhs.data());
      for (int a = 0; a < f; a++) target[free[a]] = rhs[a];
    }
    // Step toward the subproblem solution, stopping at the first bound.
    double alpha = 1;
    int block = -1, block_side = 0;
    for (const int i : free) {
      const double d = target[i] - x[i];
      if (d > 0 && target[i] > high[i] + tol) {
        const double a = (high[i] - x[i]) / d;
        if (a < alpha) { alpha = a; block = i; block_side = 1; }
      } else if (d < 0 && target[i] < low[i] - tol) {
        const double a = (low[i] - x[i]) / d;
        if (a < alpha) { alpha = a; block = i; block_side = -1; }
      }
    }
    for (const int i : free) x[i] += alpha * (target[i] - x[i]);
    if (block >= 0) {
      x[block] = block_side > 0 ? high[block] : low[block];
      state[block] = block_side;
      continue;
    }
    // Subproblem optimum is feasible: check multipliers of fixed variables.
    int worst = -1;
    double worst_val = tol;
    for (int i = 0; i < n; i++) {
      if (state[i] == 0 || low[i] == high[i]) continue;
      double g = -c[i];
      for (int j = 0; j < n; j++) g += Q[i * n + j] * x[j];
      // At the lower bound the gradient must be >= 0; at the upper, <= 0.
      const double violation = state[i] < 0 ? -g : g;
      if (violation > worst_val) { worst_val = violation; worst = i; }
    }
    if (worst < 0) {
      out.iterations = iterations + 1;
      out.converged = true;
      return out;
    }
    state[worst] = 0;
  }
  out.iterations = iterations;
  out.converged = false;
  return out;
}

BoundedLsResult bounded_ls(const double* jacobian, const double* offset, const double* desired, double ridge,
                           const double* low, const double* high) {
  constexpr int nr = kAccelRows, n = kActions;
  const auto& w = kAccelWeights;
  double Q[n * n], c[n];
  for (int a = 0; a < n; a++) {
    for (int b = 0; b < n; b++) {
      double s = a == b ? ridge : 0;
      for (int r = 0; r < nr; r++) s += w[r] * jacobian[r * n + a] * jacobian[r * n + b];
      Q[a * n + b] = s;
    }
    double s = 0;
    for (int r = 0; r < nr; r++) s += w[r] * jacobian[r * n + a] * (desired[r] - offset[r]);
    c[a] = s;
  }
  const BoxQPResult qp = box_qp(Q, c, low, high, n);
  if (!qp.converged) throw std::runtime_error("Bounded least squares did not converge");
  BoundedLsResult out;
  std::copy(qp.x.begin(), qp.x.end(), out.action.begin());
  const double* x = qp.x.data();
  // Same diagnostics as the Python reference.
  double projected = 0;
  for (int a = 0; a < n; a++) {
    double g = -c[a];
    for (int b = 0; b < n; b++) g += Q[a * n + b] * x[b];
    const double p = x[a] - std::min(high[a], std::max(low[a], x[a] - g));
    projected = std::max(projected, std::abs(p));
  }
  double res[nr];
  for (int r = 0; r < nr; r++) {
    double s = offset[r] - desired[r];
    for (int a = 0; a < n; a++) s += jacobian[r * n + a] * x[a];
    res[r] = std::sqrt(w[r]) * s;
  }
  out.diagnostics.projected_gradient_inf = projected;
  out.diagnostics.weighted_residual = norm(res, nr);
  out.diagnostics.iterations = qp.iterations;
  return out;
}

}  // namespace tpht
