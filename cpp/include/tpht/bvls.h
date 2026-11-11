// Bounded least squares. Port of src/controllers/bvls.js, itself a port of
// mujoco_reference/common.py::bounded_ls.
//
// Python solves   min || [sqrt(w) J; sqrt(ridge) I] x - [sqrt(w)(desired - offset); 0] ||^2
// subject to      low <= x <= high
// with scipy.optimize.lsq_linear(method='bvls'). The ridge makes the problem
// strictly convex, so the minimizer is unique and any exact method returns the
// same point; this is a primal active-set method on the normal equations.
#pragma once

#include <array>

#include "tpht/linalg.h"

namespace tpht {

constexpr int kAccelRows = 9;   // planar cheetah: rootx, rootz, rooty, six joints
constexpr int kActions = 6;
extern const std::array<double, kAccelRows> kAccelWeights;

struct BoxQPResult {
  Vec x;
  int iterations = 0;
  bool converged = false;
};

// Minimizes 0.5 x'Qx - c'x on a box. Q is n x n SPD (row-major).
BoxQPResult box_qp(const double* Q, const double* c, const double* low, const double* high, int n,
                   int max_iter = 100, double tol = 1e-12);

struct LsDiagnostics {
  double projected_gradient_inf = 0;  // ls_projected_gradient_inf
  double weighted_residual = 0;       // weighted_acceleration_residual
  int iterations = 0;                 // ls_iterations
};

struct BoundedLsResult {
  std::array<double, kActions> action{};
  LsDiagnostics diagnostics;
};

// jacobian: 9x6 row-major; offset, desired: 9; low, high: 6. Throws when the
// active-set method does not converge.
BoundedLsResult bounded_ls(const double* jacobian, const double* offset, const double* desired, double ridge,
                           const double* low, const double* high);

}  // namespace tpht
