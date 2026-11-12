// Small dense linear algebra for the 6x6 / 9x9 problems of the controllers.
// Row-major. Port of src/controllers/linalg.js, operation for operation, so the
// C++ and JavaScript backends round the same way wherever the compiler lets them.
#pragma once

#include <vector>

namespace tpht {

using Vec = std::vector<double>;

// Cholesky factor L (lower, n x n) of an SPD matrix a. Returns false (and leaves
// l unspecified) when a is not positive definite, including on NaN.
bool cholesky(const double* a, int n, double* l);

// Solves (L L^T) x = b in place for one right-hand side of length n.
void chol_solve(const double* l, int n, double* x);

// Solves A X = B for SPD A (n x n) and B (n x m); returns X (n x m). Throws
// std::runtime_error when A is not SPD.
Vec solve_spd(const double* a, int n, const double* b, int m);

// y = A x, A is r x c.
Vec mat_vec(const double* a, int r, int c, const double* x);

double norm(const double* v, int n);
inline double norm(const Vec& v) { return norm(v.data(), static_cast<int>(v.size())); }
double max_abs_diff(const double* a, const double* b, int n);

// numpy.interp: clamps outside [xs[0], xs[n-1]]; xs ascending.
double interp(double x, const double* xs, const double* ys, int n);
inline double interp(double x, const Vec& xs, const Vec& ys) {
  return interp(x, xs.data(), ys.data(), static_cast<int>(xs.size()));
}

}  // namespace tpht
