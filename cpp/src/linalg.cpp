#include "tpht/linalg.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tpht {

bool cholesky(const double* a, int n, double* l) {
  std::fill(l, l + n * n, 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double s = a[i * n + j];
      for (int k = 0; k < j; k++) s -= l[i * n + k] * l[j * n + k];
      if (i == j) {
        if (!(s > 0)) return false;
        l[i * n + i] = std::sqrt(s);
      } else {
        l[i * n + j] = s / l[j * n + j];
      }
    }
  }
  return true;
}

void chol_solve(const double* l, int n, double* x) {
  for (int i = 0; i < n; i++) {
    double s = x[i];
    for (int k = 0; k < i; k++) s -= l[i * n + k] * x[k];
    x[i] = s / l[i * n + i];
  }
  for (int i = n - 1; i >= 0; i--) {
    double s = x[i];
    for (int k = i + 1; k < n; k++) s -= l[k * n + i] * x[k];
    x[i] = s / l[i * n + i];
  }
}

Vec solve_spd(const double* a, int n, const double* b, int m) {
  Vec l(static_cast<size_t>(n) * n);
  if (!cholesky(a, n, l.data())) throw std::runtime_error("Matrix is not symmetric positive definite");
  Vec x(static_cast<size_t>(n) * m);
  Vec col(n);
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < n; i++) col[i] = b[i * m + j];
    chol_solve(l.data(), n, col.data());
    for (int i = 0; i < n; i++) x[i * m + j] = col[i];
  }
  return x;
}

Vec mat_vec(const double* a, int r, int c, const double* x) {
  Vec y(r);
  for (int i = 0; i < r; i++) {
    double s = 0;
    for (int j = 0; j < c; j++) s += a[i * c + j] * x[j];
    y[i] = s;
  }
  return y;
}

double norm(const double* v, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) s += v[i] * v[i];
  return std::sqrt(s);
}

double max_abs_diff(const double* a, const double* b, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) s = std::max(s, std::abs(a[i] - b[i]));
  return s;
}

double interp(double x, const double* xs, const double* ys, int n) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  int lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  const double t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

}  // namespace tpht
