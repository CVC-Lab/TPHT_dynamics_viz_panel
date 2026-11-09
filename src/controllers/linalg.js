// Small dense linear algebra for 6x6 / 9x9 problems. Row-major Float64Arrays.

// Cholesky factorization of an n x n SPD matrix. Returns lower factor L or null.
export function cholesky(a, n) {
  const l = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i * n + j];
      for (let k = 0; k < j; k++) s -= l[i * n + k] * l[j * n + k];
      if (i === j) {
        if (!(s > 0)) return null;
        l[i * n + i] = Math.sqrt(s);
      } else {
        l[i * n + j] = s / l[j * n + j];
      }
    }
  }
  return l;
}

// Solve (L L^T) x = b in place for one right-hand side of length n.
export function cholSolveVec(l, n, b) {
  const x = Float64Array.from(b);
  for (let i = 0; i < n; i++) {
    let s = x[i];
    for (let k = 0; k < i; k++) s -= l[i * n + k] * x[k];
    x[i] = s / l[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let k = i + 1; k < n; k++) s -= l[k * n + i] * x[k];
    x[i] = s / l[i * n + i];
  }
  return x;
}

// Solve A X = B for SPD A (n x n) and B (n x m), returns X (n x m).
export function solveSPD(a, n, b, m) {
  const l = cholesky(a, n);
  if (!l) throw new Error('Matrix is not symmetric positive definite');
  const x = new Float64Array(n * m);
  const col = new Float64Array(n);
  for (let j = 0; j < m; j++) {
    for (let i = 0; i < n; i++) col[i] = b[i * m + j];
    const s = cholSolveVec(l, n, col);
    for (let i = 0; i < n; i++) x[i * m + j] = s[i];
  }
  return x;
}

// y = A x, A is r x c.
export function matVec(a, r, c, x) {
  const y = new Float64Array(r);
  for (let i = 0; i < r; i++) {
    let s = 0;
    for (let j = 0; j < c; j++) s += a[i * c + j] * x[j];
    y[i] = s;
  }
  return y;
}

export const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
export const maxAbs = (v) => v.reduce((s, x) => Math.max(s, Math.abs(x)), 0);
export const maxAbsDiff = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s = Math.max(s, Math.abs(a[i] - b[i]));
  return s;
};

// np.interp semantics: clamps outside [xs[0], xs[-1]]; xs ascending.
export function interp(x, xs, ys) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}
