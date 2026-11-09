// Binary traces written by tools/export_web_assets.py traces.
// Each file is float32, row-major, columns listed in traces/manifest.json.

// get: (pathInsideTracesFolder) => Promise<Response>
export async function loadTraceIndex(get) {
  const r = await get('manifest.json');
  if (!r.ok) throw new Error('No trace manifest');
  return r.json();
}

const cache = new Map();

export async function loadTrace(get, index, caseId, clock, method) {
  const entry = index.list.find((t) => t.case === caseId && t.clock === clock && t.method === method);
  if (!entry) return null;
  if (!cache.has(entry.file)) {
    const r = await get(entry.file);
    if (!r.ok) throw new Error(`Missing trace ${entry.file}`);
    cache.set(entry.file, new Float32Array(await r.arrayBuffer()));
  }
  return makeTrace(cache.get(entry.file), index.columns, entry);
}

export function makeTrace(buf, columns, entry) {
  const nc = columns.length, n = buf.length / nc;
  const col = (name) => columns.indexOf(name);
  const iT = col('time'), iH = col('h'), iQ = col('q0'), iV = col('v0'), iA = col('a0');
  const times = new Float64Array(n);
  const holds = new Float64Array(n);
  for (let i = 0; i < n; i++) { times[i] = buf[i * nc + iT]; holds[i] = buf[i * nc + iH]; }
  const row = (i) => ({
    time: times[i],
    q: Array.from(buf.subarray(i * nc + iQ, i * nc + iQ + 9)),
    v: Array.from(buf.subarray(i * nc + iV, i * nc + iV + 9)),
    action: Array.from(buf.subarray(i * nc + iA, i * nc + iA + 6)),
    get: (name) => buf[i * nc + col(name)],
  });
  return {
    entry, n, times, holds, row,
    // Live runs start from row 0's state at sim time 0, so trace time = t0 + sim time.
    t0: times[0],
    rowAt: (t) => (n ? row(rowIndexAt(times, t)) : null),
  };
}

// Last row with time <= t (clamped to the ends).
export function rowIndexAt(times, t) {
  let lo = 0, hi = times.length - 1;
  if (t <= times[0]) return 0;
  if (t >= times[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) lo = mid; else hi = mid;
  }
  return lo;
}

export const rowAtTime = rowIndexAt;
