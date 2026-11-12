import { test, assert, loadJSON, loadBytes, exists } from './harness.js';
import { makeTrace } from '../src/backend/traces.js';

// The built site serves public/assets/ as assets/; the repository checkout keeps it under public/.
const dir = new URL('../assets/reference/traces/', import.meta.url);

test('binary traces round-trip the exported rows', { skip: async () => !(await exists(new URL('manifest.json', dir))) }, async () => {
  const index = await loadJSON(new URL('manifest.json', dir));
  assert.equal(index.list.length, 20);
  for (const entry of index.list) {
    const raw = await loadBytes(new URL(entry.file, dir));
    const buf = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    const t = makeTrace(buf, index.columns, entry);
    assert.equal(t.n, entry.rows);
    const last = t.row(t.n - 1);
    assert.equal(last.q.length, 9);
    assert.ok(Number.isFinite(last.q[0]));
    // rowAt returns the last row at or before the requested time.
    assert.equal(t.rowAt(t.times[10] + 1e-6).time, t.times[10]);
  }
});
