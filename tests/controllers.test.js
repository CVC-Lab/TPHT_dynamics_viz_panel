// Runs in the browser: tests/index.html (tpht-lab serve) or tpht-lab test (headless).
// Fixtures: python tools/make_test_fixtures.py
import { test, assert, loadJSON } from './harness.js';
import { boundedLS } from '../src/controllers/bvls.js';
import { references, desiredAcceleration } from '../src/controllers/gait.js';
import { interp } from '../src/controllers/linalg.js';

const load = (f) => loadJSON(new URL(`./fixtures/${f}`, import.meta.url));

test('boundedLS matches scipy lsq_linear(method="bvls") on 400 cheetah-shaped problems', async () => {
  let worst = 0, worstRes = 0;
  for (const c of await load('bvls_cases.json')) {
    const { action, diagnostics } = boundedLS(c.J, c.offset, c.desired, 0.01, c.low, c.high);
    for (let i = 0; i < 6; i++) worst = Math.max(worst, Math.abs(action[i] - c.x[i]));
    worstRes = Math.max(worstRes, Math.abs(diagnostics.weighted_acceleration_residual - c.residual) / Math.max(1, c.residual));
    assert.ok(diagnostics.ls_projected_gradient_inf < 1e-8, 'KKT residual');
  }
  assert.ok(worst < 1e-7, `max |x_js - x_scipy| = ${worst}`);
  assert.ok(worstRes < 1e-9, `residual mismatch ${worstRes}`);
});

test('gait references and acceleration target match the Python formulas', async () => {
  const { options, limits, cases } = await load('gait_cases.json');
  let worst = 0;
  for (const c of cases) {
    const r = references(c.p, c.phase, limits);
    const d = desiredAcceleration(
      { qpos: c.qpos, qvel: c.qvel, torsoPos: c.root, torsoLinvelX: c.linvel_x },
      c.scan, c.p, c.phase, 3, options, limits);
    for (let i = 0; i < 6; i++) {
      worst = Math.max(worst, Math.abs(r.q[i] - c.qr[i]), Math.abs(r.v[i] - c.vr[i]), Math.abs(r.a[i] - c.ar[i]));
    }
    for (let i = 0; i < 9; i++) worst = Math.max(worst, Math.abs(d[i] - c.desired[i]) / Math.max(1, Math.abs(c.desired[i])));
  }
  assert.ok(worst < 1e-10, `max mismatch ${worst}`);
});

test('interp clamps like numpy.interp', () => {
  assert.equal(interp(-5, [0, 1], [2, 4]), 2);
  assert.equal(interp(5, [0, 1], [2, 4]), 4);
  assert.equal(interp(0.25, [0, 1], [2, 4]), 2.5);
});
