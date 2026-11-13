// Runs in the browser: tests/index.html (tpht-lab serve) or tpht-lab test (headless).
// The C++ backend (cpp/, compiled to WebAssembly) against the JavaScript port, on
// dm_control's cheetah from the site's assets. The benchmark's own models are
// git-ignored, so the reference controllers run on the cheetah with
// fixtures/cheetah_gait.json (also used by cpp/tests/sim_test.cpp).
import { test, assert, loadJSON } from './harness.js';
import { CppBackend } from '../src/backend/cpp_backend.js';
import { JsBackend } from '../src/backend/js_backend.js';

const asset = (p) => new URL(`../assets/${p}`, import.meta.url);
const fixture = (f) => loadJSON(new URL(`./fixtures/${f}`, import.meta.url));

async function cheetah(Backend) {
  const be = await Backend.create();
  const r = await fetch(asset('ctrl/cheetah.xml'));
  if (!r.ok) throw new Error(`fetch cheetah.xml: HTTP ${r.status}`);
  const meta = await loadJSON(asset('ctrl/cheetah_meta.json'));
  const loaded = be.load({ kind: 'xml', name: 'cheetah.xml', data: await r.text() }, meta, { timestep: meta.timestep });
  return { be, loaded };
}

const maxDiff = (a, b) => a.reduce((s, x, i) => Math.max(s, Math.abs(x - b[i])), 0);

// Values that depend on the MuJoCo release, keyed by version (shared with
// cpp/tests/sim_test.cpp). demo_dx_5s: the demo-gait rollout through the MuJoCo
// Python package of that version, stepping as the page does.
async function mujocoReference(key, version) {
  const ref = await fixture('mujoco_reference.json');
  const v = ref[key][version];
  if (typeof v !== 'number') throw new Error(`fixtures/mujoco_reference.json has no ${key} for MuJoCo ${version}`);
  return v;
}

// The MuJoCo release the site was built with (tools/build_site.py writes it),
// or null when the tests run outside a built site.
async function siteMujocoVersion() {
  try { return (await loadJSON(new URL('../build-info.json', import.meta.url))).mujoco ?? null; } catch { return null; }
}

function demoRollout(be, steps) {
  be.resetData();
  be.useSineGait();
  const x0 = be.qpos()[0];
  for (let k = 0; k < steps; k++) be.stepHold(be.decide(3, 0.01).action, 1);
  return be.qpos()[0] - x0;
}

test('C++ and JavaScript backends load the cheetah; every binding self-test passes', async () => {
  const notes = [];
  const site = await siteMujocoVersion();
  const versions = new Set();
  for (const Backend of [CppBackend, JsBackend]) {
    const { be, loaded } = await cheetah(Backend);
    for (const t of loaded.selfTests) assert.ok(t.ok, `${be.name}: ${t.name}: ${t.detail}`);
    if (site) assert.equal(be.mujocoVersion, site, `${be.name} runs MuJoCo ${be.mujocoVersion}, the site was built with ${site}`);
    versions.add(be.mujocoVersion);
    notes.push(`${be.kind}: ${loaded.selfTests.length} pass via ${loaded.via}`);
    be.dispose();
  }
  assert.equal(versions.size, 1, `the backends run different MuJoCo versions: ${[...versions].join(', ')}`);
  return `MuJoCo ${[...versions][0]}; ${notes.join('; ')}`;
});

test('demo gait, 5 s: C++ = native Python; JavaScript port within 1e-6', async () => {
  const { be: cpp } = await cheetah(CppBackend);
  const { be: js } = await cheetah(JsBackend);
  const python = await mujocoReference('demo_dx_5s', cpp.mujocoVersion);
  const dc = demoRollout(cpp, 500), dj = demoRollout(js, 500);
  cpp.dispose(); js.dispose();
  assert.ok(Math.abs(dc - python) < 1e-9, `C++ moved ${dc} m, Python ${python} m`);
  assert.ok(Math.abs(dj - dc) < 1e-6, `JavaScript moved ${dj} m, C++ ${dc} m`);
  return `C++ ${dc.toFixed(12)} m, JavaScript ${dj.toFixed(12)} m, |diff| ${Math.abs(dj - dc).toExponential(1)}`;
});

test('reference controllers on the cheetah: C++ = JavaScript port (LS, FD-LS, MPC)', async () => {
  const gait = await fixture('cheetah_gait.json');
  const notes = [];
  for (const [method, decisions, tol] of [['mujoco_ls', 60, 1e-9], ['mujoco_fd_ls', 60, 1e-6], ['mujoco_mpc', 25, 1e-6]]) {
    const runs = [];
    for (const Backend of [CppBackend, JsBackend]) {
      const { be } = await cheetah(Backend);
      be.setGait(gait);
      be.resetData();
      be.useNative(method);
      const actions = [];
      let plans = 0;
      for (let k = 0; k < decisions; k++) {
        const { action, diag } = be.decide(1, 0.01);
        actions.push(...action);
        plans += diag.planned ? 1 : 0;
        be.stepHold(action, 1);
      }
      runs.push({ actions, plans, x: be.qpos()[0] });
      be.dispose();
    }
    const d = maxDiff(runs[0].actions, runs[1].actions);
    assert.ok(d <= tol, `${method}: max |action C++ - action JS| = ${d} over ${decisions} decisions`);
    assert.equal(runs[0].plans, runs[1].plans, `${method}: ${runs[0].plans} vs ${runs[1].plans} plans`);
    notes.push(`${method} ${d.toExponential(1)}`);
  }
  return `max action difference: ${notes.join(', ')}`;
});

test('C++ errors reach JavaScript with BackendError codes', async () => {
  const be = await CppBackend.create();
  let code = null;
  try { be.load({ kind: 'xml', name: 'broken.xml', data: '<mujoco><worldbody><body' }, {}, {}); } catch (e) { code = e.code; }
  assert.equal(code, 'model-compile');
  const { be: c } = await cheetah(CppBackend);
  code = null;
  try { c.useNative('mujoco_ls'); } catch (e) { code = e.code; }
  assert.equal(code, 'gait-missing');
  be.dispose(); c.dispose();
});
