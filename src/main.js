import { Viewport, Strip } from './ui/viewport.js';
import { REFERENCE_METHODS, CTRL_ACCESS, LIVE_CAVEATS } from './controllers/registry.js';
import { MODULES } from './explain/modules.js';
import { BASE, fetchAsset, setLocalAssets, assetEntriesFromFiles } from './base.js';

const $ = (id) => document.getElementById(id);
const ui = {
  status: $('status'), method: $('method'), caseSel: $('case'), clock: $('clock'), ghost: $('ghost'),
  target: $('target'), scale: $('scale'), play: $('play'), access: $('access'), backend: $('backend'),
};
// ?backend=js picks the JavaScript port; the C++ core is the default.
ui.backend.value = new URLSearchParams(location.search).get('backend') === 'js' ? 'js' : 'cpp';
const view = new Viewport($('viewport'));
const strip = new Strip($('strip'));
const state = { section: null, running: false, ready: false, policies: [], traces: [], results: null, frame: null };

const worker = new Worker(new URL('./backend/sim.worker.js', import.meta.url), { type: 'module' });
const send = (type, payload = {}) => worker.postMessage({ type, ...payload });

function setStatus(text, error = false) {
  ui.status.textContent = text;
  ui.status.classList.toggle('error', error);
}

// ------------------------------------------------------------ worker events --
worker.onmessage = ({ data: m }) => {
  switch (m.type) {
    case 'status': setStatus(m.text); break;
    case 'error':
      setStatus(m.message, true);
      view.message = m.message;
      if (!state.frame) view.draw(null);
      break;
    case 'ready': onReady(m); break;
    case 'frame': onFrame(m); break;
    case 'running':
      state.running = m.running;
      ui.play.textContent = m.running ? 'Pause' : 'Run';
      break;
    case 'golden': renderGolden(m); break;
  }
};
worker.onerror = (e) => setStatus(`Worker failed to start: ${e.message}`, true);

function onReady(m) {
  state.ready = true;
  ui.backend.value = m.backend; // the worker may have fallen back to the JavaScript port
  state.traces = m.traces;
  state.policies = m.policies;
  view.setGround(m.profile, m.planes);
  renderSelfTests(m.selfTests);
  fillMethods();
  fillGhosts();
  ui.target.value = m.target; $('target-out').textContent = `${(+m.target).toFixed(1)} m/s`;
  const missing = state.section === 'reference' && !m.gaitLoaded;
  setStatus(missing
    ? 'Model loaded. Native controllers need gait.json from final.pt (see README); the golden replay still works.'
    : `Model loaded (timestep ${m.timestep} s).`);
  if (!missing) reset();
}

function onFrame(f) {
  state.frame = f;
  view.draw(f);
  strip.push(f.t, f.speed, f.target);
  strip.draw(view.color);
  renderStats(f.stats);
  renderDiag(f.diag, f.action);
}

// --------------------------------------------------------------- sections ----
function route() {
  const tab = (location.hash || '#reference').slice(1);
  document.querySelectorAll('.tabs a').forEach((a) => {
    if (a.dataset.tab === tab) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  const explain = tab === 'explain';
  $('sim-view').hidden = explain;
  $('explain-view').hidden = !explain;
  if (explain) { renderExplainList(); return; }
  if (tab !== state.section) loadSection(tab);
}

function loadSection(section) {
  state.section = section;
  state.ready = false;
  state.frame = null;
  strip.reset();
  document.querySelectorAll('[data-only]').forEach((el) => { el.hidden = el.dataset.only !== section; });
  $('caveats').innerHTML = LIVE_CAVEATS
    .filter((c) => section === 'reference' || !c.includes('Scan') && !c.includes('MPC'))
    .concat(section === 'ctrl' ? ['Irregular clock draws holds uniformly between 2 and 30 ms; the CT-RL time-grid generator is not ported.',
      'Episodes start from the model default pose; dm_control randomizes the start pose.'] : [])
    .map((c) => `<li>${c}</li>`).join('');
  ui.method.innerHTML = '';
  if (section === 'reference') fillMethods(); // static list; ghosts fill in once traces load
  view.message = 'Loading…';
  view.draw(null);
  send('init', { section, caseId: ui.caseSel.value, backend: ui.backend.value });
  if (section === 'reference') loadResults();
}

function fillMethods() {
  if (state.section === 'reference') {
    ui.method.innerHTML = REFERENCE_METHODS.map((m) =>
      `<option value="${m.id}" ${m.live ? '' : 'disabled'}>${m.label}${m.live ? '' : ' (ghost only)'}</option>`).join('');
  } else if (state.policies.length) {
    ui.method.innerHTML = state.policies.map((p) => `<option value="${p.id}">${p.label}</option>`).join('');
  } else {
    ui.method.innerHTML = '<option value="">No exported policies</option>';
    setStatus('No CT-RL policies yet. Export one with: python3 tools/export_web_assets.py ctrl-policy …');
  }
  describeMethod();
}

function fillGhosts() {
  const list = state.traces.filter((t) => t.case === ui.caseSel.value && t.clock === ui.clock.value);
  ui.ghost.innerHTML = '<option value="">None</option>' + list.map((t) => {
    const m = REFERENCE_METHODS.find((x) => x.id === t.method);
    return `<option value="${t.method}">${m?.label ?? t.method}</option>`;
  }).join('');
  if (list.some((t) => t.method === ui.method.value)) ui.ghost.value = ui.method.value;
  if (ui.caseSel.value === 'J') setStatus('Case J draws its course per episode: the loaded J model may not be the course of the J traces.');
}

function describeMethod() {
  const legend = $('legend');
  if (state.section === 'reference') {
    const m = REFERENCE_METHODS.find((x) => x.id === ui.method.value) ?? REFERENCE_METHODS[0];
    view.color = getComputedStyle(document.documentElement).getPropertyValue(m.color.slice(4, -1)).trim();
    ui.access.innerHTML = `<strong>${m.label}</strong> reads: ${m.access} ${m.privileged ? '<span class="priv">Privileged.</span>' : ''}<br><small>${m.code}</small>`;
    legend.innerHTML = `<span style="--c:${view.color}">Live: ${m.label}</span>` + (ui.ghost.value ? `<span class="dashed" style="--c:var(--ghost)">Recorded: ${ui.ghost.selectedOptions[0].text}</span>` : '');
    renderRecorded();
  } else {
    view.color = getComputedStyle(document.documentElement).getPropertyValue('--m1').trim();
    const p = state.policies.find((x) => x.id === ui.method.value);
    ui.access.innerHTML = p ? `<strong>${p.label}</strong> reads: ${CTRL_ACCESS}` : '';
    legend.innerHTML = p ? `<span style="--c:${view.color}">Live: ${p.label}</span>` : '';
  }
}

function reset() {
  if (!state.ready || !ui.method.value) return;
  strip.reset();
  describeMethod();
  send('reset', { method: ui.method.value, clock: ui.clock.value, ghostMethod: ui.ghost.value || null, initial: 'trace' });
  send('set', { target: +ui.target.value, timeScale: +ui.scale.value, friction: +$('friction').value });
}

// ---------------------------------------------------------------- renders ----
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? '–' : typeof v === 'number' ? v.toFixed(d) : String(v));
const rows = (el, pairs, cls = () => '') => {
  el.querySelector('tbody').innerHTML = pairs.map(([k, v], i) => `<tr class="${cls(i)}"><td>${k}</td><td>${v}</td></tr>`).join('');
};

function renderStats(s) {
  if (!s) return rows($('stats'), []);
  rows($('stats'), [
    ['Decisions', s.decisions],
    ['Distance', `${fmt(s.distance, 2)} m`],
    ['Speed RMSE (time-weighted)', `${fmt(s.speedRmse)} m/s`],
    ['Saturation (time-weighted)', `${fmt(100 * s.saturation, 1)} %`],
    ['Mean decision time', `${fmt(s.decisionMs, 2)} ms`],
    ...(state.section === 'reference' ? [['MPC replans', s.plans]] : [['Cheetah-run reward (integrated)', fmt(s.reward, 2)]]),
    ['Real-time factor', s.realtime ? `${fmt(s.realtime, 2)}×` : '–'],
  ]);
}

const NAMES = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot'];
function renderDiag(d, action) {
  rows($('diag'), d ? Object.entries(d).map(([k, v]) => [k.replaceAll('_', ' '), fmt(v, 4)]) : []);
  const bars = $('action-bars');
  if (!bars.children.length) bars.innerHTML = NAMES.map((n) => `<div data-name="${n}"><i></i></div>`).join('');
  if (!action) return;
  [...bars.children].forEach((b, i) => {
    const a = Math.max(-1, Math.min(1, action[i])), bar = b.firstChild;
    bar.style.top = a >= 0 ? `${50 - 50 * a}%` : '50%';
    bar.style.height = `${50 * Math.abs(a)}%`;
    b.classList.toggle('sat', Math.abs(a) > 0.99);
    b.title = `${NAMES[i]}: ${a.toFixed(3)}`;
  });
}

function renderSelfTests(tests) {
  $('selftests').innerHTML = tests.map((t) =>
    `<li class="${t.ok ? 'ok' : 'bad'}"><b>${t.ok ? 'Pass' : 'Fail'}</b>${t.name}<small>${t.detail ?? ''}</small></li>`).join('');
}

function renderGolden(m) {
  const pairs = m.results.map((r) => [r.name, r.value === null ? r.note : `${r.value.toExponential(2)} (tol ${r.tol.toExponential(0)})`]);
  rows($('golden'), [[`Sample: case ${m.caseId}, ${m.clock} clock, ${m.method}`, ''], ...pairs],
    (i) => (i === 0 ? '' : m.results[i - 1].value === null ? '' : m.results[i - 1].value <= m.results[i - 1].tol ? 'pass' : 'fail'));
  setStatus('Golden check finished. Press Reset episode to continue simulating.');
}

async function loadResults() {
  try {
    const r = await fetchAsset('reference/results.json');
    state.results = r.ok ? await r.json() : null;
  } catch { state.results = null; }
  renderRecorded();
}

function renderRecorded() {
  if (!state.results) return rows($('recorded'), [['results.json', 'not exported']]);
  const row = state.results.rows.find((r) => r.case === ui.caseSel.value && r.clock === ui.clock.value && r.method === ui.method.value);
  if (!row) return rows($('recorded'), [['No recorded row', '']]);
  const pm = row.path_metrics ?? {};
  rows($('recorded'), [
    ['Episodes in this cell', state.results.episodes_per_cell],
    ['End', pm.terminal_reason ?? (pm.timeout ? 'timeout' : '–')],
    ['Distance travelled', `${fmt(pm.distance_travelled, 2)} m`],
    ['Speed RMSE', `${fmt(pm.speed_rmse)} m/s`],
    ['Saturation (time mean)', `${fmt(100 * row.saturation_time_mean, 1)} %`],
    ['Wall time (Python)', `${fmt(row.wall_seconds, 1)} s`],
  ]);
}

// ------------------------------------------------------------- explainers ----
let explainSelected = MODULES[0].id;
function renderExplainList() {
  $('explain-list').innerHTML = MODULES.map((m) =>
    `<button data-id="${m.id}" aria-pressed="${m.id === explainSelected}">${m.title}<small>${m.type} · ${m.sections.join(', ')}</small></button>`).join('');
  showExplainer(explainSelected);
}

async function showExplainer(id) {
  explainSelected = id;
  document.querySelectorAll('#explain-list button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.id === id));
  const m = MODULES.find((x) => x.id === id);
  const frame = $('explain-frame');
  const url = `${BASE}explain/${id}.html`;
  let exists = false;
  try {
    const r = await fetch(url, { method: 'HEAD' });
    exists = r.ok && (r.headers.get('content-type') ?? '').includes('html');
  } catch { /* offline */ }
  const code = `<p class="hint">Source: ${m.code.map((c) => `<code>${c}</code>`).join(', ')}</p>`;
  if (exists) {
    frame.innerHTML = `<h2>${m.title}</h2>${code}<iframe src="${url}" title="${m.title}"></iframe>`;
  } else {
    frame.innerHTML = `<h2>${m.title}</h2>${code}
      <p>Not generated yet. Paste this into an agent with the archify skill, then save its HTML as <code>public/explain/${id}.html</code>.</p>
      <pre id="prompt-text"></pre><div class="buttons"><button id="copy-prompt">Copy prompt</button></div>`;
    $('prompt-text').textContent = m.prompt;
    $('copy-prompt').onclick = async () => {
      await navigator.clipboard.writeText(m.prompt);
      $('copy-prompt').textContent = 'Copied';
    };
  }
}

// ---------------------------------------------------------------- controls ---
window.addEventListener('hashchange', route);
$('explain-list').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) showExplainer(b.dataset.id); });
ui.method.addEventListener('change', () => { fillGhosts(); reset(); });
ui.caseSel.addEventListener('change', () => { state.section = null; route(); });
ui.backend.addEventListener('change', () => {
  const url = new URL(location.href);
  if (ui.backend.value === 'js') url.searchParams.set('backend', 'js'); else url.searchParams.delete('backend');
  history.replaceState(null, '', url);
  state.section = null;
  route();
});
ui.clock.addEventListener('change', () => { fillGhosts(); reset(); });
ui.ghost.addEventListener('change', reset);
$('reset').onclick = reset;
ui.play.onclick = () => send(state.running ? 'pause' : 'start');
$('step').onclick = () => send('step');
ui.target.oninput = () => { $('target-out').textContent = `${(+ui.target.value).toFixed(1)} m/s`; send('set', { target: +ui.target.value }); };
ui.scale.oninput = () => { $('scale-out').textContent = `${(+ui.scale.value).toFixed(1)}×`; send('set', { timeScale: +ui.scale.value }); };
$('friction').oninput = (e) => { $('fric-out').textContent = `${(+e.target.value).toFixed(2)}×`; send('set', { friction: +e.target.value }); };
$('force').oninput = (e) => { $('force-out').textContent = `${e.target.value} N`; };
$('dur').oninput = (e) => { $('dur-out').textContent = `${(+e.target.value).toFixed(2)} s`; };
document.querySelectorAll('[data-push]').forEach((b) => b.addEventListener('click', () => {
  const F = +$('force').value, duration = +$('dur').value, lever = 0.3;
  // Planar cheetah: x forward, z up; +y torque pitches the nose down.
  const dir = { back: [-F, 0, 0], fwd: [F, 0, 0], up: [0, F, 0], down: [0, -F, 0], pitch: [0, 0, F * lever], unpitch: [0, 0, -F * lever] }[b.dataset.push];
  send('push', { fx: dir[0], fz: dir[1], ty: dir[2], duration });
}));
$('golden-fixed').onclick = () => send('golden', { clock: 'fixed' });
$('golden-irregular').onclick = () => send('golden', { clock: 'irregular' });
window.addEventListener('resize', () => view.draw(state.frame));

route();

// ------------------------------------------------------------ local assets ---
// Checkpoint-derived files (gait, policies) can stay off the public site:
// lab members pick their exported assets folder and the page reads it locally.
$('local-assets').addEventListener('change', (e) => {
  const entries = assetEntriesFromFiles(e.target.files);
  if (!entries.length) {
    setStatus('No files under an "assets" folder. Pick public/assets (or a folder named assets).', true);
    return;
  }
  setLocalAssets(entries);
  send('local-assets', { entries });
  $('local-assets-out').textContent = `${entries.length} local files in use (not uploaded).`;
  state.section = null;
  route();
});
