// A minimal in-browser stand-in for node:test and node:assert, so the
// controller tests run with no node at all: open tests/index.html through
// `tpht-lab serve` (any browser), or run `tpht-lab test` (headless Chromium
// from cvcpkg). Test files register with test(); index.html calls run().

const registry = [];

// test(name, fn) or test(name, {skip}, fn). skip may be a boolean or an async
// function returning one (fetch-based existence checks are async here). A test
// may return a string, shown with its result (measured values, for example).
export function test(name, opts, fn) {
  if (typeof opts === 'function') { fn = opts; opts = {}; }
  registry.push({ name, opts: opts ?? {}, fn });
}

class AssertionError extends Error {}

export const assert = {
  ok(value, message = 'expected a truthy value') {
    if (!value) throw new AssertionError(message);
  },
  equal(actual, expected, message) {
    if (actual !== expected) {
      throw new AssertionError(message ?? `expected ${String(expected)}, got ${String(actual)}`);
    }
  },
};

// Paths are relative to the calling module, like new URL(path, import.meta.url).
export async function loadJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return r.json();
}

export async function loadBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

export async function exists(url) {
  try { return (await fetch(url, { method: 'HEAD' })).ok; } catch { return false; }
}

// Runs every registered test in order and renders the results into `out`.
// Sets <body data-status="pass|fail"> and #summary's text, which `tpht-lab
// test` reads from the headless browser's DOM.
export async function run(out) {
  const results = [];
  for (const t of registry) {
    const started = performance.now();
    let status = 'pass', detail = '';
    try {
      const skip = typeof t.opts.skip === 'function' ? await t.opts.skip() : t.opts.skip;
      if (skip) status = 'skip';
      else {
        const note = await t.fn();
        if (typeof note === 'string') detail = note;
      }
    } catch (e) {
      status = 'fail';
      detail = e?.message ?? String(e);
    }
    results.push({ name: t.name, status, detail, ms: Math.round(performance.now() - started) });
  }
  const failed = results.filter((r) => r.status === 'fail').length;
  const passed = results.filter((r) => r.status === 'pass').length;
  const skipped = results.filter((r) => r.status === 'skip').length;
  for (const r of results) {
    const li = document.createElement('li');
    li.className = r.status;
    li.textContent = `${r.status.toUpperCase()}  ${r.name}  (${r.ms} ms)${r.detail ? ` - ${r.detail}` : ''}`;
    out.appendChild(li);
  }
  const summary = `${passed} passed, ${failed} failed, ${skipped} skipped`;
  document.getElementById('summary').textContent = summary;
  document.body.dataset.status = failed ? 'fail' : 'pass';
  return { results, summary, failed };
}
