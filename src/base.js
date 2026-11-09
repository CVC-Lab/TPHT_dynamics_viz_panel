// Where the site lives: '/' in `npm run dev`, '/<repo>/' on a GitHub Pages
// project site (set by BASE_PATH at build time, see vite.config.js).
export const BASE = (import.meta.env && import.meta.env.BASE_URL) || '/';

// Files a lab member picked from their own exported assets folder. They are
// never uploaded: they stay in this tab and take precedence over deployed files.
const local = new Map();

// entries: [relativePathUnderAssets, Blob] pairs, e.g. ['reference/gait/F.json', file]
export function setLocalAssets(entries) {
  local.clear();
  for (const [path, blob] of entries) local.set(path, blob);
}

export const localAssetCount = () => local.size;

// path is relative to assets/, e.g. 'reference/manifest.json'.
export async function fetchAsset(path) {
  const blob = local.get(path);
  if (blob) return new Response(blob);
  return fetch(`${BASE}assets/${path}`);
}

// A folder picked with <input webkitdirectory> reports paths like
// 'assets/reference/gait/F.json' or 'public/assets/...'; keep the part after 'assets/'.
export function assetEntriesFromFiles(files) {
  const out = [];
  for (const f of files) {
    const rel = f.webkitRelativePath || f.name;
    const i = rel.indexOf('assets/');
    if (i >= 0) out.push([rel.slice(i + 'assets/'.length), f]);
  }
  return out;
}
