// Copies ONNX Runtime Web's .wasm/.mjs runtime files to public/ort so the
// worker can load them from /ort/ (see src/controllers/onnx_policy.js).
import { cpSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const dst = join(root, 'public', 'ort');
if (!existsSync(src)) {
  console.log('onnxruntime-web not installed yet; skipping wasm copy');
} else {
  mkdirSync(dst, { recursive: true });
  const files = readdirSync(src).filter((f) => /^ort-wasm.*\.(wasm|mjs)$/.test(f));
  for (const f of files) cpSync(join(src, f), join(dst, f));
  console.log(`Copied ${files.length} ONNX Runtime files to public/ort`);
}
