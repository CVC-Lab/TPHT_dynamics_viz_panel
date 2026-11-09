import { defineConfig } from 'vite';

export default defineConfig({
  // '/' for local dev; '/<repo>/' for a GitHub Pages project site.
  // The Pages workflow sets BASE_PATH; to test that layout locally:
  //   BASE_PATH=/rl-baseline-lab/ npm run build && npm run preview
  base: process.env.BASE_PATH || '/',
  // Both packages ship .wasm files that must be served as-is, not pre-bundled.
  optimizeDeps: { exclude: ['@mujoco/mujoco', 'onnxruntime-web'] },
  worker: { format: 'es' },
  // Keep Vite's compiled JS apart from the exported data in public/assets.
  build: { target: 'es2022', assetsDir: 'static' },
});
