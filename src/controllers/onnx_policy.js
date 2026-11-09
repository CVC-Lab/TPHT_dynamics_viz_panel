// Deterministic CT-RL actor exported by tools/export_web_assets.py ctrl-policy.
// The ONNX graph already contains tanh squashing and action-bound rescaling.

import { BASE } from '../base.js';

let ortPromise = null;
async function loadOrt() {
  if (!ortPromise) {
    ortPromise = import('onnxruntime-web').then((ort) => {
      ort.env.wasm.wasmPaths = `${BASE}ort/`; // copied by tools/copy-ort-wasm.mjs
      ort.env.wasm.numThreads = 1;
      return ort;
    });
  }
  return ortPromise;
}

// Builds the flattened dm_control observation described by meta.obs_layout,
// e.g. [{key:'position', source:'qpos', start:1, size:8}, {key:'velocity', source:'qvel', start:0, size:9}].
export function buildObservation(layout, data) {
  const size = layout.reduce((s, p) => s + p.size, 0);
  const obs = new Float32Array(size);
  let k = 0;
  for (const part of layout) {
    const src = data[part.source];
    for (let i = 0; i < part.size; i++) obs[k++] = src[part.start + i];
  }
  return obs;
}

export class OnnxPolicy {
  // model: the .onnx file's bytes (Uint8Array), from the deployed site or a local folder.
  static async create(model, meta) {
    const ort = await loadOrt();
    const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
    return new OnnxPolicy(ort, session, meta);
  }

  constructor(ort, session, meta) {
    this.ort = ort;
    this.session = session;
    this.meta = meta;
    this.input = meta.input_name ?? session.inputNames[0];
    this.output = meta.output_name ?? session.outputNames[0];
    this.diagnostics = {};
  }

  async act(data) {
    const started = performance.now();
    const obs = buildObservation(this.meta.obs_layout, data);
    const feeds = { [this.input]: new this.ort.Tensor('float32', obs, [1, obs.length]) };
    const out = await this.session.run(feeds);
    const action = Float32Array.from(out[this.output].data);
    this.diagnostics = {
      decision_ms: performance.now() - started,
      action_saturation: action.reduce((s, x) => s + (Math.abs(x) > 0.99), 0) / action.length,
    };
    return action;
  }
}
