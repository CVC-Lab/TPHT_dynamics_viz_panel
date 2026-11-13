// A built-in open-loop gait for the CT-RL tab, so the page shows MuJoCo
// running before any trained policy has been exported. It is NOT a baseline:
// fixed sinusoids on the six cheetah actuators (back thigh/shin/foot, front
// thigh/shin/foot), the front leg a quarter cycle behind the back leg. It
// reads only the simulation time.
//
// Parameters from a native sweep on the dm_control cheetah (MuJoCo 4.0, the same
// on 3.17; 12 s, default start pose): about 14.7 m forward (1.2 m/s) with torso pitch
// within 0.3 rad. A half-cycle offset at amplitude 0.8 flips the robot.
// tpht_lab/check.py rolls out the same formula natively.

export const DEMO_POLICY = Object.freeze({
  id: 'demo-sine',
  label: 'Demo: open-loop sine gait (not a trained policy)',
  builtin: true,
});

export class SineGait {
  // nu: number of actuators (6 for the dm_control cheetah).
  constructor(nu, { frequency = 2.5, amplitude = 0.7, jointLag = 0.6, legOffset = Math.PI / 2 } = {}) {
    this.nu = nu;
    this.frequency = frequency;
    this.amplitude = amplitude;
    // Phase per actuator: thigh, shin, foot each jointLag behind the last; front leg offset by legOffset.
    this.phase = Array.from({ length: nu }, (_, i) => (i % 3) * jointLag + (i >= 3 ? legOffset : 0));
    this.diagnostics = {};
  }

  act(data) {
    const started = performance.now();
    const w = 2 * Math.PI * this.frequency * data.time;
    const action = new Float32Array(this.nu);
    for (let i = 0; i < this.nu; i++) action[i] = this.amplitude * Math.sin(w + this.phase[i]);
    this.diagnostics = {
      decision_ms: performance.now() - started,
      action_saturation: 0,
    };
    return action;
  }
}
