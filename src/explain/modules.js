// Explainer modules. Generate each one with archify (run in Claude Code,
// Codex, Cursor or OpenCode with the skill installed), save the HTML as
// public/explain/<id>.html, and the Explainers tab embeds it.
// Prompts name real files and functions so the diagrams stay source-backed.

export const MODULES = [
  {
    id: 'native-decision',
    title: 'One control decision of the MuJoCo LS and FD-LS references',
    type: 'workflow',
    sections: ['reference'],
    code: [
      'mujoco_reference/controller.py: NativeController.act, fit, desired',
      'mujoco_reference/common.py: acceleration_map, bounded_ls, step_native',
      'pure_cpht/controller.py: Controller.references',
    ],
    prompt: `Use Archify to make a workflow diagram from TPHT_CT_RL_Benchmark/mujoco_reference/controller.py.
Show one call of NativeController.act for method mujoco_ls and mujoco_fd_ls:
phase update from the saved gait frequency, copy of the live MjData, forward (mj_forward + mj_subtreeVel),
desired() acceleration target (speed, clearance from the scan, pitch, joint PD on Controller.references),
then branch: acceleration_map (M^-1 B with the measured constraint force frozen) for mujoco_ls,
or a six-column finite-difference Jacobian with a 0.5 trust box for mujoco_fd_ls.
Both branches join at bounded_ls (weights 1,4,4,0.15x6, ridge feedback_ridge, box [-1,1]), then diagnostics,
then the float32 action applied for the hold with dm_control's split step (mj_step2 then mj_step1).
Mark which steps read privileged MuJoCo state. Cite source lines for each node.`,
  },
  {
    id: 'mpc-replan',
    title: 'MuJoCo MPC replanning cycle',
    type: 'sequence',
    sections: ['reference'],
    code: [
      'pure_cpht/controller.py: Controller.plan',
      'mujoco_reference/controller.py: NativeController.rollout',
    ],
    prompt: `Use Archify to make a sequence diagram of one mujoco_mpc replan, every replan_seconds (0.1 s in results.json).
Participants: NativeController.act, Controller.plan (CEM), NativeController.rollout, a copied MjData, bounded_ls.
Sequence: sample 8 candidates of 19 gait coefficients around the mean with per-coefficient std, clamp to [low, high],
keep the previous best as candidate 0; for each candidate copy the live MjData, then for 20 intervals call fit(),
mj_step1, and the hold's split steps while accumulating stage cost (speed, clearance, pitch, effort, scan coverage)
and checking validity (clearance < 0.15, |pitch| > 1.6, |qvel| > 80, joint limits +-0.3); add the progress term;
pick 3 elites, update mean and std (floor 0.015), keep the best candidate as the new gait.
Note that rollouts never advance the live simulation. Cite source lines.`,
  },
  {
    id: 'information-access',
    title: 'What each baseline is allowed to read',
    type: 'dataflow',
    sections: ['reference', 'ctrl'],
    code: [
      'validation/mujoco_reference/results.json: privileged_access',
      'environment_belief/observer.py, environment_belief/runtime.py',
      'MUJOCO_REFERENCE.md: The five controls',
    ],
    prompt: `Use Archify to make a data-flow diagram of information access in the TPHT cheetah comparison.
Sources: true MuJoCo state and contact solve, compiled terrain and current friction, causal environment observer
(environment_belief/observer.py: local quadratic Kalman filter and gated contact-force inverse), saved periodic gait,
calibrated learned pH/contact model from final.pt, and the dm_control proprioceptive observation used by CT-RL actors.
Consumers: learned_feedback, learned_mpc, mujoco_ls, mujoco_fd_ls, mujoco_mpc, and CT-SAC/SAC actors.
Draw a trust boundary called "privileged" around the MuJoCo sources and route each consumer exactly as described
in results.json privileged_access. Add a note that the browser lab currently feeds a privileged ground-profile scan
instead of the causal observer.`,
  },
  {
    id: 'ct-sac-update',
    title: 'CT-SAC decoupled actor-critic update',
    type: 'workflow',
    sections: ['ctrl'],
    code: [
      'ct-rl-master/algorithms/ct_sac.py: CTSAC.train, _policy_act',
      'ct-rl-master/algorithms/off_policy.py: learn, _sample_action, _store_transition',
      'Decoupled Continuous-Time RL via Hamiltonian Flow, Sections 3-4',
    ],
    prompt: `Use Archify to make a workflow diagram of one CT-SAC training iteration from ct-rl-master/algorithms/ct_sac.py
and off_policy.py. Show: environment step with a variable hold time, replay storage of (obs, t, action, reward,
next_obs, next_t), the q-function update from the diffusion generator applied to V, the V update by the Hamiltonian
value flow, and the entropy-regularized actor update. Keep the alternating V/q structure visible, since that is the
paper's departure from coupled martingale losses. Cite source lines; do not invent update equations that are not in
the code.`,
  },
  {
    id: 'web-lab',
    title: 'How this browser lab is put together',
    type: 'architecture',
    sections: ['reference', 'ctrl'],
    code: [
      'src/main.js, src/ui/*',
      'src/backend/sim.worker.js, src/backend/mujoco_env.js',
      'src/controllers/*, tools/export_web_assets.py',
    ],
    prompt: `Use Archify to make an architecture diagram of the rl-baseline-lab repository.
Components: page UI (src/main.js, ui/viewport.js, ui/strip.js), simulation Web Worker (backend/sim.worker.js),
MuJoCo WASM adapter (backend/mujoco_env.js, with self-tests and binding fallbacks), controller ports
(controllers/native.js, bvls.js, gait.js), ONNX policy runner (controllers/onnx_policy.js), static assets
(public/assets), and the Python exporter (tools/export_web_assets.py) that reads the TPHT bundle and CT-RL checkpoints.
Primary path: UI message -> worker -> controller -> MuJoCo step -> frame back to UI. Put the golden-check path
(reference_samples replay) as a secondary route.`,
  },
];
