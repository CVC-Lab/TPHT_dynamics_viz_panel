// What each method may read, and whether it runs live in the browser yet.
// Information-access text mirrors validation/mujoco_reference/results.json.

export const REFERENCE_METHODS = [
  {
    id: 'mujoco_ls', label: 'MuJoCo LS', color: 'var(--m2)', live: true, privileged: true,
    access: 'Privileged current MuJoCo mechanics and contact solve; same inferred scan in acceleration target.',
    code: 'mujoco_reference/controller.py · NativeController.fit',
  },
  {
    id: 'mujoco_fd_ls', label: 'MuJoCo FD-LS', color: 'var(--m3)', live: true, privileged: true,
    access: 'Privileged current MuJoCo mechanics and contact response under perturbed controls; same inferred scan.',
    code: 'mujoco_reference/controller.py · NativeController.fit(finite_difference=True)',
  },
  {
    id: 'mujoco_mpc', label: 'MuJoCo MPC', color: 'var(--m5)', live: true, privileged: true,
    access: 'Privileged full compiled terrain, current friction and mechanics throughout prediction; same inferred scan in target and cost.',
    code: 'mujoco_reference/controller.py · NativeController.rollout + pure_cpht Controller.plan',
  },
  {
    id: 'learned_feedback', label: 'Learned feedback', color: 'var(--m1)', live: false, privileged: false,
    access: 'Causal state/force history, inferred terrain and friction, calibrated pH mechanics.',
    code: 'pure_cpht/controller.py · Controller.feedback',
    pending: 'Needs the pure-CPHT model exported from sensor.zip; replay its recorded trace as a ghost for now.',
  },
  {
    id: 'learned_mpc', label: 'Learned MPC', color: 'var(--m4)', live: false, privileged: false,
    access: 'Same causal input; rollout uses the calibrated pH/contact model and inferred scan.',
    code: 'pure_cpht/controller.py · Controller.plan + rollout',
    pending: 'Needs the pure-CPHT model exported from sensor.zip; replay its recorded trace as a ghost for now.',
  },
];

export const CTRL_ACCESS = 'Proprioceptive dm_control observation (qpos without root x, qvel). No model access.';

// Browser-run differences that must stay visible next to any live number.
export const LIVE_CAVEATS = [
  'Scan comes from the exported ground profile (privileged); the causal environment observer is not ported.',
  'MPC noise uses a seeded JS generator, so candidates differ from the Python run.',
  'Live numbers are for debugging only. Benchmark results come from the Python pipeline.',
];
