#!/usr/bin/env python3
"""Export browser assets from the TPHT CT-RL benchmark bundle.

Subcommands and what they need:
  traces       numpy           validation/mujoco_reference/traces/*.json -> float32 binaries
  samples      (stdlib)        validation/reference_samples/{fixed,irregular}/*.json (golden checks)
  results      (stdlib)        slim copy of validation/mujoco_reference/results.json
  ref-model    mujoco==3.17.0  models/*.mjb + metadata (ids, ground profile via mj_ray)
  gait         torch + bundle  saved gait, limits, CEM bounds and options from a pure-CPHT final.pt
  ctrl-model   mujoco          flattened dm_control cheetah XML + metadata (reads dm_control's
                               cheetah.xml from disk; never imports dm_control, so it works headless)
  ctrl-policy  torch, onnx     CT-RL StochasticActor checkpoint -> deterministic ONNX actor

Every subcommand updates public/assets/<section>/manifest.json.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'public' / 'assets'
TRACE_COLUMNS = (['time', 'h'] + [f'q{i}' for i in range(9)] + [f'v{i}' for i in range(9)]
                 + [f'a{i}' for i in range(6)]
                 + ['progress', 'com_speed', 'saturation', 'weighted_acceleration_residual'])
METHODS = ['learned_feedback', 'mujoco_ls', 'mujoco_fd_ls', 'learned_mpc', 'mujoco_mpc']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update_manifest(path, patch):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v
    merge(data, patch)
    path.write_text(json.dumps(data, indent=2) + '\n')


# ----------------------------------------------------------------- traces ----
def cmd_traces(args):
    import numpy as np
    src = Path(args.bundle) / 'validation' / 'mujoco_reference' / 'traces'
    out = Path(args.out) / 'reference' / 'traces'
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(src.glob('*.json')):
        case, clock, rest = path.stem.split('_', 2)
        method = next(m for m in METHODS if rest.startswith(m + '_'))
        rows = json.loads(path.read_text())
        table = np.array([[r['time'], r['h'], *r['q'], *r['v'], *r['action'], r['progress'], r['com_speed'],
                           r['saturation'], r.get('weighted_acceleration_residual', float('nan'))]
                          for r in rows], dtype=np.float32)
        assert table.shape[1] == len(TRACE_COLUMNS)
        name = f'{case}_{clock}_{method}.f32'
        table.tofile(out / name)
        entries.append(dict(case=case, clock=clock, method=method, file=name, rows=len(rows),
                            source=path.name, source_sha256=sha256(path)))
        print(f'{path.name}: {len(rows)} rows -> {name}')
    (out / 'manifest.json').write_text(json.dumps(dict(columns=TRACE_COLUMNS, dtype='float32', list=entries), indent=2))


# ---------------------------------------------------------------- samples ----
def cmd_samples(args):
    src = Path(args.bundle) / 'validation' / 'reference_samples'
    out = Path(args.out) / 'reference' / 'samples'
    cases = {}
    for clock in ('fixed', 'irregular'):
        for path in sorted((src / clock).glob('*_transition.json')):
            (out / clock).mkdir(parents=True, exist_ok=True)
            shutil.copy(path, out / clock / path.name)
            case = path.stem.split('_')[0]
            cases.setdefault(case, {})['sample'] = f'samples/fixed/{case}_transition.json'
    update_manifest(Path(args.out) / 'reference' / 'manifest.json', dict(cases=cases, target_speed=3.0))
    print('samples:', sorted(cases))


# ---------------------------------------------------------------- results ----
def slim_row(row, keep):
    out = {k: row.get(k) for k in keep}
    # The per-step trajectory is already in the binary traces.
    out['path_metrics'] = {k: v for k, v in (row.get('path_metrics') or {}).items() if k != 'trajectory'}
    return out


def cmd_results(args):
    src = Path(args.bundle) / 'validation' / 'mujoco_reference' / 'results.json'
    data = json.loads(src.read_text())
    keep = ['case', 'clock', 'method', 'seed', 'episode', 'path_metrics', 'wall_seconds', 'decisions',
            'saturation_time_mean', 'information_access', 'checkpoint_sha256', 'trace_file']
    slim = dict(schema=data['schema'], parameters=data['parameters'], mujoco_version=data['mujoco_version'],
                privileged_access=data['privileged_access'], episodes_per_cell=data['episodes_per_cell'],
                rows=[slim_row(r, keep) for r in data['rows']], source_sha256=sha256(src))
    out = Path(args.out) / 'reference' / 'results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(slim, indent=1))
    print(f'results: {len(slim["rows"])} rows')


# -------------------------------------------------------------- ref-model ----
def ground_profile(mujoco, model, x_range, step=0.02, z_top=10.0):
    import numpy as np
    data = mujoco.MjData(model)
    data.qpos[1] = 60.0  # lift the robot (rootz slide) above the rays
    mujoco.mj_forward(model, data)
    xs = np.arange(x_range[0], x_range[1] + 1e-9, step)
    zs = np.empty_like(xs)
    geomid = np.array([-1], dtype=np.int32)
    normal = np.zeros(3)
    down = np.array([0.0, 0.0, -1.0])

    def ray(x):
        start = np.array([x, 0.0, z_top])
        try:
            return mujoco.mj_ray(model, data, start, down, None, 1, -1, geomid)
        except TypeError:  # newer signature with a surface-normal output
            return mujoco.mj_ray(model, data, start, down, None, 1, -1, geomid, normal)

    for i, x in enumerate(xs):
        dist = ray(x)
        zs[i] = z_top - dist if dist >= 0 else np.nan
    return xs, zs


def model_meta(mujoco, model, x_range, legacy_step=True):
    import numpy as np
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    xs, zs = ground_profile(mujoco, model, x_range)
    ok = np.isfinite(zs)
    planes = [float(model.geom_pos[g, 2]) for g in range(model.ngeom)
              if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    return dict(torso_body_id=int(torso), timestep=float(model.opt.timestep), legacy_step=legacy_step,
                mujoco_version=mujoco.__version__, nq=model.nq, nv=model.nv, nu=model.nu,
                jnt_range=model.jnt_range.tolist(), actuator_gear=model.actuator_gear[:, 0].tolist(),
                planes=planes,
                profile=dict(x=[round(float(x), 4) for x in xs[ok]], z=[round(float(z), 5) for z in zs[ok]]))


def cmd_ref_model(args):
    import mujoco
    src = Path(args.bundle) / 'validation' / 'reference_samples' / 'models'
    out = Path(args.out) / 'reference' / 'models'
    out.mkdir(parents=True, exist_ok=True)
    cases = {}
    for path in sorted(src.glob('*_transition.mjb')):
        case = path.stem.split('_')[0]
        model = mujoco.MjModel.from_binary_path(str(path))
        shutil.copy(path, out / path.name)
        meta = model_meta(mujoco, model, args.x_range)
        meta['source_sha256'] = sha256(path)
        (out / f'{case}_meta.json').write_text(json.dumps(meta))
        cases[case] = dict(model=dict(kind='mjb', file=f'models/{path.name}'), meta=f'models/{case}_meta.json')
        print(f'{case}: {path.name}, timestep {meta["timestep"]}, profile points {len(meta["profile"]["x"])}')
    update_manifest(Path(args.out) / 'reference' / 'manifest.json', dict(cases=cases))
    print(f'Use @mujoco/mujoco@{mujoco.__version__} in package.json: .mjb files only load in the same MuJoCo version.')


# ------------------------------------------------------------------- gait ----
def cmd_gait(args):
    sys.path.insert(0, str(Path(args.bundle).resolve()))
    from mujoco_reference.common import read_checkpoint
    from pure_cpht.controller import Controller
    saved = read_checkpoint(args.checkpoint)
    params = json.loads((Path(args.bundle) / 'validation' / 'mujoco_reference' / 'results.json').read_text())['parameters']
    options = dict(saved['settings']['pure'])
    options.update(candidates=params['candidates'], elites=min(3, params['candidates'] - 1),
                   iterations=params['iterations'], horizon_steps=params['horizon_steps'],
                   replan_seconds=params['replan_seconds'])
    ctl = Controller(saved['pure_model'], options, saved['neutral'], 0, gait=saved['gait'], feedback_only=True)
    tolist = lambda t: t.detach().cpu().double().numpy().tolist()
    cfg = dict(gait=tolist(ctl.mean), limits=tolist(ctl.robot.limits), low=tolist(ctl.low), high=tolist(ctl.high),
               scale=tolist(ctl.scale), options={k: v for k, v in options.items() if isinstance(v, (int, float, bool, str))},
               checkpoint_sha256=saved['_sha256'], planning_parameters=params)
    out = Path(args.out) / 'reference' / 'gait' / f'{args.case}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=1))
    update_manifest(Path(args.out) / 'reference' / 'manifest.json', dict(cases={args.case: dict(gait=f'gait/{args.case}.json')}))
    print(f'gait for case {args.case}: {out}')


# ------------------------------------------------------------- ctrl-model ----
def locate_dm_control_cheetah():
    """Path to dm_control/suite/cheetah.xml, found WITHOUT importing dm_control.

    Importing dm_control.suite initializes an OpenGL backend (GLFW first), which
    fails on headless machines with "X11: The DISPLAY environment variable is
    missing". The export only needs the XML file, so look the package up on disk.
    find_spec on a top-level package locates it without running its __init__.
    """
    import importlib.util
    spec = importlib.util.find_spec('dm_control')
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit('dm_control is not installed here. Install it, or pass --xml /path/to/cheetah.xml '
                         '(dm_control/suite/cheetah.xml with its common/ folder next to it).')
    path = Path(next(iter(spec.submodule_search_locations))) / 'suite' / 'cheetah.xml'
    if not path.exists():
        raise SystemExit(f'Expected {path}; pass --xml to point at cheetah.xml explicitly.')
    return path


def dm_control_version():
    from importlib import metadata
    try:
        return metadata.version('dm_control')
    except metadata.PackageNotFoundError:
        return None


def cmd_ctrl_model(args):
    import mujoco
    xml_path = Path(args.xml) if args.xml else locate_dm_control_cheetah()
    # from_xml_path resolves ./common/*.xml includes relative to the file,
    # matching cheetah.get_model_and_assets(); no rendering context is created.
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    out = Path(args.out) / 'ctrl'
    out.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(out / 'cheetah.xml'), model)  # flattened: includes resolved
    meta = model_meta(mujoco, model, args.x_range)
    meta.update(
        obs_note='dm_control cheetah: position = qpos[1:], velocity = qvel',
        source_xml=str(xml_path),
        # Same bytes physical_fj.mechanics.NominalRobot hashes, so the two can be compared.
        source_xml_sha256=sha256(xml_path),
        dm_control_version=dm_control_version(),
    )
    (out / 'cheetah_meta.json').write_text(json.dumps(meta))
    update_manifest(out / 'manifest.json', dict(model=dict(kind='xml', file='cheetah.xml'), meta='cheetah_meta.json'))
    print(f'ctrl model: {out / "cheetah.xml"} from {xml_path} (dm_control {meta["dm_control_version"]}), '
          f'timestep {meta["timestep"]}')


# ------------------------------------------------------------ ctrl-policy ----
def cmd_ctrl_policy(args):
    import numpy as np
    import torch
    from gymnasium import spaces
    sys.path.insert(0, str(Path(args.ctrl_repo).resolve()))
    from models.actor import StochasticActor
    state = torch.load(args.checkpoint, map_location='cpu')
    actor_state = state['actor'] if 'actor' in state else state
    body = sorted((k for k in actor_state if k.startswith('body.') and k.endswith('.weight')),
                  key=lambda k: int(k.split('.')[1]))
    if not body or 'mu.weight' not in actor_state:
        raise SystemExit(f'Not a StochasticActor state dict (keys: {sorted(actor_state)[:12]}...). '
                         'Deterministic TD3-style actors need their own wrapper.')
    outs = [actor_state[k].shape[0] for k in body]
    obs_dim = actor_state[body[0]].shape[1]
    act_dim = actor_state['mu.weight'].shape[0]
    net_arch = outs[:-1]  # create_mlp(hidden=net_arch) + output layer of size net_arch[-1]
    obs_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
    act_space = spaces.Box(-1.0, 1.0, (act_dim,), np.float32)
    actor = StochasticActor(obs_space, act_space, net_arch, device='cpu')
    actor.load_state_dict(actor_state, strict=True)  # strict load verifies the inferred structure
    actor.eval()

    class Deterministic(torch.nn.Module):
        def __init__(self, a):
            super().__init__()
            self.a = a

        def forward(self, obs):
            return self.a(obs, deterministic=True)[0]

    wrapper = Deterministic(actor)
    out = Path(args.out) / 'ctrl' / 'policies'
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / f'{args.algo}.onnx'
    example = torch.zeros(1, obs_dim)
    torch.onnx.export(wrapper, example, str(onnx_path), input_names=['obs'], output_names=['action'],
                      dynamic_axes={'obs': {0: 'batch'}, 'action': {0: 'batch'}}, opset_version=17)
    check = None
    try:
        import onnxruntime as ort
        x = torch.randn(64, obs_dim)
        ref = wrapper(x).detach().numpy()
        got = ort.InferenceSession(str(onnx_path)).run(['action'], {'obs': x.numpy()})[0]
        check = float(np.max(np.abs(ref - got)))
        print(f'ONNX vs torch max abs difference: {check:.2e}')
    except ImportError:
        print('onnxruntime not installed; skipped numerical check')
    if obs_dim != 17:
        print(f'Warning: obs_dim={obs_dim}; edit obs_layout in {args.algo}.json to match the training wrapper.')
    meta = dict(algo=args.algo, label=args.label or args.algo, obs_dim=obs_dim, act_dim=act_dim, net_arch=net_arch,
                obs_layout=[dict(key='position', source='qpos', start=1, size=8),
                            dict(key='velocity', source='qvel', start=0, size=9)],
                input_name='obs', output_name='action', onnx_vs_torch_max_abs=check,
                checkpoint_sha256=sha256(args.checkpoint))
    (out / f'{args.algo}.json').write_text(json.dumps(meta, indent=1))
    manifest = Path(args.out) / 'ctrl' / 'manifest.json'
    current = json.loads(manifest.read_text()) if manifest.exists() else {}
    policies = [p for p in current.get('policies', []) if p['id'] != args.algo]
    policies.append(dict(id=args.algo, label=meta['label'], onnx=f'policies/{args.algo}.onnx', meta=f'policies/{args.algo}.json'))
    update_manifest(manifest, dict(policies=policies))
    print('policy:', onnx_path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(DEFAULT_OUT))
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('traces', 'samples', 'results', 'ref-model', 'gait'):
        s = sub.add_parser(name)
        s.add_argument('--bundle', required=True, help='Path to TPHT_CT_RL_Benchmark/')
    sub.choices['ref-model'].add_argument('--x-range', type=float, nargs=2, default=[-5.0, 40.0])
    sub.choices['gait'].add_argument('--checkpoint', required=True, help='pure_cpht final.pt from sensor.zip')
    sub.choices['gait'].add_argument('--case', required=True, choices=['F', 'J'])
    s = sub.add_parser('ctrl-model')
    s.add_argument('--x-range', type=float, nargs=2, default=[-5.0, 120.0])
    s.add_argument('--xml', help='cheetah.xml to use instead of the installed dm_control copy')
    s = sub.add_parser('ctrl-policy')
    s.add_argument('--checkpoint', required=True)
    s.add_argument('--algo', required=True, help='e.g. ct_sac, sac')
    s.add_argument('--label')
    s.add_argument('--ctrl-repo', required=True, help='Path to ct-rl-master/')
    args = p.parse_args(argv)
    {'traces': cmd_traces, 'samples': cmd_samples, 'results': cmd_results, 'ref-model': cmd_ref_model,
     'gait': cmd_gait, 'ctrl-model': cmd_ctrl_model, 'ctrl-policy': cmd_ctrl_policy}[args.cmd](args)


if __name__ == '__main__':
    main()
