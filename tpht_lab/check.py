"""Native MuJoCo checks on the same assets the page loads.

The browser's "Binding self-tests" panel (src/backend/mujoco_env.js selfTest)
checks the model and the bindings inside the WebAssembly build. This module runs
the same checks with cvcpkg's native MuJoCo (the mujoco-core Python package),
so a failure in the page can be split into "the asset is wrong" (fails here
too) and "the wasm binding call is wrong" (passes here). It also rolls the
CT-RL model out under the demo sine gait (src/controllers/demo_gait.js), as a
smoke test that the model walks and stays finite.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Result:
    section: str
    name: str
    ok: bool
    detail: str


def _load_model(mujoco, section_dir: Path, spec: dict):
    path = section_dir / spec['file']
    if not path.is_file():
        raise FileNotFoundError(f'missing {path} (git-ignored or not exported)')
    if spec.get('kind') == 'xml':
        return mujoco.MjModel.from_xml_path(str(path)), f'MjModel.from_xml_path({spec["file"]})'
    return mujoco.MjModel.from_binary_path(str(path)), f'MjModel.from_binary_path({spec["file"]})'


def sine_gait(nu: int, t: float, frequency: float = 2.5, amplitude: float = 0.7,
              joint_lag: float = 0.6, leg_offset: float = math.pi / 2):
    """Same actions as SineGait.act in src/controllers/demo_gait.js."""
    w = 2 * math.pi * frequency * t
    return [amplitude * math.sin(w + (i % 3) * joint_lag + (leg_offset if i >= 3 else 0.0)) for i in range(nu)]


def binding_checks(mujoco, model, meta: dict, expect: dict, section: str) -> list[Result]:
    import numpy as np
    out: list[Result] = []
    add = lambda name, ok, detail: out.append(Result(section, name, bool(ok), detail))  # noqa: E731

    if 'dims' in expect:
        dims = (model.nq, model.nv, model.nu)
        add('Model dimensions', dims == tuple(expect['dims']),
            f'nq={dims[0]} nv={dims[1]} nu={dims[2]}, expected {"/".join(map(str, expect["dims"]))}')
    for key in ('nq', 'nv', 'nu'):
        if key in meta:
            add(f'{key} matches metadata', getattr(model, key) == meta[key], f'{getattr(model, key)} vs {meta[key]}')
    if 'timestep' in meta:
        add('Physics timestep', abs(model.opt.timestep - meta['timestep']) < 1e-12,
            f'{model.opt.timestep} s, metadata {meta["timestep"]} s')
    if expect.get('euler'):
        add('Euler integrator', model.opt.integrator == 0, f'integrator={model.opt.integrator}')
    if 'actuator_gear' in meta:
        gear = model.actuator_gear[:, 0].tolist()
        add('Actuator gears match metadata', np.allclose(gear, meta['actuator_gear']), f'{gear}')
    cr = model.actuator_ctrlrange.ravel()
    want = np.tile([-1.0, 1.0], model.nu)
    add('Normalized actuator bounds', np.allclose(cr, want, atol=1e-12),
        'ctrlrange=' + ' '.join(f'{x:.2f}' for x in cr))

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Linear gear map: qfrc_actuator(ctrl) equals the sum of the unit responses.
    def actuator_force(ctrl):
        d = mujoco.MjData(model)
        d.qpos[:] = data.qpos
        d.ctrl[:] = ctrl
        mujoco.mj_forward(model, d)
        return d.qfrc_actuator.copy()
    ctrl = np.array([-0.4 + 0.8 * i / max(1, model.nu - 1) for i in range(model.nu)])
    B = np.column_stack([actuator_force(np.eye(model.nu)[c]) for c in range(model.nu)]) if model.nu else np.zeros((model.nv, 0))
    gear_err = float(np.max(np.abs(actuator_force(ctrl) - B @ ctrl))) if model.nu else 0.0
    add('Linear gear map (qfrc_actuator = B ctrl)', gear_err < 1e-9, f'max error {gear_err:.2e}')

    probe = mujoco.MjData(model)
    probe.qpos[:] = data.qpos
    probe.ctrl[:] = ctrl
    mujoco.mj_forward(model, probe)
    M = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, probe, M)       # MuJoCo 4.x: (m, d, dst)
    except TypeError:
        mujoco.mj_fullM(model, M, probe.qM)    # MuJoCo 3.x: (m, dst, qM)
    asym = float(np.max(np.abs(M - M.T)))
    f = probe.qfrc_actuator + probe.qfrc_passive - probe.qfrc_bias + probe.qfrc_constraint + probe.qfrc_applied
    scale = max(1.0, float(np.max(np.abs(f))))
    balance = float(np.max(np.abs(M @ probe.qacc - f))) / scale
    col = np.zeros(model.nv)
    mulm_err = 0.0
    for j in range(model.nv):
        e = np.zeros(model.nv)
        e[j] = 1.0
        mujoco.mj_mulM(model, probe, col, e)
        mulm_err = max(mulm_err, float(np.max(np.abs(col - M[:, j]))))
    pd = bool(np.all(np.linalg.eigvalsh(0.5 * (M + M.T)) > 0))
    add('Mass matrix (mj_fullM)', asym < 1e-9 and balance < 1e-8 and pd,
        f'asymmetry {asym:.2e}, force balance {balance:.2e} (relative), positive definite {pd}')
    add('mj_mulM agrees with mj_fullM', mulm_err < 1e-9, f'max column error {mulm_err:.2e}')
    return out


def rollout(mujoco, model, meta: dict, seconds: float, section: str) -> Result:
    import numpy as np
    d = mujoco.MjData(model)
    torso = int(meta.get('torso_body_id', 1))
    mujoco.mj_forward(model, d)
    x0 = float(d.xpos[torso, 0])
    steps = int(round(seconds / model.opt.timestep))
    max_pitch = 0.0
    for _ in range(steps):
        d.ctrl[:] = sine_gait(model.nu, d.time)
        mujoco.mj_step(model, d)
        # torso pitch from its rotation matrix (planar cheetah: rotation about y)
        R = d.xmat[torso]
        max_pitch = max(max_pitch, abs(math.atan2(R[2], R[0])))
    finite = bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)))
    dx = float(d.xpos[torso, 0]) - x0
    ok = finite and dx > 0 and max_pitch < math.pi / 2
    return Result(section, f'Demo sine gait rollout ({seconds:g} s, {steps} steps)', ok,
                  f'torso moved {dx:+.2f} m along x, max pitch {max_pitch:.2f} rad, state finite {finite}')


def run(assets: Path, seconds: float = 5.0, site: Path | None = None) -> list[Result]:
    import mujoco
    results: list[Result] = []
    if site is not None and (site / 'build-info.json').is_file():
        info = json.loads((site / 'build-info.json').read_text(encoding='utf-8'))
        results.append(Result('site', 'Same MuJoCo version as the site', info.get('mujoco') == mujoco.__version__,
                              f'native {mujoco.__version__}, site wasm {info.get("mujoco")}'))

    ctrl_dir = assets / 'ctrl'
    if (ctrl_dir / 'manifest.json').is_file():
        man = json.loads((ctrl_dir / 'manifest.json').read_text(encoding='utf-8'))
        meta = json.loads((ctrl_dir / man['meta']).read_text(encoding='utf-8')) if man.get('meta') else {}
        try:
            model, via = _load_model(mujoco, ctrl_dir, man['model'])
            results.append(Result('ctrl', 'Model loader', True, via))
            results += binding_checks(mujoco, model, meta, {}, 'ctrl')
            results.append(rollout(mujoco, model, meta, seconds, 'ctrl'))
        except Exception as e:  # noqa: BLE001 - report and continue
            results.append(Result('ctrl', 'Model loader', False, str(e)))

    ref_dir = assets / 'reference'
    if (ref_dir / 'manifest.json').is_file():
        man = json.loads((ref_dir / 'manifest.json').read_text(encoding='utf-8'))
        for case, c in sorted(man.get('cases', {}).items()):
            if 'model' not in c:
                continue
            section = f'reference/{case}'
            meta_path = ref_dir / c['meta'] if c.get('meta') else None
            meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path and meta_path.is_file() else {}
            try:
                model, via = _load_model(mujoco, ref_dir, c['model'])
            except FileNotFoundError as e:
                results.append(Result(section, 'Model loader (skipped)', True, str(e)))
                continue
            except Exception as e:  # noqa: BLE001 - e.g. an .mjb from another MuJoCo version
                results.append(Result(section, 'Model loader', False, str(e)))
                continue
            results.append(Result(section, 'Model loader', True, via))
            results += binding_checks(mujoco, model, meta, {'dims': [9, 9, 6], 'euler': True}, section)
    if not results:
        results.append(Result('assets', 'Assets found', False, f'no ctrl/ or reference/ manifest under {assets}'))
    return results


def report(results: list[Result], as_json: bool = False) -> int:
    failed = [r for r in results if not r.ok]
    if as_json:
        print(json.dumps([asdict(r) for r in results], indent=1))
    else:
        for r in results:
            print(f'{"PASS" if r.ok else "FAIL"}  [{r.section}] {r.name}: {r.detail}')
        print(f'{len(results) - len(failed)} passed, {len(failed)} failed')
    return 1 if failed else 0
