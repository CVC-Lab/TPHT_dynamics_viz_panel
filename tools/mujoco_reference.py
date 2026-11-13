"""Compute the MuJoCo-version-dependent values in tests/fixtures/mujoco_reference.json.

Runs `import mujoco` (whatever version the interpreter has) on dm_control's cheetah
from public/assets/ctrl, independently of the C++ and JavaScript backends, and
prints the values for that version. With --write it stores them in the fixture
under mj_versionString(), next to the other versions' values.

    <prefix>/python tools/mujoco_reference.py [--write]

demo_dx_5s: the demo sine gait (cpp/src/backend.cpp SineGait, src/controllers/
demo_gait.js) for 500 steps from a forwarded mj_resetData, stepping as the page
does: float32 controls, then mj_step2 and mj_step1 (the model's legacy split
step). The value is how far qpos[0] moves.
"""
import argparse
import json
import math
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = ROOT / 'tests' / 'fixtures' / 'mujoco_reference.json'

# SineGait(nu, frequency 2.5 Hz, amplitude 0.7, joint lag 0.6 rad, front-leg offset pi/2).
FREQUENCY, AMPLITUDE, JOINT_LAG, LEG_OFFSET = 2.5, 0.7, 0.6, math.pi / 2


def demo_dx_5s() -> float:
    m = mujoco.MjModel.from_xml_path(str(ROOT / 'public' / 'assets' / 'ctrl' / 'cheetah.xml'))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    phase = np.array([(i % 3) * JOINT_LAG + (LEG_OFFSET if i >= 3 else 0.0) for i in range(m.nu)])
    x0 = float(d.qpos[0])
    for _ in range(500):
        w = 2 * math.pi * FREQUENCY * d.time
        d.ctrl[:] = (AMPLITUDE * np.sin(w + phase)).astype(np.float32)
        mujoco.mj_step2(m, d)
        mujoco.mj_step1(m, d)
    assert abs(d.time - 5.0) < 1e-9, d.time
    return float(d.qpos[0]) - x0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--write', action='store_true', help=f'store the values in {FIXTURE.relative_to(ROOT)}')
    args = ap.parse_args()
    version = mujoco.mj_versionString()
    values = {'demo_dx_5s': demo_dx_5s()}
    for k, v in values.items():
        print(f'MuJoCo {version}: {k} = {v!r}')
    if args.write:
        ref = json.loads(FIXTURE.read_text(encoding='utf-8'))
        for k, v in values.items():
            ref.setdefault(k, {})[version] = v
            ref[k] = dict(sorted(ref[k].items(), key=lambda kv: [int(p) for p in kv[0].split('.')]))
        FIXTURE.write_text(json.dumps(ref, indent=2) + '\n', encoding='utf-8')
        print(f'wrote {FIXTURE.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
