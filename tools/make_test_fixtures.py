#!/usr/bin/env python3
"""Generate fixtures for tests/*.test.js (run in the browser by tests/index.html).

bounded_ls below is copied verbatim from mujoco_reference/common.py (SciPy bvls).
references/desired are line-for-line NumPy transcriptions of
pure_cpht Controller.references and NativeController.desired.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear

OUT = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures'
rng = np.random.default_rng(7)


def bounded_ls(jacobian, offset, desired, ridge, low=-1., high=1.):
    weights = np.array([1., 4., 4., .15, .15, .15, .15, .15, .15])
    root = np.sqrt(weights)
    matrix = np.vstack([root[:, None]*jacobian, np.sqrt(ridge)*np.eye(6)])
    rhs = np.r_[root*(desired-offset), np.zeros(6)]
    result = lsq_linear(matrix, rhs, bounds=(np.broadcast_to(low, (6,)), np.broadcast_to(high, (6,))),
                        method='bvls', tol=1e-10, max_iter=100)
    assert result.success, result.message
    return result.x, float(np.linalg.norm(root*(jacobian@result.x+offset-desired)))


def references(p, phase, limits):
    omega = 2*np.pi*p[18]
    wave = p[6:12]*np.sin(phase)+p[12:18]*np.cos(phase)
    q = p[:6]+wave
    v = omega*(p[6:12]*np.cos(phase)-p[12:18]*np.sin(phase))
    a = -omega**2*wave
    limited = (q > limits[:, 1]-.02) | (q < limits[:, 0]+.02)
    q = np.clip(q, limits[:, 0]+.02, limits[:, 1]-.02)
    return q, np.where(limited, 0., v), np.where(limited, 0., a)


def desired(qpos, qvel, root, linvel_x, scan, p, phase, target, o, limits):
    qr, vr, ar = references(p, phase, limits)
    height = np.interp(root[0], scan[:, 0], scan[:, 1])
    return np.r_[o['speed_kd']*(target-linvel_x),
                 o['height_kp']*(o['target_clearance']-(root[2]-height))-o['height_kd']*qvel[1],
                 -o['pitch_kp']*qpos[2]-o['pitch_kd']*qvel[2],
                 ar+o['joint_kp']*(qr-qpos[3:])+o['joint_kd']*(vr-qvel[3:])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for k in range(400):
        # Shape the Jacobian like the cheetah's: singular values spanning ~10..600.
        u, _ = np.linalg.qr(rng.normal(size=(9, 9)))
        v, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        s = np.exp(rng.uniform(np.log(10), np.log(600), 6))
        J = u[:, :6]@np.diag(s)@v.T
        offset = rng.normal(scale=40, size=9)
        target = rng.normal(scale=40, size=9)
        if k % 2:  # finite-difference trust box around a center
            center = rng.uniform(-1, 1, 6)
            low, high = np.maximum(-1, center-.5), np.minimum(1, center+.5)
        else:
            low, high = -np.ones(6), np.ones(6)
        x, res = bounded_ls(J, offset, target, .01, low, high)
        cases.append(dict(J=J.ravel().tolist(), offset=offset.tolist(), desired=target.tolist(),
                          low=low.tolist(), high=high.tolist(), x=x.tolist(), residual=res))
    (OUT / 'bvls_cases.json').write_text(json.dumps(cases))

    options = dict(joint_kp=60., joint_kd=12., height_kp=60., height_kd=12., pitch_kp=50., pitch_kd=10.,
                   speed_kd=3., target_clearance=.55)
    limits = np.array([[-.52, 1.05], [-.785, .785], [-.4, .785], [-1., .7], [-1.2, .87], [-.5, .5]])
    gait_cases = []
    for k in range(200):
        p = np.r_[rng.uniform(-.4, .4, 6), rng.uniform(-.5, .5, 12), rng.uniform(1, 4)]
        phase = rng.uniform(0, 2*np.pi)
        qpos, qvel = rng.normal(scale=.4, size=9), rng.normal(scale=2, size=9)
        root = np.r_[rng.uniform(0, 10), 0., rng.uniform(.3, .8)]
        xs = root[0]+np.array([-1, -.5, 0, .5, 1, 2, 3])
        scan = np.c_[xs, rng.normal(scale=.05, size=7), np.zeros(7)]
        lin = rng.normal(scale=2)
        qr, vr, ar = references(p, phase, limits)
        d = desired(qpos, qvel, root, lin, scan, p, phase, 3., options, limits)
        gait_cases.append(dict(p=p.tolist(), phase=phase, qpos=qpos.tolist(), qvel=qvel.tolist(), root=root.tolist(),
                               linvel_x=lin, scan=scan.tolist(), qr=qr.tolist(), vr=vr.tolist(), ar=ar.tolist(),
                               desired=d.tolist()))
    (OUT / 'gait_cases.json').write_text(json.dumps(dict(options=options, limits=limits.tolist(), cases=gait_cases)))
    print(f'{len(cases)} bvls cases, {len(gait_cases)} gait cases -> {OUT}')


if __name__ == '__main__':
    main()
