#!/usr/bin/env python3
"""Assemble the static site from this repository and cvcpkg prefixes.

No bundler, no node and no npm: the app is plain ES modules that the browser
loads as they are, and its two third-party runtimes come from cvcpkg packages:

  mujoco           (wasm)   share/mujoco/wasm/mujoco.{js,wasm}       -> vendor/mujoco/
  onnxruntime-web  (noarch) share/onnxruntime-web/ort.wasm.min.mjs  -> vendor/ort/
                            + ort-wasm-simd-threaded.{mjs,wasm}

Each may sit in any of the --prefix directories (the cvcpkg recipe passes its
deps prefix and its build prefix).

Layout of OUT (what GitHub Pages or `tpht-lab serve` publishes):

  index.html, src/          the app, unchanged (paths are relative to src/base.js)
  assets/, explain/         copied from public/, as Vite used to
  vendor/mujoco/, vendor/ort/   the runtimes, with their licence texts
  tests/                    only with --with-tests (the in-browser test page)
  build-info.json           versions and the git commit

Usage:
  python tools/build_site.py --prefix <cvcpkg prefix> [--prefix ...] --out <dir> [--with-tests]

The cvcpkg recipe cvcpkg/recipes/tpht-viz-panel runs this; see the README.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MUJOCO_FILES = ('mujoco.js', 'mujoco.wasm')
ORT_FILES = ('ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm')


def fail(msg: str) -> None:
    sys.exit(f'build_site: {msg}')


def copy_tree(src: Path, dst: Path, ignore=None) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def read_version_cmake(path: Path) -> str | None:
    if not path.is_file():
        return None
    m = re.search(r'set\(PACKAGE_VERSION\s+"([^"]+)"\)', path.read_text(encoding='utf-8'))
    return m.group(1) if m else None


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def find_prefix(prefixes: list[Path], rel: str, files: tuple[str, ...]) -> Path | None:
    for prefix in prefixes:
        if all((prefix / rel / f).is_file() for f in files):
            return prefix
    return None


def vendor_mujoco(prefixes: list[Path], out: Path) -> str | None:
    prefix = find_prefix(prefixes, 'share/mujoco/wasm', MUJOCO_FILES)
    if prefix is None:
        fail(f'no prefix in {[str(p) for p in prefixes]} has share/mujoco/wasm/mujoco.{{js,wasm}}: '
             "install the wasm build of cvcpkg's mujoco "
             '(cvcpkg install mujoco --platform wasm --arch wasm32 --link static --prefix ...)')
    src = prefix / 'share' / 'mujoco' / 'wasm'
    dst = out / 'vendor' / 'mujoco'
    dst.mkdir(parents=True, exist_ok=True)
    for f in MUJOCO_FILES + ('mujoco.d.ts',):
        if (src / f).is_file():
            shutil.copy2(src / f, dst / f)
    lic = prefix / 'share' / 'licenses' / 'mujoco'
    if lic.is_dir():
        copy_tree(lic, dst / 'LICENSES')
    return read_version_cmake(prefix / 'lib' / 'cmake' / 'mujoco' / 'mujocoConfigVersion.cmake')


def vendor_ort(prefixes: list[Path], out: Path) -> str | None:
    prefix = find_prefix(prefixes, 'share/onnxruntime-web', ORT_FILES)
    if prefix is None:
        fail(f'no prefix in {[str(p) for p in prefixes]} has share/onnxruntime-web/ort.wasm.min.mjs '
             'and its runtime: build the onnxruntime-web recipe into the prefix '
             '(cvcpkg build onnxruntime-web --platform any --local --recipes-dir cvcpkg/recipes --prefix ...)')
    src = prefix / 'share' / 'onnxruntime-web'
    dst = out / 'vendor' / 'ort'
    dst.mkdir(parents=True, exist_ok=True)
    for f in ORT_FILES + ('package.json',):
        if (src / f).is_file():
            shutil.copy2(src / f, dst / f)
    doc = prefix / 'share' / 'licenses' / 'onnxruntime-web'
    for f in ('LICENSE', 'ThirdPartyNotices.txt'):
        if (doc / f).is_file():
            shutil.copy2(doc / f, dst / f)
    pkg = src / 'package.json'
    return json.loads(pkg.read_text(encoding='utf-8')).get('version') if pkg.is_file() else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prefix', required=True, type=Path, action='append',
                   help='cvcpkg prefix holding mujoco (wasm) and/or onnxruntime-web; repeatable')
    p.add_argument('--out', required=True, type=Path, help='output directory (replaced)')
    p.add_argument('--src', type=Path, default=REPO, help='repository root (default: this checkout)')
    p.add_argument('--with-tests', action='store_true', help='also copy tests/ for the in-browser test page')
    args = p.parse_args(argv)

    src, out = args.src.resolve(), args.out.resolve()
    prefixes = [q.resolve() for q in args.prefix]
    for need in ('index.html', 'src', 'public'):
        if not (src / need).exists():
            fail(f'{src / need} not found: --src must be the repository root')
    if out == src or out in src.parents or (src in out.parents and out.parent == src):
        fail(f'refusing to replace {out}: pick an output directory outside the source tree '
             '(or a subdirectory such as build/site)')
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copy2(src / 'index.html', out / 'index.html')
    copy_tree(src / 'src', out / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    copy_tree(src / 'public', out)                   # public/assets -> assets, public/explain -> explain
    if args.with_tests:
        copy_tree(src / 'tests', out / 'tests', ignore=shutil.ignore_patterns('__pycache__', '*.py'))

    info = dict(
        mujoco=vendor_mujoco(prefixes, out),
        onnxruntime_web=vendor_ort(prefixes, out),
        commit=git_commit(src),
        tests=args.with_tests,
    )
    (out / 'build-info.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')

    assets = out / 'assets'
    published = sorted(str(d.relative_to(out)).replace('\\', '/')
                       for d in [*assets.glob('*'), *assets.glob('*/*')] if d.is_dir())
    size = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    print(f'build_site: {out} ({size / 1e6:.1f} MB), MuJoCo {info["mujoco"]}, '
          f'onnxruntime-web {info["onnxruntime_web"]}')
    print('build_site: published asset folders: ' + (', '.join(published) or '(none)'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
