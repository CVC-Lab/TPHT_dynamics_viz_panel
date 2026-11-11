"""tpht-lab: the native host for the TPHT dynamics viz panel.

  tpht-lab serve    serve the built site (correct .mjs/.wasm types; --isolate adds COOP/COEP)
  tpht-lab check    native MuJoCo checks on the site's assets (the page's self-tests, natively)
  tpht-lab test     run tests/index.html in a headless browser and report the result
  tpht-lab export   asset exporter (traces, samples, results, ref-model, gait, ctrl-model, ctrl-policy)

The site comes from the cvcpkg recipe tpht-viz-panel (<prefix>/share/tpht-viz-panel)
or from `python tools/build_site.py --out <dir>`; pass --dir to choose.
"""
from __future__ import annotations

import argparse
import glob
import os
import queue
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import webbrowser
from pathlib import Path

from . import __version__


def installed_site() -> Path | None:
    """<prefix>/share/tpht-viz-panel for the interpreter's prefix (cvcpkg installs the site there)."""
    for base in {sys.prefix, sysconfig.get_config_var('prefix') or sys.prefix}:
        p = Path(base) / 'share' / 'tpht-viz-panel'
        if (p / 'index.html').is_file():
            return p
    return None


def site_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    for p in (installed_site(), Path('build') / 'site'):
        if p is not None and (p / 'index.html').is_file():
            return p
    raise SystemExit('No built site found: pass --dir, build one with `python tools/build_site.py --prefix <prefix> '
                     '--out build/site`, or install the tpht-viz-panel cvcpkg package into this prefix.')


def cmd_serve(args) -> int:
    from .serve import make_server
    directory = site_dir(args.dir)
    server = make_server(directory, host=args.host, port=args.port, isolate=args.isolate)
    host, port = server.server_address[:2]
    url = f'http://{host}:{port}/'
    print(f'tpht-lab: serving {directory.resolve()} at {url}' + (' (cross-origin isolated)' if args.isolate else ''))
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_check(args) -> int:
    from . import check
    if args.assets:
        assets, site = Path(args.assets), None
    else:
        site = site_dir(args.dir) if (args.dir or installed_site() or (Path('build/site') / 'index.html').is_file()) else None
        assets = site / 'assets' if site else Path('public') / 'assets'
    print(f'tpht-lab: checking {assets.resolve()}')
    return check.report(check.run(assets, seconds=args.seconds, site=site), as_json=args.json)


def find_browser(arg: str | None) -> str:
    """A headless-capable Chromium: --browser, $TPHT_BROWSER, cvcpkg's playwright-chromium-headless-shell
    in this prefix (linux x86_64), then Chrome/Edge/Chromium on PATH."""
    if arg:
        return arg
    if os.environ.get('TPHT_BROWSER'):
        return os.environ['TPHT_BROWSER']
    pattern = 'share/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell'
    for base in (sys.prefix, os.environ.get('CVCPKG_PREFIX', '')):
        hits = sorted(glob.glob(str(Path(base) / pattern))) if base else []
        if hits:
            return hits[-1]
    for name in ('chrome-headless-shell', 'chromium', 'chromium-browser', 'google-chrome', 'chrome', 'msedge'):
        if found := shutil.which(name):
            return found
    raise SystemExit('No headless browser found. On linux x86_64: cvcpkg install playwright-chromium-headless-shell '
                     '--prefix <this prefix>; elsewhere pass --browser /path/to/chrome (or set TPHT_BROWSER).')


def cmd_test(args) -> int:
    from .serve import serve_in_thread
    directory = site_dir(args.dir)
    if not (directory / 'tests' / 'index.html').is_file():
        raise SystemExit(f'{directory} has no tests/: build it with `python tools/build_site.py ... --with-tests`')
    browser = find_browser(args.browser)
    # The page POSTs its results to /__results (tests/index.html?report=1), so this works the
    # same with any Chromium-family browser; --dump-dom prints nothing from Edge on Windows.
    sink: queue.Queue = queue.Queue()
    server, base = serve_in_thread(directory, results_sink=sink)
    # A throwaway profile, so a desktop Chrome/Edge that is already running does not take the
    # command over.
    with tempfile.TemporaryDirectory(prefix='tpht-lab-browser-', ignore_cleanup_errors=True) as profile:
        cmd = [browser, '--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
               f'--user-data-dir={profile}', base + 'tests/index.html?report=1']
        if hasattr(os, 'geteuid') and os.geteuid() == 0:
            cmd.insert(1, '--no-sandbox')  # Chromium refuses to sandbox as root (CI containers)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            report = sink.get(timeout=args.timeout)
        except queue.Empty:
            report = None
        finally:
            proc.kill()
            stderr = proc.communicate()[1] or ''
            server.shutdown()
    if report is None:
        sys.stderr.write(stderr[-4000:])
        print(f'tpht-lab test: no results from the page within {args.timeout:g} s ({browser})')
        return 1
    for r in report.get('results', []):
        detail = f' - {r["detail"]}' if r.get('detail') else ''
        print(f'{r["status"].upper():4}  {r["name"]}  ({r["ms"]} ms){detail}')
    print(f'tpht-lab test: {report.get("summary")} ({browser})')
    return 1 if report.get('failed') else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['export']:  # hand the rest to the exporter's own parser
        from . import export
        return export.main(argv[1:], prog='tpht-lab export') or 0
    p = argparse.ArgumentParser(prog='tpht-lab', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', action='version', version=f'tpht-lab {__version__}')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('serve', help='serve the built site')
    s.add_argument('--dir', help='site directory (default: <prefix>/share/tpht-viz-panel, then build/site)')
    s.add_argument('--host', default='127.0.0.1')
    s.add_argument('--port', type=int, default=8000)
    s.add_argument('--isolate', action='store_true', help='send COOP/COEP (cross-origin isolation)')
    s.add_argument('--open', action='store_true', help='open the page in the default browser')

    s = sub.add_parser('check', help='native MuJoCo checks on the site assets')
    s.add_argument('--dir', help='site directory; checks its assets/ and compares MuJoCo versions')
    s.add_argument('--assets', help='an assets directory instead (e.g. public/assets)')
    s.add_argument('--seconds', type=float, default=5.0, help='demo-gait rollout length')
    s.add_argument('--json', action='store_true')

    s = sub.add_parser('test', help='run the in-browser test page headless')
    s.add_argument('--dir', help='site directory built with --with-tests')
    s.add_argument('--browser', help='Chromium-family executable')
    s.add_argument('--timeout', type=float, default=180.0, help='wall-clock limit in seconds')

    sub.add_parser('export', help='asset exporter; `tpht-lab export --help` for its subcommands')

    args = p.parse_args(argv)
    return {'serve': cmd_serve, 'check': cmd_check, 'test': cmd_test}[args.cmd](args)


if __name__ == '__main__':
    raise SystemExit(main())
