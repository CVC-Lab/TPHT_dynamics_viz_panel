#!/usr/bin/env python3
"""Compatibility shim: the exporter now lives in tpht_lab/export.py.

`python tools/export_web_assets.py <subcommand>` keeps working from a checkout
and writes into this checkout's public/assets by default; installed, the same
tool is `tpht-lab export <subcommand>`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpht_lab import export  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(export.main(default_out=ROOT / 'public' / 'assets'))
