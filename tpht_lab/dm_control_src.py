"""The dm_control cheetah, fetched reproducibly without installing dm_control.

The CT-RL tab runs dm_control's cheetah. Installing dm_control pulls in an
OpenGL stack and ~90 MB of packages for five small text files, so this module
downloads exactly those files from the dm_control repository at tag 1.0.51 and
checks each against a pinned sha256. The cheetah.xml hash is the one
physical_fj.mechanics.NominalRobot records, so the two can be compared.

dm_control is Apache-2.0; its LICENSE comes along and ctrl-model publishes it
next to the derived XML.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

VERSION = '1.0.51'
BASE_URL = f'https://raw.githubusercontent.com/google-deepmind/dm_control/{VERSION}/'

# path in the repository -> sha256
FILES = {
    'LICENSE': '58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd',
    'dm_control/suite/cheetah.xml': '47e3152e1213d6895ef36b36bbd399f1c7e0edf7d39f05675cd72bc5fbc14faa',
    'dm_control/suite/common/materials.xml': '83ee4f7d3f0c9b3003ad121c2c35169a32a922c437321278f27db60d577ccfac',
    'dm_control/suite/common/skybox.xml': '71db77c7c88dd4e1b54eb394deecfba673bbf539d110de68aa229dee00abf107',
    'dm_control/suite/common/visual.xml': '1e338ca511de35b902c4ca944d857c23fe4e7a5aea46a475cdc317dd454245b0',
}


@dataclass(frozen=True)
class Tree:
    root: Path

    @property
    def cheetah_xml(self) -> Path:
        return self.root / 'dm_control' / 'suite' / 'cheetah.xml'

    @property
    def licence(self) -> Path:
        return self.root / 'LICENSE'


def default_cache() -> Path:
    env = os.environ.get('TPHT_LAB_CACHE')
    return Path(env) if env else Path.home() / '.cache' / 'tpht-lab'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(cache: str | os.PathLike | None = None) -> Tree:
    """Download (or reuse) the pinned files; returns their directory tree.

    A cached file whose hash does not match is downloaded again; a download
    whose hash does not match is an error and is not kept.
    """
    root = Path(cache) if cache else default_cache()
    root = root / f'dm_control-{VERSION}'
    for rel, want in FILES.items():
        dst = root / rel
        if dst.is_file() and _sha256(dst) == want:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + '.part')
        url = BASE_URL + rel
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed https URL
            tmp.write_bytes(r.read())
        got = _sha256(tmp)
        if got != want:
            tmp.unlink()
            raise SystemExit(f'{url}: sha256 {got}, expected {want}')
        tmp.replace(dst)
    return Tree(root)
