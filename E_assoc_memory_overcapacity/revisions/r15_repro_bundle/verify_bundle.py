"""Verify the complete immutable bundle, including toolchain and cached TeX."""
from pathlib import Path
import hashlib
import json


def verify(root):
    root=root.resolve();manifest=json.loads((root/'MANIFEST.json').read_text())
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    assert actual==set(manifest['files']), 'Missing or unexpected bundle files'
    for name,digest in manifest['files'].items():
        path=root/name
        assert not path.is_symlink() and path.resolve().is_relative_to(root)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    return manifest


if __name__=='__main__':
    manifest=verify(Path(__file__).resolve().parent)
    print('PASS',len(manifest['files']),'files; all SHA-256 hashes match')
