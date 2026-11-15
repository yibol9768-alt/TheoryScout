"""Verify shipped bytes without reading any path outside this package."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
manifest=json.loads((BASE/'manifest.json').read_text())
for name,digest in manifest['sha256'].items():
    p=BASE/name
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    assert not p.is_symlink() and p.resolve().is_relative_to(BASE)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(status='PASS',verified_files=len(manifest['sha256']),external_project_files_used=0)))
