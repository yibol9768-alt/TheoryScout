"""Finish checks and packaging after both resumed workers complete."""
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd()
folder = root/'supplement/full_experiments_continuation'
while not all((folder/design/'result.json').exists() for design in ['matched', 'confirmation']):
    active = subprocess.run(['systemctl', '--user', 'is-active', 'pkg-resume-full.scope'], capture_output=True, text=True)
    assert active.stdout.strip() == 'active', 'Continuation scope stopped before completion; retain and inspect logs.'
    time.sleep(5)
for script, name in [('PKG_validate_continuation.py', 'PKG_continuation_validation.log'), ('PKG_release_resume.py', 'PKG_release.log')]:
    with (root/name).open('x') as handle:
        proc = subprocess.run([sys.executable, '-B', '-u', script], cwd=root, stdout=handle, stderr=subprocess.STDOUT)
    assert proc.returncode == 0, (root/name).read_text()[-5000:]
    print('PASS: '+script, flush=True)
print('PASS: final report and verified submission archive are ready', flush=True)
