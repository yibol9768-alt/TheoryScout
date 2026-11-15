"""Continue interrupted experiments in one constrained scope."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

root = Path.cwd()
base = root/'supplement'
dest = base/'full_experiments_continuation'
dest.mkdir(exist_ok=False)
initial = json.loads((root/'PKG_original_sources.json').read_text())
(root/'PKG_resume_source_baseline.json').write_text(json.dumps({name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in initial}, indent=2))
cg = next(row.split(':', 2)[2] for row in Path('/proc/self/cgroup').read_text().splitlines() if row.startswith('0::'))
cgroup = Path('/sys/fs/cgroup')/cg.lstrip('/')
running = []
records = []
for index, design in enumerate(['matched', 'confirmation'], 1):
    job = ['experiments_replay/resume_retrieval.py', '--design', design,
        '--existing', 'full_experiments_recomputation/'+design,
        '--output', 'full_experiments_continuation/'+design]
    log = (dest/f'command_{index}.log').open('x')
    start = time.monotonic()
    proc = subprocess.Popen([sys.executable, '-B', '-u', *job], cwd=base, stdout=log, stderr=subprocess.STDOUT)
    running.append((proc, log, job, start))
maxthreads = 0
while running:
    limits = {k: (cgroup/k).read_text().strip() for k in ['memory.max', 'memory.swap.max', 'memory.peak', 'memory.events']}
    assert limits['memory.max'] == '3221225472' and limits['memory.swap.max'] == '0'
    for pid in (cgroup/'cgroup.procs').read_text().split():
        try:
            maxthreads = max(maxthreads, len(list((Path('/proc')/pid/'task').iterdir())))
        except FileNotFoundError:
            pass
    assert maxthreads <= 2
    (root/'PKG_continuation_resources.json').write_text(json.dumps(dict(resource_limits=limits, max_threads_per_process=maxthreads), indent=2))
    for item in list(running):
        proc, log, job, start = item
        status = proc.poll()
        if status is None:
            continue
        log.close()
        running.remove(item)
        record = dict(command='python '+' '.join(job), seconds=time.monotonic()-start, exit_code=status)
        records.append(record)
        (dest/'timings.json').write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
        if status:
            for other, handle, _, _ in running:
                other.terminate()
                other.wait()
                handle.close()
            raise SystemExit(status)
    if running:
        time.sleep(2)
print('PASS: both interrupted designs completed', flush=True)
