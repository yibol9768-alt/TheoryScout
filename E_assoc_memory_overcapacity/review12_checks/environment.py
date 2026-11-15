import importlib.util
import json
import os
from pathlib import Path


def resources():
    rel = next(x.split(':', 2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    cg = Path('/sys/fs/cgroup') / rel.lstrip('/')
    info = {s: (cg / s).read_text().strip() for s in ('memory.max', 'memory.swap.max', 'memory.peak', 'memory.events')}
    info['threads'] = len(list(Path('/proc/self/task').iterdir()))
    assert info['memory.max'] == '3221225472'
    assert info['memory.swap.max'] == '0'
    assert info['threads'] <= 2
    return info


if __name__ == '__main__':
    info = {'modules': {m: bool(importlib.util.find_spec(m)) for m in ('mpmath', 'flint', 'numpy', 'scipy', 'sympy')}, 'resources': resources()}
    with open('review12_checks/environment.json', 'x') as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
