import hashlib, json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'HOLDOUT_D_runner_v5/regression_results'
RUNNER=ROOT/'HOLDOUT_D_runner_v5'
def digest(raw): return hashlib.sha256(raw).hexdigest()
def save(name, value):
    with (OUT/name).open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def resources():
    rel=Path('/proc/self/cgroup').read_text().strip().split('0::')[-1]
    cg=Path('/sys/fs/cgroup')/rel.lstrip('/')
    z=dict(cgroup=rel,memory_max=(cg/'memory.max').read_text().strip(),swap_max=(cg/'memory.swap.max').read_text().strip(),
           threads=len(list(Path('/proc/self/task').iterdir())),env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    assert z['memory_max']=='2147483648' and z['swap_max']=='0' and z['threads']<=2
    assert all(v in ('1','2') for v in z['env'].values())
    return z
def guards(loopback=False):
    def inside(p):
        if isinstance(p,int):return True
        return Path(os.fsdecode(p)).resolve().is_relative_to(OUT)
    def audit(event,args):
        if event=='open':
            p,mode,flags=args
            if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND) and not inside(p):
                raise RuntimeError('AUDIT_WRITE_BLOCKED:'+str(p))
        if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.chmod','os.truncate','os.symlink','os.link'):
            paths=args[:2] if event in ('os.rename','os.symlink','os.link') else args[:1]
            if any(not inside(p) for p in paths):raise RuntimeError('AUDIT_MUTATION_BLOCKED:'+repr(paths))
        if event.startswith('socket.') and event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            address=args[1] if event=='socket.connect' else (args[:2] if event=='socket.getaddrinfo' else args[-1])
            if not (loopback and isinstance(address,tuple) and address[0]=='127.0.0.1' and address[1]!=18319):
                raise RuntimeError('AUDIT_NETWORK_BLOCKED:'+repr(address))
    sys.addaudithook(audit)
def modules():
    # Only remap writable output and reconciliation evidence directory, in memory.
    import types
    sys.path.insert(0,str(RUNNER))
    source=(RUNNER/'engine.py').read_text().replace("ROOT/"+repr(RUNNER.name),'Path('+repr(str(OUT))+')')
    m=types.ModuleType('engine');m.__file__=str(RUNNER/'engine.py');sys.modules['engine']=m
    exec(compile(source,m.__file__,'exec'),m.__dict__)
    return m
