"""D5: deny every Internet/DNS send and all writes outside the new audit tree."""
import os, sys
from pathlib import Path
OUT=Path('/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/TheoryScout/B_agent_reliability/HOLDOUT_D_runner_v5/amend5_checks/results')
def inside(p):
    return isinstance(p,int) or Path(os.fsdecode(p)).resolve().is_relative_to(OUT)
def guard(event,args):
    if event=='open' and args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
        if not inside(args[0]):raise RuntimeError('D5_WRITE_BLOCKED:'+str(args[0]))
    if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.chmod','os.truncate','os.symlink','os.link'):
        paths=args[:2] if event in ('os.rename','os.symlink','os.link') else args[:1]
        if any(not inside(p) for p in paths):raise RuntimeError('D5_MUTATION_BLOCKED:'+repr(paths))
    if event=='subprocess.Popen':
        executable,argv,cwd,env=args
        quota='/home/liuyibo/auto-llm-driving/TheoryScout/tools/plus_quota.py'
        if quota in str(argv):
            import shutil
            resolved=shutil.which(os.fsdecode(executable),path=(env or os.environ).get('PATH'))
            if resolved is None or Path(resolved).resolve()!=OUT/'command_bin/python3':
                raise RuntimeError('D5_REAL_QUOTA_EXECUTION_BLOCKED')
    if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo','socket.sendto','socket.bind'):
        raise RuntimeError('D5_NETWORK_BLOCKED:'+event)
sys.addaudithook(guard)
