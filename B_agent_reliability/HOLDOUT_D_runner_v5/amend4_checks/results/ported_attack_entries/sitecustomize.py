"""D4: deny every Internet/DNS send and all writes outside the new audit tree."""
import os, sys
from pathlib import Path
OUT=Path('/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/TheoryScout/B_agent_reliability/HOLDOUT_D_runner_v4/amend4_checks/results')
def inside(p):
    return isinstance(p,int) or Path(os.fsdecode(p)).resolve().is_relative_to(OUT)
def guard(event,args):
    if event=='open' and args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
        if not inside(args[0]):raise RuntimeError('D4_WRITE_BLOCKED:'+str(args[0]))
    if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.chmod','os.truncate','os.symlink','os.link'):
        paths=args[:2] if event in ('os.rename','os.symlink','os.link') else args[:1]
        if any(not inside(p) for p in paths):raise RuntimeError('D4_MUTATION_BLOCKED:'+repr(paths))
    if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo','socket.sendto','socket.bind'):
        raise RuntimeError('D4_NETWORK_BLOCKED:'+event)
sys.addaudithook(guard)
