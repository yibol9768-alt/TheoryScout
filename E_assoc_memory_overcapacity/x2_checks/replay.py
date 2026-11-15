"""Replay in a new directory inside x2_checks; refuses to overwrite results."""
import os,sys,runpy,time,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
assert all(os.environ[s]=='1' for s in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'])
out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE/'replay_output'
assert out.is_relative_to(HERE) and out!=HERE
out.mkdir(exist_ok=False);(out/'verified').mkdir()
sys.path.insert(0,str(HERE))
start=time.monotonic()
for name in ['checks','scatter_checks','rank_and_small','ft_rank_reduction','summarize','audit']:
    os.environ['X2_OUTPUT']=str(out/'verified' if name=='checks' else out)
    print('RUN',name,flush=True)
    runpy.run_path(str(HERE/(name+'.py')),run_name='__main__')
with (out/'replay_metadata.json').open('x') as f:json.dump(dict(total_seconds=time.monotonic()-start),f)
