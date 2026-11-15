import os
assert all(os.environ[s]=='1' for s in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'])
import sys,hashlib,json,time,math,ast,resource
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from checks import rank_coeff,ft1
from rank_and_small import reduced as gd_reduce
from ft_rank_reduction import reduced as ft_reduce
ROOT=Path(__file__).resolve().parent.parent
H=Path(os.environ.get('X2_OUTPUT',str(Path(__file__).resolve().parent)))
start=time.monotonic()
hashes=json.loads((H/'verified/input_hashes.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==s for p,s in hashes.items())
for p in Path(__file__).resolve().parent.glob('*.py'):ast.parse(p.read_text())
# Check closed rank/recall coefficients against the unsimplified Gaussian integral.
rng=np.random.default_rng(220923)
errs=[]
for j in range(20):
    mu,m,l=rng.normal(size=3);h=float(rng.uniform(.1,1.));chi=float(rng.uniform(0,.8));kappa=float(rng.uniform(.4,2.))
    r=rank_coeff(mu,m,l,h,mu*mu/2+chi,kappa)
    g=m-mu*l/2-mu/4;c=h/kappa;t=1.2815515655446004-mu
    bq=math.exp(-mu*mu/4)/math.sqrt(4*math.pi)*(-g+mu*(1+c+chi)/4)
    br=math.exp(-t*t/2)/math.sqrt(2*math.pi)*(g-mu/4-mu*c/2+t*(chi-c)/2)
    errs.append(max(abs(bq-r['mean_rank_1']),abs(br-r['recall_1'])))
assert max(errs)<1e-11
for kap in [.5,1.,2.]:
 for k in [0,1]:
  for x,y in [(0.,0.),(1.,.5),(.6,.7)]:ft1(k,.14,x,y,kap)
gh=[]
for d in [64,512]:
    a=gd_reduce(d,221000,gh=48);b=gd_reduce(d,221000,gh=80)
    c=ft_reduce(d,550100,gh=48);e=ft_reduce(d,550100,gh=80)
    gh.append(dict(d=d,GD48=a,GD80=b,FT48=c,FT80=e,GD_difference=max(abs(np.array(a)-b)),FT_difference=max(abs(np.array(c)-e))))
assert max(r['GD_difference'] for r in gh)<1e-10
assert max(r['FT_difference'] for r in gh)<1e-10
cg=Path('/sys/fs/cgroup')/next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')).lstrip('/')
out=dict(seconds=time.monotonic()-start,closed_formula_max_error=max(errs),input_hashes_unchanged=True,
  hermite_doubling=gh,python=sys.executable,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
  controls={s:(cg/s).read_text().strip() for s in ['memory.max','memory.swap.max','memory.peak']},process_threads=len(list(Path('/proc/self/task').iterdir())))
with (H/'audit.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
