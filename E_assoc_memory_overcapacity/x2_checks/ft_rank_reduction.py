"""Exact scalar reduction for old-task FT1, independent tasks, n=d^2.
Only continuum recall replaces the random sphere order statistic by z_rho.
"""
import os
assert all(os.environ[s]=='1' for s in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'])
import json,time,math,sys,resource
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri
from scipy.stats import qmc,chi2
from checks import ft_independent
H=Path(os.environ.get('X2_OUTPUT',str(Path(__file__).resolve().parent)))
start=time.monotonic()

def reduced(d,seed,gh=48):
    n=d*d; alpha=1-math.sqrt(.75)
    un=qmc.Sobol(8,scramble=True,seed=seed).random_base2(16)
    Ax,A,B,C,Ay,Cy,Cn=[chi2.ppf(un[:,j],df) for j,df in enumerate([d,n,d-1,n-1,d,d-1,n-2])]
    z2=ndtri(un[:,7]); Ax/=d; A/=n; Ay/=d
    t=Ax*(1-alpha*A)
    hh=Ax*((1-alpha*A)**2+alpha*alpha*A*B/n)
    HH=d/n*t*t+hh*C/n
    L=HH+alpha*alpha*Ax*A
    a0=t; ss=np.sqrt(L-a0*a0/d); sqA=np.sqrt(Ay)
    corr=np.sqrt(np.maximum(HH-a0*a0/d,0))/ss
    inn=np.sqrt(np.maximum(1-corr*corr,0))*z2
    def state(z):
        zz=sqA*z; bb=(z*z+Cy)/d
        D=a0*Ay+ss*zz
        vn=(L-a0*a0/d)*bb+a0*a0/d*Ay+2*a0*ss*zz/d
        vp=(L-a0*a0/d)*2*z/d+2*a0*ss*sqA/d
        ym=a0*zz/math.sqrt(d)+math.sqrt(d)*ss*bb
        yp=a0*sqA/math.sqrt(d)+math.sqrt(d)*ss*2*z/d
        yv=corr*ym+inn*np.sqrt(vn)
        yp=corr*yp+inn*vp/(2*np.sqrt(vn))
        rr=(yv*yv+vn*Cn)/(n-1)
        rrp=(2*yv*yp+vp*Cn)/(n-1)
        return D,np.sqrt(rr),ss*sqA,rrp
    zz,ww=np.polynomial.hermite.hermgauss(gh)
    rank=np.zeros(len(Ax))
    for z,w in zip(zz*math.sqrt(2),ww/math.sqrt(math.pi)):
        D,R,_,_=state(z); s=D/R
        rank+=w*(ndtr(-s)-(s**3-3*s)*np.exp(-s*s/2)/math.sqrt(2*math.pi)/(4*(n-1)))
    mu=ft_independent(1,alpha)['mu']; zr=ndtri(.9)
    root=np.full(len(Ax),zr-mu)
    for _ in range(8):
        D,R,dp,rp=state(root)
        root-=(D-zr*R)/(dp-zr*rp/(2*R))
    return float(rank.mean()),float(ndtr(-root).mean())

def main():
    rows=[]; pred=ft_independent(1,1-math.sqrt(.75))
    for d in [64,128,256,512]:
        v=np.array([reduced(d,550100+j) for j in range(8)])
        m=v.mean(axis=0); se=v.std(axis=0,ddof=1)/math.sqrt(8)
        r=dict(d=d,rank=float(m[0]),rank_se=float(se[0]),recall_continuum=float(m[1]),recall_se=float(se[1]),
          rank_beta=float(d*(m[0]-pred['mean_rank_0'])),recall_beta=float(d*(m[1]-pred['recall_0'])))
        rows.append(r);print(json.dumps(r),flush=True)
    with (H/'ft_rank_reduction.json').open('x') as f:json.dump(dict(prediction=pred,rows=rows),f,indent=2)
    cg=Path('/sys/fs/cgroup')/next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')).lstrip('/')
    with (H/'ft_rank_metadata.json').open('x') as f:json.dump(dict(seconds=time.monotonic()-start,python=sys.executable,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
     controls={s:(cg/s).read_text().strip() for s in ['memory.max','memory.swap.max','memory.peak']},process_threads=len(list(Path('/proc/self/task').iterdir()))),f,indent=2)

if __name__=='__main__': main()
