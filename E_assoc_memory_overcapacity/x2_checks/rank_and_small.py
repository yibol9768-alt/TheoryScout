import os
for s in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ[s]=='1'
import json,time,math,resource,sys
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri
from scipy.stats import qmc,chi2
from checks import gd,radial_moments
from wick import moment,evaluate

HERE=Path(os.environ.get('X2_OUTPUT',str(Path(__file__).resolve().parent)))
start=time.monotonic()

def reduced(d,seed,power=15,gh=48):
    n=d*d; N=2**power
    uu=qmc.Sobol(5,scramble=True,seed=seed).random_base2(power)
    Ax,Ck,Ay,Cy,Cn=[chi2.ppf(uu[:,j],df) for j,df in enumerate([d,n-1,d,d-1,n-2])]
    Ax/=d; Ay/=d
    L=Ax*(Ck/n+d/n*Ax); a0=Ax
    ss=np.sqrt(L-a0*a0/d); sqA=np.sqrt(Ay)
    def state(z):
        H=sqA*z; B=(z*z+Cy)/d
        D=a0*Ay+ss*H
        vn=(L-a0*a0/d)*B+a0*a0/d*Ay+2*a0*ss*H/d
        yv=a0*H/math.sqrt(d)+np.sqrt(d)*ss*B
        R2=(yv*yv+vn*Cn)/(n-1)
        dp=ss*sqA
        vp=(L-a0*a0/d)*2*z/d+2*a0*ss*sqA/d
        yp=a0*sqA/math.sqrt(d)+math.sqrt(d)*ss*2*z/d
        r2p=(2*yv*yp+vp*Cn)/(n-1)
        return D,np.sqrt(R2),dp,r2p
    zz,ww=np.polynomial.hermite.hermgauss(gh)
    ranks=np.zeros(N)
    for z,w in zip(zz*math.sqrt(2),ww/math.sqrt(math.pi)):
        D,R,_,_=state(z); s=D/R
        # Exact sphere marginal through order 1/(n-1), remainder O(n^-2).
        tail=ndtr(-s)-(s**3-3*s)*np.exp(-s*s/2)/math.sqrt(2*math.pi)/(4*(n-1))
        ranks+=w*tail
    zrho=ndtri(.9)
    root=np.full(N,zrho-1.)
    for _ in range(7):
        D,R,dp,r2p=state(root)
        root-=(D-zrho*R)/(dp-zrho*r2p/(2*R))
    rec=ndtr(-root)
    # Root corresponds to continuum normal candidate threshold; finite rank
    # rounding/order-statistic displacement is O(1/n), not included here.
    return float(ranks.mean()),float(rec.mean())

def small():
    d=3; n=9; reps=160000; alpha=.14; rx=.6; ry=.7
    rng=np.random.default_rng(922044)
    vals={s:[] for s in ['ft_mean','ft_second','ft_wrong','adam_mean','adam_var','adam_wrong']}
    for _ in range(reps//1000):
        x0,y0,xn,yn=[rng.standard_normal((1000,d,n))/math.sqrt(d) for j in range(4)]
        x1=rx*x0+math.sqrt(1-rx*rx)*xn
        y1=ry*y0+math.sqrt(1-ry*ry)*yn
        tr=lambda a:a.swapaxes(-1,-2)
        z0=d/math.sqrt(n)*(y0@tr(x0)); z1=d/math.sqrt(n)*(y1@tr(x1))
        s1=d/n*(x1@tr(x1)); u=z0-alpha*(z0@s1-z1)
        scores=tr(y0)@u@x0
        vals['ft_mean'].extend(scores[:,0,0]); vals['ft_second'].extend(scores[:,0,0]**2)
        vals['ft_wrong'].extend(scores[:,1,0]**2)
        ua=z0/(abs(z0)+.1)
        da=np.diagonal(tr(y0)@ua@x0,axis1=1,axis2=2)
        vals['adam_mean'].extend(da.mean(axis=1))
        vals['adam_var'].extend(da.var(axis=1,ddof=1))
        vals['adam_wrong'].extend((tr(y0)@ua@x0)[:,1,0]**2)
    preds={s:evaluate(moment(0,o,w),d,n,alpha,rx,ry,o)
           for s,o,w in [('ft_mean',1,False),('ft_second',2,False),('ft_wrong',2,True)]}
    aa=radial_moments(d,.1,80)
    preds.update(adam_mean=aa['mean']*math.sqrt(aa['q']),adam_var=aa['sample_variance']*aa['q'],adam_wrong=aa['wrong_variance']*aa['q'])
    out={}
    for key,v in vals.items():
        v=np.array(v); se=float(v.std(ddof=1)/math.sqrt(len(v)))
        out[key]=dict(mean=float(v.mean()),se=se,prediction=preds[key],z=(float(v.mean())-preds[key])/se)
    return out

def main():
    rows=[]; pred=gd(1,1.)
    for d in [64,128,256,512]:
        v=np.array([reduced(d,221000+j) for j in range(8)])
        m=v.mean(axis=0); se=v.std(axis=0,ddof=1)/math.sqrt(len(v))
        row=dict(d=d,reps=8,points_per_rep=2**15,mean_rank=float(m[0]),rank_se=float(se[0]),
          recall_continuum=float(m[1]),recall_se=float(se[1]),
          rank_beta=float(d*(m[0]-pred['mean_rank_0'])),
          recall_beta=float(d*(m[1]-pred['recall_0'])))
        rows.append(row); print(json.dumps(row),flush=True)
    with (HERE/'rank_reduction.json').open('x') as f: json.dump(rows,f,indent=2)
    with (HERE/'small_checks.json').open('x') as f: json.dump(small(),f,indent=2)
    cg=Path('/sys/fs/cgroup')/next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')).lstrip('/')
    with (HERE/'rank_metadata.json').open('x') as f:
        json.dump(dict(seconds=time.monotonic()-start,python=sys.executable,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          controls={s:(cg/s).read_text().strip() for s in ['memory.max','memory.swap.max','memory.peak']},
          process_threads=len(list(Path('/proc/self/task').iterdir()))),f,indent=2)

if __name__=='__main__': main()
