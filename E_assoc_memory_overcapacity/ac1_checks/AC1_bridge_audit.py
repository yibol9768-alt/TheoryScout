"""完整联合散布检查：独立于残差根积分的直接 D,V 表示。"""
import os, json, time, resource
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from AC1_simulate import endpoint, bartlett, load

root=Path(__file__).resolve().parent
mom=json.loads((root/'serial/AC1_design.json').read_text())['moments']
rng=np.random.default_rng(2026110702)
wall,cpu=time.time(),time.process_time()
records=[]
for d in [64,128,256,512]:
    for method in ['GD3','Sign2','Adam2']:
        n,seff=load(d,mom[method]['c'],0.)
        b=ndtri(1-1/(n-1))
        for rep in range(12):
            L=bartlett(rng,d,n)
            B=bartlett(rng,d,n-d)
            G=rng.standard_normal((d,d))
            S=L@L.T
            Z=G@L.T
            U=endpoint(S,Z,method)
            unorm=np.linalg.norm(U)
            C=np.block([[S,Z.T/np.sqrt(n)],
                        [Z/np.sqrt(n),G@G.T/n+(n-d)/n*(B@B.T)]])
            eig=np.linalg.eigvalsh(C)
            delta=np.max(abs(eig-1))
            R=np.linalg.norm(U,2)/unorm
            mu=np.sum(U*Z)/np.sqrt(n)/unorm
            q=256
            gg=rng.standard_normal((2*d,q))
            rr=np.sqrt(np.sum(gg*gg,axis=0)+rng.chisquare(n-2*d,q))
            uu=gg/rr
            xx=L@uu[:d]
            yy=G@uu[:d]/np.sqrt(n)+np.sqrt((n-d)/n)*(B@uu[d:])
            D=n*np.sum(yy*(U@xx),axis=0)/unorm
            ay=L.T@(U.T@yy)/unorm
            V=(n*n*np.sum(ay*ay,axis=0)-D*D)/(n-1)
            if np.min(V)<-1e-10: raise RuntimeError('negative V')
            e=b*(np.sqrt(np.maximum(V,0))-1)
            records.append({'d':d,'method':method,'n':n,'rep':rep,'s_eff':seff,
                            'b_delta':float(b*delta),'b_R':float(b*R),'mu_minus_mu0':float(mu-(b+seff)),
                            'b3_n':float(b**3/n),'energy_scaled_mean':float(e.mean()),
                            'energy_scaled_RMS':float(np.sqrt(np.mean(e*e))),
                            'centered_D_mean':float((D-mu).mean()),'centered_D_SD':float(D.std(ddof=1))})
cg=Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
cgp=Path('/sys/fs/cgroup')/cg.lstrip('/')
res={'wall_seconds':time.time()-wall,'cpu_seconds':time.process_time()-cpu,
     'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
     'process_threads':next(s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('Threads:')),
     'env_threads':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}}
for k in ['memory.max','memory.swap.max','memory.peak']:
    try:res[k]=(cgp/k).read_text().strip()
    except OSError:pass
with (root/'AC1_bridge_audit.json').open('x') as f:
    json.dump({'seed':2026110702,'B':12,'Q':256,'rows':records,'resources':res},f,indent=2)
print(json.dumps(res))
