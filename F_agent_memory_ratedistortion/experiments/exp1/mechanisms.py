"""Predeclared mechanism supplements: same entropy and distractor load intervention."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,resource,math,time
from pathlib import Path
import numpy as np
from core import prepare,rng,entropy,retrieve
resource.setrlimit(resource.RLIMIT_AS,(1900000000,1900000000))
start=time.monotonic(); output={'scope':'specified-channel interventions, not a substitute for independently calibrated E1/E2'}
a=.7729078047806517; distributions=[np.array([.5,.5,0]),np.array([a,(1-a)/2,(1-a)/2])]; same=[]
for seed in (0,1,2):
    g=3;n=8192;r=rng(seed,g,10); z=r.integers(2,size=(g,32),dtype='int8'); e=r.random((g,4,32))<4/15;e[:,0,:]=False;source=z[:,None,:]^e
    rt=rng(seed,g,50); uu=rt.random(n); jj=rt.integers(4,size=n);tt=rt.integers(32,size=n); coin=rng(seed,g,52).integers(2,size=n)
    accuracies=[]
    for p in distributions:
        ii=np.searchsorted(np.cumsum(p),uu); truth=source[ii,jj,tt]; good=np.where(ii==0,z[ii,tt],coin)==truth
        same.append(dict(seed=seed,p=p.tolist(),entropy=entropy(p),pmax=float(p.max()),theoretical_risk=(1-.6*p.max())/2,actual_risk=float(1-good.mean()),n=n))
        accuracies.append(good)
    same[-1]['paired_accuracy_difference_vs_p']=float(np.mean(accuracies[1].astype(int)-accuracies[0].astype(int)))
output['same_entropy']=same
interventions=[]
for seed in (0,1,2):
    g=32;n=8192; p,ph,ii,jj,cs,cr,guess,vs,vr=prepare(seed,g,0,4/15,n,0,0)
    for lam in (31.,112.,255.):
        rt=rng(seed,g,53); hits={d:0 for d in (0,32,96)}; actual={d:0 for d in hits}; gains={d:0 for d in hits}; route=0
        for at in range(0,n,128):
            en=min(at+128,n); nn=en-at; bg=rt.gumbel(size=(nn,g)); pad=rt.gumbel(size=(nn,96)); row=np.arange(nn); target=bg[row,ii[at:en]]+math.log1p(lam); maximum=bg.max(1)
            route+=np.where(target>=maximum,cr[at:en],guess[at:en]).sum()
            for d in hits:
                mx=np.maximum(maximum,pad[:,:d].max(1)) if d else maximum
                hit=target>=mx; correct=np.where(hit,cs[at:en],guess[at:en]); actual[d]+=int(correct.sum());hits[d]+=int(hit.sum());gains[d]+=int(np.sum(hit*(2*cs[at:en].astype(int)-1)))
        for d in hits:
            interventions.append(dict(seed=seed,lambda_=lam,d=d,recall=hits[d]/n,predicted_recall=(lam+1)/(lam+g+d),accuracy=actual[d]/n,predicted_accuracy=.5+.3*(lam+1)/(lam+g+d),gain_no_guess=gains[d]/n,predicted_gain_ratio=(lam+g)/(lam+g+d),actual_gain_ratio=gains[d]/gains[0],route_accuracy=route/n,route_predicted_accuracy=.5+.5*(lam+1)/(lam+g),n=n))
output['load_intervention']=interventions;output['elapsed_seconds']=time.monotonic()-start;output['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
Path(__file__).with_name('mechanisms_results.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'elapsed_seconds':output['elapsed_seconds'],'same_entropy_rows':len(same),'load_rows':len(interventions)}))
