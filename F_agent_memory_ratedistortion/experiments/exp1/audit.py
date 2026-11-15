"""Finite implementation audit; never changes preregistered criteria or formal data."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'
import json,itertools,resource
from pathlib import Path
import numpy as np
from core import retrieve,active_indices,rng,coefficients,value,fractional
from e0_checks import budget_dp
resource.setrlimit(resource.RLIMIT_AS,(1900000000,1900000000))
r=np.random.default_rng(729); cases=0; maxgap=0.
for g in (2,5):
    ii=r.integers(g,size=31); jj=r.integers(4,size=31)
    for lam,ws,wr in ((7.,1.,1.),(31.,.5,1.3)):
        score=r.gumbel(size=(31,5*g))+np.log(np.r_[np.full(g,ws),np.full(4*g,wr)])
        pad=r.gumbel(size=(31,3*g))+np.log(wr)
        for zz in itertools.product((0,1,2),repeat=g):
            z=np.array(zz); ids=active_indices(z)
            for padded in (False,True):
                fast=retrieve(score,z,ii,jj,'E1',(lam,ws,wr),pad=pad if padded else None)
                expected=[]
                for row in range(31):
                    mode=z[ii[row]]; target=int(ii[row] if mode==1 else g+4*ii[row]+jj[row]); vals=score[row].copy()
                    if mode: vals[target]+=np.log((lam+(ws if mode==1 else wr))/(ws if mode==1 else wr))
                    if len(ids):
                        winner=ids[np.lexsort((ids,-vals[ids]))[0]]
                        hit=mode>0 and winner==target and (not padded or vals[target]>pad[row].max())
                    else: hit=False
                    expected.append(hit)
                assert np.array_equal(fast,expected)
                cases+=1
    p=r.dirichlet(np.ones(g)); c,w,u=coefficients(31,.5,.6,1.)
    for b in range(4*g+1):
        got,z=budget_dp(p,c,u,w,b,31,tolerance=1e-11)
        exact=max(value(p,np.array(zz),31,.5,.6,1.) for zz in itertools.product((0,1,2),repeat=g) if c[list(zz)].sum()<=b)
        maxgap=max(maxgap,abs(got-exact)); assert abs(got-exact)<1e-10
        fractional(p,b,31,.5,.6,1.)
report={'direct_gumbel_reference_policy_cases':cases,'dp_vs_brute_max_gap':maxgap,'fractional_rounding_checks':'PASS','status':'PASS'}
Path(__file__).with_name('audit_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
