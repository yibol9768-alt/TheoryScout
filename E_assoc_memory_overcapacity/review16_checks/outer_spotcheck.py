"""Independently rebuild selected original outer draws, no project-code imports."""
import csv
import json
import math
from pathlib import Path
import numpy as np
from independent_audit import moment_data, endpoint, bands, threadpool_info

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
d=512; n=d*d//4
coeff=list(csv.DictReader((ROOT/'y3_checks/Y3_coefficients.csv').open()))
stored=list(csv.DictReader((ROOT/'z2_checks/Z2_outer_rows.csv').open()))
rng=np.random.default_rng(2026110603)
zlo,zhi=bands(n)
rows=[]
for rep in range(512):
    # Same random stream is necessary to identify existing draws, but the score
    # representation below is direct, without the reviewed Gram compression.
    xx=np.tril(rng.standard_normal((d,d)),-1)
    xx[np.diag_indices(d)]=np.sqrt(rng.chisquare(n-np.arange(d)))
    yy=np.tril(rng.standard_normal((d,d)),-1)
    yy[np.diag_indices(d)]=np.sqrt(rng.chisquare(n-d-np.arange(d)))
    gx=rng.standard_normal((d,d)); ge=rng.standard_normal((d,d))
    if rep not in (0,1,127,255,511): continue
    lx=np.concatenate([xx,np.zeros((d,d))],axis=1)/math.sqrt(n)
    ly=np.concatenate([gx,yy],axis=1)/math.sqrt(n)
    S=lx@lx.T; Z0=math.sqrt(n)*(ly@lx.T)
    H=ge@xx.T/math.sqrt(n)-Z0@S
    for method in ('GD','Sign'):
        param=next(v for v in coeff if float(v['r'])==0 and v['task']=='old' and v['method']==method)
        U=Z0+float(param['amplitude'])*(H if method=='GD' else np.sign(H))
        A=ly.T@U@lx; M=(A+A.T)/2; N=A.T@A; q=float(param['q'])
        info=moment_data(M,N,n,d,q)
        lo=endpoint(M,N,n,d,q,zhi,info)[0]
        hi=endpoint(M,N,n,d,q,zlo,info)[0]
        lower=max(0.,lo['lower']-.002); upper=min(1.,hi['upper']+.002)
        sl=float(stored[rep][method+'_lower']); su=float(stored[rep][method+'_upper'])
        rows.append(dict(rep=rep,method=method,lower=lower,upper=upper,
                         stored_lower=sl,stored_upper=su,error=max(abs(lower-sl),abs(upper-su))))
    print(json.dumps(dict(rep=rep,max_error=max(v['error'] for v in rows))),flush=True)
with (OUT/'outer_spotcheck.json').open('x') as f:
    json.dump(dict(rows=rows,max_error=max(v['error'] for v in rows),
                   threadpools=threadpool_info(),rounding_certificate=False),f,indent=2)
