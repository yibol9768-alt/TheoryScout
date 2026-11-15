"""Well-conditioned deterministic scatter giving an actual rank-one MR Sign U.
This tests the necessity of the moment envelope, not the conditional theorem.
"""
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from independent_audit import moment_data, endpoint
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
co=list(csv.DictReader((ROOT/'y3_checks/Y3_coefficients.csv').open()))
param=next(v for v in co if float(v['r'])==0 and v['method']=='Sign' and v['task']=='old')
a=float(param['amplitude']); q=float(param['q']); rows=[]
for d in (128,256,512):
    n=d*d//4; ones=np.ones((d,d)); I=np.eye(d); O=np.zeros((d,d)); delta=.1
    # Full C=[[I,0,delta*J/d],[0,I,0],[delta*J/d,0,I]], eig(C) in [.9,1.1].
    # Z0=0, H=sqrt(n)*delta*J/d=(delta/2)*J, so actual U=alpha*J.
    U=a*ones
    A=np.block([[O,O],[U,O]])
    M=(A+A.T)/2; N=A.T@A
    info=moment_data(M,N,n,d,q)
    stats=endpoint(M,N,n,d,q,float(ndtri(.9)),info)[0]
    rows.append(dict(d=d,C_eigenvalue_min=1-delta,C_eigenvalue_max=1+delta,
                     H_entry=delta/2,U_rank=1,sd=stats['sd'],gamma=stats['gamma'],
                     required_G=math.sqrt(d)*stats['gamma'],required_E=d*stats['e'],
                     fixed_threshold_lower=max(0.,stats['lower']),
                     fixed_threshold_upper=min(1.,stats['upper'])))
with (OUT/'mr_low_rank_stress.json').open('x') as f:
    json.dump(dict(rows=rows,note='Legal deterministic C; no assertion that this is a typical Wishart draw. Continuity yields nearby positive-density bad scatters.'),f,indent=2)
print(json.dumps(rows,indent=2))
