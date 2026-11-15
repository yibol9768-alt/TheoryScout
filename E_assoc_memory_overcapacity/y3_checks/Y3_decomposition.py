"""Read frozen scalar coefficients; deterministic matching tangent only."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    assert os.environ.get(k)=='2'
import csv,json,math
from pathlib import Path
P=Path(__file__).resolve().parent
with (P/'Y3_coefficients.csv').open() as f:
    rows=list(csv.DictReader(f))
groups={}
for r in rows:groups.setdefault((r['ir'],r['method']),{})[r['task']]=r
out=[]
for (_,method),g in groups.items():
    o,n=g['old'],g['new']
    alpha,q,beta,qh,sigma=[float(o[k]) for k in ['amplitude','q','beta','Q','sigma']]
    dq=-sigma*sigma*beta+2*alpha*qh
    deriv={}
    for task,rr in g.items():
        r=float(rr['r']);a=float(rr['a_k']);mu=float(rr['mu'])
        da=(r-1)*beta if task=='old' else (1-r)*beta
        dmu=(da-a*dq/(2*q))/math.sqrt(.25*q)
        deriv[task]=math.exp(-(1.2815515655446004-mu)**2/2)/math.sqrt(2*math.pi)*dmu
    assert deriv['new']>0
    bn,bo=float(n['B_recall']),float(o['B_recall'])
    ratio=deriv['old']/deriv['new']
    out.append(dict(r=float(o['r']),method=method,alpha=alpha,
                    old_derivative=deriv['old'],new_derivative=deriv['new'],
                    alpha_shift_coefficient=-bn/deriv['new'],
                    relative_shift_at_d128=-bn/(128*deriv['new']*alpha),
                    B_old=bo,B_new=bn,B_matching_tangent=ratio*bn,
                    B_old_after_hypothetical_matching=bo-ratio*bn))
with (P/'Y3_matching_tangent.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(json.dumps(out,ensure_ascii=False,indent=2))
