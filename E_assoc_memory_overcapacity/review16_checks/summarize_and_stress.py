"""Small independent quadrature, adversarial low-rank check, audit summaries."""
import csv
import itertools
import json
import math
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import ndtri
from independent_audit import moment_data, endpoint, dump, save_csv, INPUTS, HASHES
import hashlib

OUT=Path(__file__).resolve().parent; ROOT=OUT.parent
rng=np.random.default_rng(16061109)
nodes,weights=hermegauss(5); weights/=math.sqrt(2*math.pi)
errors=[]
for rep in range(8):
    p=3; n=29; d=2; q=.7
    A=rng.standard_normal((p,p))
    if rep==0: A=np.outer(A[:,0],A[:,1])
    if rep==1: A=A-A.T
    M=(A+A.T)/2; N=A.T@A
    info=moment_data(M,N,n,d,q); exact=np.zeros(3)
    c=n/d/math.sqrt(q)
    for ix in itertools.product(range(5),repeat=p):
        g=nodes[list(ix)]; weight=float(np.prod(weights[list(ix)]))
        a=float(g@M@g); b=float(g@N@g)
        s2=a*a/(n*(n+2))
        sv=b/n-s2
        sv2=b*b/(n*(n+2))-2*a*a*b/(n*(n+2)*(n+4))+a**4/(n*(n+2)*(n+4)*(n+6))
        exact+=weight*np.array([c*c*s2,c*c/(n-1)*sv,c**4/(n-1)**2*sv2])
    target=np.array([info['D2'],info['EV'],info['EV2']])
    errors.append(float(np.max(np.abs(exact-target)/np.maximum(1,np.abs(target)))))

# A rank-one allowed endpoint on C=I: conditional theorem still holds but
# increasing d alone supplies neither s_* uniform nor gamma=O(1/sqrt(d)).
stress=[]
for d in (128,256,512):
    n=d*d//4; p=3*d
    A=np.zeros((p,p)); A[d,0]=1.
    M=(A+A.T)/2; N=A.T@A
    info=moment_data(M,N,n,d,1.)
    result=endpoint(M,N,n,d,1.,float(ndtri(.9)),info)[0]
    stress.append(dict(d=d,sd=result['sd'],gamma=result['gamma'],
                       required_G=result['gamma']*math.sqrt(d),
                       v=info['v'],width=min(1.,result['upper'])-max(0.,result['lower'])))

rows=list(csv.DictReader((OUT/'conditional_all.csv').open()))
formal=list(csv.DictReader((OUT/'formal_comparison.csv').open()))
summaries=[]
for d in (128,256,512):
    group=[v for v in rows if int(v['d'])==d]
    comparison=[v for v in formal if int(v['d'])==d]
    summaries.append(dict(d=d,cells=len(group),covered=sum(v['covered']=='True' for v in group),
        min_width=min(float(v['width']) for v in group),max_width=max(float(v['width']) for v in group),
        mean_width=float(np.mean([float(v['width']) for v in group])),
        E_range=[min(float(v['E_envelope']) for v in group),max(float(v['E_envelope']) for v in group)],
        G_range=[min(float(v['G_envelope']) for v in group),max(float(v['G_envelope']) for v in group)],
        s_range=[min(float(v['s_star']) for v in group),max(float(v['s_star']) for v in group)],
        cor4_error_range=[min(float(v['cor4_error']) for v in group),max(float(v['cor4_error']) for v in group)],
        cor4_covers=sum(abs(float(v['observed'])-float(v['cor4_center']))<=float(v['cor4_error']) for v in group),
        formal_descriptive_covered=sum(float(v['mean_conditional_lower'])<=float(v['formal_observed'])<=float(v['mean_conditional_upper']) for v in comparison)))
selected=[]
for d in (128,256,512):
    old=[v for v in formal if int(v['d'])==d and float(v['r'])==0 and v['task']=='old']
    gd=next(v for v in old if v['method']=='GD'); sign=next(v for v in old if v['method']=='Sign')
    selected.append(dict(d=d,GD=[float(gd[k]) for k in ('mean_conditional_lower','mean_conditional_upper','mean_mc','formal_observed')],
        Sign=[float(sign[k]) for k in ('mean_conditional_lower','mean_conditional_upper','mean_mc','formal_observed')],
        difference_interval=[float(gd['mean_conditional_lower'])-float(sign['mean_conditional_upper']),float(gd['mean_conditional_upper'])-float(sign['mean_conditional_lower'])],
        formal_difference=float(gd['formal_observed'])-float(sign['formal_observed'])))
raw=json.loads((OUT/'raw_data_check.json').read_text())
result=dict(quadrature_relative_errors=errors,max_quadrature_relative_error=max(errors),
            low_rank_stress=stress,dimension_summaries=summaries,r0_old_comparisons=selected,
            raw_cells=len(raw['rows']),raw_all_covered=all(v['covered'] for v in raw['rows']),
            max_raw_D_error=max(v['max_D_identity_error'] for v in raw['rows']),
            max_raw_V_error=max(v['max_V_identity_error'] for v in raw['rows']),
            reviewed_inputs_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in HASHES.items()))
dump('audit_aggregates.json',result)
print(json.dumps(result,indent=2),flush=True)
