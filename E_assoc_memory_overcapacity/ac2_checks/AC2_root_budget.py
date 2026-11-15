"""Analytic exceptional-branch probability budgets; no historical code imports."""
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtri, betainc
from scipy.stats import beta, binom

root = Path(__file__).resolve().parent
design = json.loads((root.parent/'ac1_checks/serial/AC1_design.json').read_text())
rows=[]
for d in (64,128,256,512):
    for method,mom in design['moments'].items():
        for s in (-.75,0.,.75):
            c=mom['c']; lo,hi=4,2*d*d
            def f(n): return c*d/np.sqrt(n)-ndtri(1-1/(n-1))-s
            while hi-lo>1:
                mid=(hi+lo)//2
                if f(mid)>0: lo=mid
                else: hi=mid
            n=min((lo,hi),key=lambda k: abs(f(k))); m=n-1
            # h<=1/2 and T<=sqrt(m/2), T>0 imply H>=m/4>0.
            neg=float(binom.cdf(4,m,.5))
            leverage=float(beta.sf(.5,d/2,(n-d)/2))
            coordtail=.5*float(betainc((m-1)/2,.5,.5))
            for L in (1,5):
                neg=float(binom.cdf(L-1,m,.5))
                threshold=min(1.,m/L*coordtail)
                rows.append(dict(d=d,n=n,method=method,s=s,L=L,
                    negative_T_probability=neg,leverage_tail=leverage,
                    sphere_threshold_tail=threshold,
                    root_exception_union_bound=min(1.,neg+leverage+threshold)))
out=dict(rows=rows,simultaneous_99_Hoeffding_halfwidth=float(np.sqrt(np.log(144/.01)/(2*128))),
         note='Bounds T<=0 or H<=0; Ux=0 is handled by exact tied-score probability L/n, not deletion. Underflow zeros are float64, not certified exact zeros.')
with (root/'AC2_root_budget.json').open('x') as f: json.dump(out,f,indent=2)
print(json.dumps(dict(max_union_bound=max(x['root_exception_union_bound'] for x in rows),
                     halfwidth=out['simultaneous_99_Hoeffding_halfwidth'])))
