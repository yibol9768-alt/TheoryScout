"""Descriptive 3-dimension P2 readout; no new gates or data."""
import csv,json,math
from collections import defaultdict
import numpy as np
from design import *
from analyze import stat
by=defaultdict(dict)
wanted={'Sign1','Sign2_a0.6_actual','Sign2_a1.5_actual'}
for line in (OUT/'formal_rows.jsonl').open():
    r=json.loads(line)
    if r['name'] in wanted and r['kappa'] in (1.,2.):by[r['d'],r['kappa'],r['name']][r['seed']]=r
rr=[]
for orig,k,alpha,metric in FOCI:
    name=f'Sign2_a{alpha:g}_actual';target=pred(sign_c(alpha),k,metric)-pred(math.sqrt(2/math.pi),k,metric)
    for d in (128,256,512):
        aa=by[d,k,name];bb=by[d,k,'Sign1'];values=[]
        for seed in sorted(aa.keys()&bb.keys()):values.append(aa[seed]['full'][metric]-bb[seed]['full'][metric])
        mean,se=stat(values)
        rr.append(dict(original_d=orig,d=d,kappa=k,name=name,metric=metric,seeds=len(values),queries=512,
            paired_difference=mean,paired_SE=se,asymptotic_target=target,baseline='Sign1',
            readout='frozen G1 original/high' if d in (orig,512) else 'descriptive intermediate; no added test'))
with (OUT/'P2_dimension_gradient.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
print('Saved 12 descriptive dimension readouts; no additional gate')
