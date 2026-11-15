"""Post-analysis numerical audit; no new random observations or gate changes."""
import csv
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.special import ndtr, roots_hermite
from design import DIMS,LOADS,ALPHAS,METHODS,snr,predict
from run_v2 import resources,check_freeze

start=time.time()
meta=resources()
check_freeze()
base=Path(__file__).resolve().parents[2]
out=base/'experiments/killer'
for line in (out/'IMPLEMENTATION_FIX.sha256').read_text().splitlines():
    digest,name=line.split('  ',1)
    assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest
c=json.loads((out/'summary.json').read_text())['c']
z,w=roots_hermite(384)
z*=np.sqrt(2); w/=np.sqrt(np.pi)
worst={'difference':0.}
for d in DIMS:
    for k in LOADS:
        for a in ALPHAS:
            for m in METHODS:
                mu=snr(d,k,a,m)
                p384=float((ndtr(c*mu[:,None]+z[None,:])**(d*k-1)@w).mean())
                diff=abs(p384-predict(mu,d*k,c))
                if diff>worst['difference']:
                    worst={'difference':diff,'d':d,'load':k,'alpha':a,'method':m}
with (out/'method_summary.csv').open() as f:
    methods=list(csv.DictReader(f))
with (out/'cell_summary.csv').open() as f:
    cells=list(csv.DictReader(f))
chosen=[]
for cell in cells:
    if float(cell['alpha'])<1: continue
    name=f"delta_{float(cell['beta_pred']):g}"
    chosen.append(next(r for r in methods if r['method']==name and
                 all(r[k]==cell[k] for k in ('d','load','alpha'))))
audit={'quadrature_96_vs_384_max':worst,
       'selected_beta_absolute_max_error':max(float(r['abs_error']) for r in chosen),
       'selected_beta_absolute_fail_count':sum(float(r['abs_error'])>.03 for r in chosen),
       'formal_rows':sum(1 for _ in (out/'raw.jsonl').open()),
       'elapsed_seconds':time.time()-start,
       'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       'resource_metadata':meta}
(out/'audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({k:v for k,v in audit.items() if k!='resource_metadata'},indent=2))
