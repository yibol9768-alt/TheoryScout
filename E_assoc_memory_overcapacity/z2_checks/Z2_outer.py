"""Independent Wishart outer integration, frozen N=512, d=512, r=0.

Uses rigorous analytic lower/upper expressions; ordinary floating-point evaluation
is a numerical illustration of the confidence theorem, not an interval certificate.
No y3d/formal inputs. No parameter selection or early stopping.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key)=='2'
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
from Z2_bridge import (moments, compressed_matrices, fixed_threshold_bound,
                       sphere_thresholds, threadpool_info)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'z2_checks'
CONFIG=dict(seed=2026110603,N=512,d=512,r=0.,rho=.1,eta=.002,
            delta_outer=.01,methods=['GD','Sign'],task='old')

def dump(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)

def main():
    start=time.monotonic()
    inputs=['experiments/x1/X1_population.json','y3_checks/Y3_coefficients.csv',
            'z2_checks/Z2_bridge.py','z2_checks/Z2_outer.py']
    hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in inputs}
    dump('Z2_outer_design.json',dict(**CONFIG,input_hashes=hashes))
    coeff=list(csv.DictReader((ROOT/'y3_checks/Y3_coefficients.csv').open()))
    coeff={m:next(c for c in coeff if float(c['r'])==0 and c['method']==m and c['task']=='old')
           for m in CONFIG['methods']}
    d=CONFIG['d'];n=d*d//4
    zlo,zhi=sphere_thresholds(n,CONFIG['rho'],CONFIG['eta'])
    rng=np.random.default_rng(CONFIG['seed'])
    rows=[]
    for rep in range(CONFIG['N']):
        # Marginal Bartlett construction of (S,C00,C0x,CEx); no CEE is needed.
        Bxx=np.tril(rng.normal(size=(d,d)),-1)
        Bxx[np.diag_indices(d)]=np.sqrt(rng.chisquare(n-np.arange(d)))
        B00=np.tril(rng.normal(size=(d,d)),-1)
        B00[np.diag_indices(d)]=np.sqrt(rng.chisquare(n-d-np.arange(d)))
        G0=rng.normal(size=(d,d));GE=rng.normal(size=(d,d))
        S=Bxx@Bxx.T/n
        C0x=G0@Bxx.T/n
        C00=(G0@G0.T+B00@B00.T)/n
        Z0=math.sqrt(n)*C0x
        ZE=GE@Bxx.T/math.sqrt(n)
        H=ZE-Z0@S
        vals={}
        for method in CONFIG['methods']:
            amplitude=float(coeff[method]['amplitude']);q=float(coeff[method]['q'])
            U=Z0+amplitude*(H if method=='GD' else np.sign(H))
            M,N=compressed_matrices(S,C00,C0x,U)
            mom=moments(M,N,n,d,q)
            low=fixed_threshold_bound(M,N,n,d,q,zhi,mom)
            high=fixed_threshold_bound(M,N,n,d,q,zlo,mom)
            vals[method]=(max(0,low['lower']-CONFIG['eta']),
                          min(1,high['upper']+CONFIG['eta']))
        row=dict(rep=rep,GD_lower=vals['GD'][0],GD_upper=vals['GD'][1],
                 Sign_lower=vals['Sign'][0],Sign_upper=vals['Sign'][1],
                 difference_lower=vals['GD'][0]-vals['Sign'][1],
                 difference_upper=vals['GD'][1]-vals['Sign'][0])
        rows.append(row)
        if (rep+1)%32==0:
            print(json.dumps(dict(completed=rep+1,N=CONFIG['N'],
                                 wall_seconds=time.monotonic()-start)),flush=True)
    with (OUT/'Z2_outer_rows.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    mean_lo=float(np.mean([r['difference_lower'] for r in rows]))
    mean_hi=float(np.mean([r['difference_upper'] for r in rows]))
    radius=math.sqrt(2*math.log(2/CONFIG['delta_outer'])/CONFIG['N'])
    assert hashes=={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in inputs}
    cg=Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
    cgp=Path('/sys/fs/cgroup')/cg.lstrip('/')
    dump('Z2_outer_result.json',dict(**CONFIG,mean_difference_lower=mean_lo,
         mean_difference_upper=mean_hi,hoeffding_radius=radius,
         expected_difference_lower=max(-1,mean_lo-radius),
         expected_difference_upper=min(1,mean_hi+radius),
         wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),
         peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         memory_max=(cgp/'memory.max').read_text().strip(),
         memory_swap_max=(cgp/'memory.swap.max').read_text().strip(),
         threadpools=threadpool_info(),input_hashes_unchanged=True,
         rounding_status='ordinary float64; not outward-rounded interval arithmetic'))

if __name__=='__main__':main()
