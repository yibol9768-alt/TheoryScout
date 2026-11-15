import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import json, math
from pathlib import Path
from fractions import Fraction as F
import numpy as np

OUT=Path(__file__).resolve().parent
cg=next(v[3:] for v in Path('/proc/self/cgroup').read_text().splitlines() if v.startswith('0::'))
base=Path('/sys/fs/cgroup')/cg.lstrip('/')
limits={k:(base/k).read_text().strip() for k in ('memory.max','memory.swap.max','memory.peak')}
assert limits['memory.max']=='2147483648' and limits['memory.swap.max']=='0'
threads=int(next(v.split()[1] for v in Path('/proc/self/status').read_text().splitlines() if v.startswith('Threads:')))
assert threads<=2
rng=np.random.default_rng(761903525)
errors=[]
for d,n in ((2,2),(5,4),(7,20),(16,256)):
    for eta in (.3,.6,.8):
        for rep in range(100):
            h=rng.normal(size=(d,n))
            s=h@h.T/n
            cmat=np.eye(d)-eta*s
            matrix=(s@cmat)[0,0]/math.sqrt((cmat@s@cmat)[0,0])
            unit=h[0]/np.linalg.norm(h[0])
            projections=h[1:]@unit
            orthogonal=h[1:]-projections[:,None]*unit
            a=float(np.dot(h[0],h[0])/n)
            b=float(np.dot(projections,projections))
            c=float(np.linalg.norm(orthogonal.T@projections)**2/b)
            k=1-eta*(a+b/n)
            scalar=math.sqrt(a)*k/math.hypot(k,eta*math.sqrt(b*c)/n)
            errors.append(abs(matrix-scalar))
assert max(errors)<1e-11

# Exact rational multiplication of the Taylor series in e (degree 1), f (degree 2).
coefficients=[]
for eta in (F(0),F(1,3),F(3,5),F(4,5),F(99,100)):
    t=1-eta
    a=(1-2*eta)/t
    b=-eta/t
    u=(1-3*eta)/t
    v=eta*(3*eta-2)/(t*t)
    ce=a-u/2
    cf=b-v/2
    cee=3*u*u/8-a*u/2
    assert ce==F(1,2)
    assert cf==-eta*eta/(2*t*t)
    assert cee==-(1-3*eta)*(1+eta)/(8*t*t)
    coefficients.append(dict(eta=str(eta),e=str(ce),f=str(cf),e_squared=str(cee)))
with (OUT/'algebra.json').open('x') as f:
    json.dump(dict(matrix_scalar_max_error=max(errors),matrix_scalar_cases=len(errors),
                   coefficients=coefficients,limits=limits,process_threads=threads),f,indent=2)

small=json.loads((OUT/'sign_small.json').read_text())
print('SMALL SIGN')
for r in small:
    s=r['results']['samplevar']
    maxz=max(abs(v['z']) for v in r['results'].values())
    print(r['d'],r['n'],'samplevar',s['mean'],'SE',s['se'],'target',s['target'],'max|z|',maxz)
print('DIRECT')
direct=json.loads((OUT/'direct_summary.json').read_text())
for r in direct['rows']:
    print(r['d'], 'actual',r['actual']['mean'],r['actual']['se'],
          'RB',r['rb']['mean'],r['rb']['se'], 'first',r['rb']['target'],
          'actual-RB z',r['actual_minus_rb']['z'],
          'sign var',r['sign_samplevar']['mean'],r['sign_samplevar']['se'])
print('REDUCED')
reduced=json.loads((OUT/'chi_reduction.json').read_text())
for r in reduced:
    print(r['eta'],r['kappa'],r['d'],r['mean'],r['se'],'limit',r['target'])
print('COMPATIBILITY')
print((OUT/'compatibility.json').read_text())
print('METADATA')
print((OUT/'metadata.json').read_text())
print('ALGEBRA',max(errors))
