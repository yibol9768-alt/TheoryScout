"""Post-registration analytic audit. Does not change EXP2 prediction or gates."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE))
import numpy as np
import model
from run import verify_frozen

started=time.monotonic()
verify_frozen()
rows=[]
for alpha in (1.5,3.):
    for n in (4,16,32,64,128):
        i=np.arange(1,n+1,dtype=float)
        e=i[:,None]**((alpha-2)/2)*np.tile(np.eye(4),(n//4,1))
        lam=i**-alpha
        q=model.polar(lam[:,None]*e)
        a=np.sqrt(lam)[:,None]*e
        cosine=float(np.sum(lam[:,None]*q*a)/np.sqrt(model.norm2(q,lam)*model.norm2(a,lam)))
        sine=float(np.sqrt(max(0.,1-cosine**2)))
        rows.append(dict(alpha=alpha,n=n,min_relative_error_over_all_scalars=sine,
                         min_prediction_relative_error_over_all_scalars=sine))
assert rows[0]['min_relative_error_over_all_scalars']>.36
# Check exact expression in the n=r=4 example independently.
i=np.arange(1,5,dtype=float)
expression=float(np.sqrt(1-np.sum(i**-2.5)**2/(np.sum(i**-1.5)*np.sum(i**-3.5))))
assert abs(expression-rows[0]['min_relative_error_over_all_scalars'])<1e-14
# No new theorem is marked proved merely by compiling these problem fragments.
tex=[]
for path in sorted((ROOT/'lemmas_v3').glob('L*.tex')):
    s=path.read_text()
    assert s.count('\\begin{problem}')==s.count('\\end{problem}')==1
    assert s.count('\\[')==s.count('\\]')
    assert s.count('$')%2==0
    depth=0
    for j,c in enumerate(s):
        if c in '{}' and (j==0 or s[j-1]!='\\'):
            depth+=1 if c=='{' else -1
            assert depth>=0
    assert depth==0
    tex.append(dict(file=str(path.relative_to(ROOT)),static_delimiters_pass=True))
analysis=json.loads((HERE/'analysis.json').read_text())
runtime=sum(analysis[k]['elapsed_seconds'] for k in ('formal_manifest','smoke_manifest'))
runtime+=analysis['analysis_elapsed_seconds']+time.monotonic()-started
assert runtime<3600
assert all(s['static_delimiters_pass'] for s in tex)
out=dict(label='post-registration analytic supplement; excluded from P0-P8',
         counterexamples=rows,exact_n4_expression=expression,latex_static=tex,
         total_computation_seconds=runtime,elapsed_seconds=time.monotonic()-started,
         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
         supplement_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(HERE/'supplement.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
