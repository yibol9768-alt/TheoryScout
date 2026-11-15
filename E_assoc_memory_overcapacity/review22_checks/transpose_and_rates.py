"""Supplement: test the exact distributional rescue for the transpose audit."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from scipy.special import gamma
root=Path(__file__).resolve().parent
rng=np.random.default_rng(22602)
rows=[]
for dim in (4,12,30):
    A=rng.normal(size=(dim,dim))
    K=(A+A.T)/2
    for t,s in ((.3,-.2),(1.,1.),(-2.,.1)):
        ev1=np.linalg.eigvalsh(t*K+s*A.T@A)
        ev2=np.linalg.eigvalsh(t*K+s*A@A.T)
        rows.append({'dim':dim,'t':t,'s':s,'max_pencil_eigenvalue_difference':float(abs(ev1-ev2).max())})
rate=[]
for d in (64,128,256,512,10**6,10**12,10**24):
    ell=np.log(float(d))
    rate.append({'d':d,'Sign_amplified_factor':float(ell**(2/3)/float(d)**(1/6)), 'GD_Adam_amplified_factor':float(ell/np.sqrt(float(d)))})
# Finite-m exception probability T_L<=0 is exactly binomial lower tail.
from scipy.stats import binom
out={'transpose_pencils':rows,'rates_without_constants':rate,
     'negative_top5_threshold_probability_m279':float(binom.cdf(4,279,.5))}
with (root/'transpose_and_rates.json').open('x') as f: json.dump(out,f,indent=2)
print(json.dumps(out))
