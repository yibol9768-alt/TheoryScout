"""Independent Gaussian quadrature of the iid-coordinate variance reference.
This is NOT a simulation of finite-dimensional training data.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import roots_hermitenorm
from records_numeric import endpoint, z, weights, match, resources

resources()
h,hw=roots_hermitenorm(10);hw=hw/np.sqrt(2*np.pi)
sigma=np.sqrt(1.5);tau=np.sqrt(.625)
Z0=tau*h[:,None]-sigma*z[None,:]/2
rows=[]
for e in [.5,2]:
    arms=[match(e,.3,*b) for b in [(0,.99),(.9,0),(.9,.99)]]
    psi=[]
    for arm,(b1,b2) in zip(arms,[(0,.99),(.9,0),(.9,.99)]):
        displacement=endpoint(arm['a'],e,b1,b2)
        U=tau*h[:,None]+sigma*(displacement[None,:]-z[None,:]/2)
        # Recompute q directly in the joint (H,|G|) quadrature.
        q=hw@(U*U)@weights
        c=hw@(U*Z0)@weights/np.sqrt(q)
        f=U*Z0/np.sqrt(q)-c*U*U/(2*q)-c/2
        assert abs(hw@f@weights)<1e-13
        psi.append(f)
    for j,kind in [(0,'beta1'),(1,'beta2')]:
        sd=np.sqrt(hw@((psi[2]-psi[j])**2)@weights)/2048
        gap=arms[2]['c0']-arms[j]['c0']
        rows.append(dict(e=e,kind=kind,iid_sd_d2048=sd,budget=2*abs(gap),inflation=2*abs(gap)/sd))
with Path('review21_checks/design_reference.json').open('x') as f:json.dump(dict(rows=rows,resources=resources()),f,indent=2)
print(json.dumps(rows,indent=2))
