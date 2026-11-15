"""Deterministic polynomial quadrature and event-equivalence checks."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key)=='2'
import itertools
import json
import math
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from Z2_bridge import moments

def main():
    rng=np.random.default_rng(2026110604)
    p,n,d,q=4,43,2,.7
    A=rng.normal(size=(p,p));M=(A+A.T)/2;N=A.T@A
    m1,n1=np.trace(M),np.trace(N);M2=M@M
    target=np.array([m1*m1+2*np.trace(M2),n1*n1+2*np.trace(N@N),
        n1*(m1*m1+2*np.trace(M2))+4*m1*np.trace(M@N)+8*np.trace(M2@N),
        m1**4+12*m1*m1*np.trace(M2)+12*np.trace(M2)**2
        +32*m1*np.trace(M2@M)+48*np.trace(M2@M2)])
    nodes,weights=hermegauss(5);weights=weights/math.sqrt(2*math.pi)
    result=np.zeros(4)
    for ix in itertools.product(range(5),repeat=p):
        g=nodes[list(ix)];w=np.prod(weights[list(ix)])
        x,y=g@M@g,g@N@g
        result+=w*np.array([x*x,y*y,x*x*y,x**4])
    relative=float(np.max(np.abs(result-target)/np.maximum(1,np.abs(target))))
    assert relative<2e-13
    # At every tested shift, sphere event and homogeneous Gaussian event agree.
    mom=moments(M,N,n,d,q);v=mom['v'];c=n/(d*math.sqrt(q));z=1.2815515655446004
    K=c*M-z*c*c/(2*(n-1)*math.sqrt(v))*N
    gg=rng.normal(size=(1000,n));W=np.sum(gg*gg,axis=1)
    a=np.einsum('bi,ij,bj->b',gg[:,:p],K,gg[:,:p])
    errors=[]
    for eps in (0.,.01,.2):
        threshold=z*math.sqrt(v)/2-eps
        sphere=a/W>=threshold
        homogeneous=a-threshold*W>=0
        assert np.array_equal(sphere,homogeneous)
        errors.append(int(np.sum(sphere!=homogeneous)))
    payload=dict(polynomial_quadrature_relative_error=relative,
                 event_mismatch_counts=errors,quadrature_order=5,dimension=p,
                 note='Polynomial quadrature is exact in real arithmetic up to coordinate degree 9; float evaluation shown.')
    with (Path(__file__).parent/'Z2_exact_verification.json').open('x') as f:
        json.dump(payload,f,indent=2)
    print(json.dumps(payload))

if __name__=='__main__':main()
