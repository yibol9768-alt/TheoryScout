"""Independent MP spectral quadrature, Wishart sampling, and exact-trajectory checks."""
import hashlib
import json
import math
from pathlib import Path
import urllib.request

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(10062026)
results = {'seed':10062026, 'mp_inner':[], 'wishart':[], 'rotation':[]}

# Evaluate MP integrals from the density, not the submitted closed formulas.
for gamma in (1.01, 2., 16., 64.):
    lam = 1/gamma
    for p,r in ((.031,.18),(.41,.41),(.86,.97)):
        dp,dr = 1-lam+lam*p,1-lam+lam*r
        w,v = p*dp/(1-p), r*dr/(1-r)
        t,u = p*w, r*v
        js = []
        for j in (0,1,2):
            def integrand(theta):
                s = 1+lam-2*math.sqrt(lam)*math.cos(theta)
                return 2/math.pi*math.sin(theta)**2*s**(j-1)/((s+w)*(s+v))
            js.append(quad(integrand,0,math.pi,epsabs=1e-11,epsrel=1e-11)[0])
        B = 1-lam*(1-p)*(1-r)
        expected = [(1-p)*(1-r)/(dp*dr*B), (1-p)*(1-r)/B,
                    (1-p)*(1-r)*(1-lam+lam*(p+r))/B]
        actual_kernel = (js[0]*js[2]-js[1]**2)/(p*r*(1-js[2]))
        D = p+r-p*r-lam*(1-p)*(1-r)*(p+r)
        closed_kernel = lam*(1-p)**2*(1-r)**2/(dp*dr*p*r*D)
        errors = [abs(x-y) for x,y in zip(js,expected)]
        rel = abs(actual_kernel-closed_kernel)/closed_kernel
        assert max(errors)<1e-8 and rel<1e-7
        assert abs(lam*p**3+(1-lam)*p*p+t*p-t)<1e-12
        results['mp_inner'].append({'gamma':gamma,'p':p,'r':r,
                                    'J_errors':errors,'kernel_relative_error':rel})

# Bartlett decomposition samples the same real Wishart S without d*gamma*d storage.
# Work in the S eigenbasis; the independent H remains orthogonally invariant.
for gamma in (16,64):
    for dimension in (512,1024):
        for replicate in range(3):
            m = gamma*dimension
            L = np.tril(rng.normal(size=(dimension,dimension)),-1)
            L[np.diag_indices(dimension)] = np.sqrt(rng.chisquare(m-np.arange(dimension)))
            s = eigh(L@L.T/m,eigvals_only=True,check_finite=False)
            G = rng.normal(size=(dimension,dimension))
            H = G.T@G/dimension
            A = np.sqrt(s)[:,None]*H*np.sqrt(s)[None,:]
            eig,U = eigh(A,check_finite=False)
            T = (U*np.sqrt(np.maximum(eig,0)))@U.T
            mu = np.trace(T)/dimension
            coefficients = (s[:,None]-s[None,:])**2/(s[:,None]*s[None,:])
            k = np.sum(coefficients*T*T)/(2*dimension)
            direct_k = np.sum((s[None,:]/s[:,None])*T*T)/dimension-np.trace(A)/dimension
            assert abs(k-direct_k)<2e-13
            # This plug-in is a diagnostic, not a finite-d variance theorem.
            nu = 1+k/gamma-mu*mu/gamma**2
            results['wishart'].append({'gamma':gamma,'d':dimension,'replicate':replicate,
                                       'mu':mu,'k':k,'nu_plugin':nu,
                                       'mixed_identity_error':abs(k-direct_k)})
        print(json.dumps({'gamma':gamma,'d':dimension,'matrix_checks':'done'}),flush=True)

# Finite-sample sample-index rotations for ALL THREE actual square-loss recursions.
d,n = 7,23
X,Y = rng.normal(size=(2,d,n))/math.sqrt(d)
small,_ = np.linalg.qr(rng.normal(size=(n-2,n-2)))
P = np.eye(n)
P[2:,2:] = small

def trajectories(X,Y):
    S,Z = (d/n)*(X@X.T), (d/math.sqrt(n))*(Y@X.T)
    outputs = {}
    for branch in ('GD','Sign','Adam'):
        U = np.zeros((d,d)); first=np.zeros_like(U); second=np.zeros_like(U)
        for step in range(1,5):
            residual=U@S-Z
            if branch=='GD':
                U -= .6*residual
            elif branch=='Sign':
                U -= .3*np.sign(residual)
            else:
                first=.7*first+.3*residual
                # Includes the beta2=0<beta1 corner explicitly.
                second=residual*residual
                U -= .3*(first/(1-.7**step))/(np.sqrt(second)+.2)
        outputs[branch] = U
    return outputs

old,new = trajectories(X,Y),trajectories(X@P,Y@P)
for branch in old:
    error=float(np.max(np.abs(old[branch]-new[branch])))
    L=Y.T@old[branch]@X
    rotated=(Y@P).T@new[branch]@(X@P)
    score_error=float(np.max(np.abs(rotated-P.T@L@P)))
    assert error<1e-11 and score_error<1e-11
    results['rotation'].append({'branch':branch,'trajectory_max_error':error,
                                'conjugacy_max_error':score_error})

url='https://arxiv.org/html/2605.05189v2'
request=urllib.request.Request(url,headers={'User-Agent':'ProofReview10/1.0'})
with urllib.request.urlopen(request,timeout=45) as response:
    snapshot=response.read()
(OUT/'barnfield_v2.html').write_bytes(snapshot)
results['literature']={'url':url,'sha256':hashlib.sha256(snapshot).hexdigest(),
                       'bytes':len(snapshot)}
(OUT/'structural_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({'finished':True,'literature':results['literature']}),flush=True)
