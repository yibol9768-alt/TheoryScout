"""Independent review code. No project code is imported or executed.

Only writes a previously nonexistent JSON beneath this script's directory.
The main experiment samples keys by Bartlett's construction, integrates output
rows, and uses a Gaussian variance derivative for the same-row norm correction.
An unrelated raw-array simulation checks those conditional integrations.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = Path(__file__).resolve().parent
assert Path.cwd() == ROOT
assert sys.executable == '/home/liuyibo/venvs/theory/bin/python'


def resources():
    status = Path('/proc/self/status').read_text()
    threads = int(next(s.split()[1] for s in status.splitlines() if s.startswith('Threads:')))
    cg = next(s.split(':', 2)[2] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
    base = Path('/sys/fs/cgroup') / cg.lstrip('/')
    mem, swap = [(base / f).read_text().strip() for f in ('memory.max', 'memory.swap.max')]
    assert threads <= 2 and mem == str(3 * 1024**3) and swap == '0'
    return dict(threads=threads, memory_max=mem, swap_max=swap, python=sys.executable)


def scalar_response(x, method, eps):
    if method == 'GD':
        return x
    if method == 'Sign':
        return np.sign(x)
    return x / (np.abs(x) + eps)


def moments(method, r, e):
    v = 2 * (1 - r)
    eps = e * math.sqrt(v)
    norm = math.sqrt(2 / math.pi)
    def integ(f):
        return quad(lambda x: norm * math.exp(-x*x/2) * f(x), 0, np.inf,
                    epsabs=3e-13, epsrel=3e-13)[0]
    h = lambda x: scalar_response(math.sqrt(v)*x, method, eps)
    beta = integ(lambda x: x*h(x)) / math.sqrt(v)
    Q = integ(lambda x: h(x)**2)
    db = integ(lambda x: (x**3-3*x)*h(x)) / (2*v**1.5)
    dq = integ(lambda x: (x*x-1)*h(x)**2) / (2*v)
    return beta, Q, db, dq


def predicted(method, r, alpha, e, kappa):
    beta, Q, db, dq = moments(method, r, e)
    v = 2*(1-r)
    q = 1-alpha*v*beta+alpha*alpha*Q
    a = np.array([1+alpha*(r-1)*beta, r+alpha*(1-r)*beta])
    mu = a/math.sqrt(kappa*q)
    m = alpha*(np.array([r-1, 1-r])*(3-2*r)*db-np.array([1,r])*beta)/(kappa**1.5*math.sqrt(q))
    ell = (1+2*alpha*(beta*(r-3)+(r-1)*(3-2*r)*db)+alpha*alpha*(2*v*beta*beta+(3-2*r)*dq))/(kappa*q)
    return dict(q=q, mu=mu.tolist(), m=m.tolist(), ell=ell, w=(ell+mu*mu).tolist())


class GaussianIntegration:
    def __init__(self, nodes=100):
        z, w = np.polynomial.legendre.leggauss(nodes)
        self.x = (z+1)*8  # [0,16]; folded Gaussian density
        w = w*8*math.sqrt(2/math.pi)*np.exp(-self.x*self.x/2)
        he = [np.ones(nodes), self.x.copy()]
        for j in range(1, 63):
            he.append((self.x*he[-1]-math.sqrt(j)*he[-2])/math.sqrt(j+1))
        self.W = np.array([he[j]*w for j in range(1,64,2)]).T
        self.w = w

    def response_data(self, v, method, eps):
        if method == 'GD':
            return np.ones_like(v), v.copy(), np.zeros_like(v), np.ones_like(v), None, None
        if method == 'Sign':
            beta = math.sqrt(2/math.pi)/np.sqrt(v)
            return beta, np.ones_like(v), -beta/(2*v), np.zeros_like(v), None, None
        root = np.sqrt(v)[:,None]
        x = root*self.x
        h = x/(x+eps)
        dh = eps/(x+eps)**2*self.x/(2*root)
        coeff = h@self.W
        dc = dh@self.W
        beta = coeff[:,0]/np.sqrt(v)
        dbeta = dc[:,0]/np.sqrt(v)-beta/(2*v)
        Q = (h*h)@self.w
        dQ = (2*h*dh)@self.w
        return beta, Q, dbeta, dQ, coeff, dc

    def row_energy(self, S, Z, J, K, method, alpha, eps, directions=()):
        d = len(S)
        v = np.diag(K).copy()
        beta, Q, dbeta, dQ, coeff, dc = self.response_data(v, method, eps)
        den = np.sqrt(v[:,None]*v[None,:])
        rho = np.clip(K/den, -1, 1)
        np.fill_diagonal(rho, 0)
        if method == 'GD':
            mh = K.copy()
            cutoff, tail = 1, 0.
        elif method == 'Sign':
            mh = 2/math.pi*np.arcsin(rho)
            np.fill_diagonal(mh, Q)
            cutoff, tail = 0, 0.
        else:
            rhomax = np.max(np.abs(rho))
            # L2/Parseval bound on omitted scalar covariance terms, summed in
            # the weighted trace; this does not assume the claimed 1/d answer.
            boundscale = np.sum(np.abs(S)*np.sqrt(Q[:,None]*Q[None,:]))/d
            cutoff = 11
            while cutoff < 63 and boundscale*rhomax**(cutoff+2) > 2e-13:
                cutoff += 4
            tail = boundscale*rhomax**(cutoff+2)
            mh = np.zeros_like(K)
            power = rho.copy()
            for j in range((cutoff+1)//2):
                mh += np.outer(coeff[:,j], coeff[:,j])*power
                power *= rho*rho
            np.fill_diagonal(mh, Q)
        bsum = beta[:,None]+beta[None,:]
        M = Z+alpha*J*bsum+alpha*alpha*mh
        F = np.sum(S*M)/d
        derivs = []
        for DZ, DJ, DK in directions:
            dv = np.diag(DK)
            drho = DK/den-rho*(dv[:,None]/v[:,None]+dv[None,:]/v[None,:])/2
            np.fill_diagonal(drho, 0)
            if method == 'GD':
                dm = DK
            elif method == 'Sign':
                dm = 2/math.pi*drho/np.sqrt(1-rho*rho)
                np.fill_diagonal(dm, 0)
            else:
                dm = np.zeros_like(K)
                for j in range((cutoff+1)//2):
                    m = 2*j+1
                    c = coeff[:,j]
                    dd = dc[:,j]*dv
                    dm += (np.outer(dd,c)+np.outer(c,dd))*rho**m
                    dm += np.outer(c,c)*m*rho**(m-1)*drho
                np.fill_diagonal(dm, dQ*dv)
            db = dbeta*dv
            DM = DZ+alpha*(DJ*bsum+J*(db[:,None]+db[None,:]))+alpha*alpha*dm
            derivs.append(float(np.sum(S*DM)/d))
        return float(F), beta, derivs, float(alpha*alpha*tail)

    def integrate(self, L, n, r, alpha, method, e, q):
        # Rows z0=g0 L, residual=g0 L(rI-S)+sqrt(1-r²)gE L.
        # Altering the variance of projected task-k labels yields the exact
        # same-row norm correction by differentiating the Gaussian density.
        S = L.T@L
        d = len(S)
        eye = np.eye(d)
        T0, T1 = r*eye-S, eye-r*S
        J = S@T0
        K = T0@S@T0+(1-r*r)*S
        dirs = [(S, J, T0@S@T0), (r*r*S, r*S@T1, T1@S@T1)]
        F, beta, derivs, tail = self.row_energy(S,S,J,K,method,alpha,e*math.sqrt(2*(1-r)),dirs)
        means, energies = [], []
        for k, (r0,r1) in enumerate(((1,r),(r,1))):
            # Direct Gaussian regression E[gk_l u_b], before squaring/summing.
            A = r0*L+alpha*(L@(r1*eye-r0*S))*beta[None,:]
            means.append(float(np.sum(L*A)/math.sqrt(n*q)))
            cross = (d-1)*np.sum((A@L.T)**2)/(d*n*q)
            energies.append(F/q+2*derivs[k]/(n*q)+cross)
        return np.array(means), np.array(energies), tail/q


def sampled_second(L, n, r, alpha, method, e, q, rng):
    d = len(L)
    g0, ge = rng.normal(size=(2,d,d))
    g1 = r*g0+math.sqrt(1-r*r)*ge
    S = L.T@L
    z0, z1 = g0@L, g1@L
    U = z0+alpha*scalar_response(z1-z0@S,method,e*math.sqrt(2*(1-r)))
    USroot = U@L.T
    base = np.sum(USroot*USroot)
    out = []
    for gk, zk in ((g0,z0),(g1,z1)):
        energytrace = ((n-d)*base+np.sum((gk.T@USroot)**2))/n
        B = zk.T@U
        out.append((np.trace(B)**2+np.sum(B*B.T)+n*energytrace)/(d*d*(n+2)*q))
    return np.array(out)


def coefficient_table():
    # Recompute amplitudes from the published matching equation, not from any
    # project output. Compare the printed B_R table separately from m and w.
    expected = {
        'GD': [-1.824,-1.220,-2.350,-.166,-3.274,.023],
        'Adam2': [-1.772,-1.455,-1.805,-.930,-1.756,-1.487],
        'Adam1': [-1.752,-1.544,-1.605,-1.244,-1.069,-2.090],
        'Adam.1': [-1.677,-1.735,-.925,-1.964,.051,-2.908],
        'Sign': [-1.611,-1.764,-.335,-2.444,.047,-2.805]}
    rows=[]
    for label, printed in expected.items():
        method = label if label in ('GD','Sign') else 'Adam'
        e = 1. if method != 'Adam' else float(label[4:])
        vals=[]
        for r in (.5,0.,-.5):
            beta,Q,_,_ = moments(method,r,e)
            sigma=math.sqrt(2*(1-r))
            c=sigma*beta/math.sqrt(Q)
            k=sigma*c/2
            B=(1+r)*k/math.sqrt(1-k*k)
            M2=r*r+B*B
            c0=(r*.68+B*math.sqrt(M2-.68**2))/M2
            theta=math.acos(c0)
            t=math.sin(theta)/(math.sqrt(1-k*k)*math.cos(theta)+k*math.sin(theta))
            alpha=t/math.sqrt(Q)
            pred=predicted(method,r,alpha,e,.25)
            mu,m,w=map(np.array,(pred['mu'],pred['m'],pred['w']))
            x=ndtri(.9)-mu
            br=np.exp(-x*x/2)/math.sqrt(2*math.pi)*(m-mu*w/2-mu/2-mu*mu*x/2)
            vals.extend(br.tolist())
            rows.append(dict(method=label,r=r,alpha=alpha,**pred,BR=br.tolist()))
        err=float(np.max(np.abs(np.array(vals)-printed)))
        assert err <= .00050001, (label, vals, printed)
    return rows


def raw_checks(gi, rng, samples):
    # Raw n-column labels and fixed X. No Bartlett, scatter representation,
    # Hermite formula or conditional contraction is used to generate samples.
    d,n,r,alpha=5,41,-.35,.63
    X=rng.normal(size=(d,n))/math.sqrt(d)
    S=d/n*X@X.T
    L=np.linalg.cholesky(S).T
    rows=[]
    for method,e in (('GD',1.),('Sign',1.),('Adam',.1),('Adam',2.)):
        p=predicted(method,r,alpha,e,n/d**2)
        qm=p['q']
        exactmean,exactall,tail=gi.integrate(L,n,r,alpha,method,e,qm)
        predcross=[]
        eye=np.eye(d)
        K=S@S@S-2*r*S@S+S
        beta=gi.response_data(np.diag(K),method,e*math.sqrt(2*(1-r)))[0]
        for r0,r1 in ((1,r),(r,1)):
            A=r0*L+alpha*(L@(r1*eye-r0*S))*beta
            predcross.append(np.sum((A@L.T)**2)/n)
        sums=np.zeros(8); squares=np.zeros(8)
        for start in range(0,samples,1000):
            b=min(1000,samples-start)
            y0=rng.normal(size=(b,2,n))/math.sqrt(d)
            ye=rng.normal(size=(b,2,n))/math.sqrt(d)
            y1=r*y0+math.sqrt(1-r*r)*ye
            z0=d/math.sqrt(n)*(y0@X.T)
            z1=d/math.sqrt(n)*(y1@X.T)
            U=z0+alpha*scalar_response(z1-z0@S,method,e*math.sqrt(2*(1-r)))
            R=np.einsum('aib,bc,aic->ai',U,S,U)/d
            cols=[]
            for y in (y0,y1):
                gamma=d/n*np.sum(y*y,axis=2)
                cross=d/n*np.sum(y[:,0]*y[:,1],axis=1)*np.einsum('ab,bc,ac->a',U[:,0],S,U[:,1])
                # Exact all-output-row sum expectation reconstructed from two
                # independent rows: diagonal and (d-1) off-diagonal pairs.
                allest=np.mean(gamma*R,axis=1)/qm+(d-1)/d*cross/qm
                cols.extend([np.mean(gamma*R,axis=1)/qm,cross,allest,np.mean(R,axis=1)/qm])
            a=np.array(cols).T
            sums+=np.sum(a,axis=0); squares+=np.sum(a*a,axis=0)
        mean=sums/samples
        se=np.sqrt(np.maximum(squares-sums*sums/samples,0)/(samples-1)/samples)
        for k in (0,1):
            rows.append(dict(method=method,e=e,task=k,samples=samples,
                             contraction_exact=float(predcross[k]),contraction_mc=float(mean[4*k+1]),contraction_se=float(se[4*k+1]),
                             all_energy_exact=float(exactall[k]),all_energy_mc=float(mean[4*k+2]),all_energy_se=float(se[4*k+2]),
                             diagonal_mc=float(mean[4*k]),row_energy_mc=float(mean[4*k+3]),quadrature_tail=tail))
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--reps',type=int,default=2000)
    ap.add_argument('--raw-samples',type=int,default=150000)
    ap.add_argument('--dims',default='16,24,32,48,64,96,128')
    args=ap.parse_args()
    out=OUTDIR/args.output
    assert out.parent == OUTDIR and not out.exists() and out.suffix=='.json'
    res=dict(resources=resources(),seed=1511062026,reps=args.reps,raw_samples=args.raw_samples)
    inputs=[ROOT/'Z1_G1_PROOF.md',ROOT/'paper/appendix/M_nonlinear_finite_d.tex']+sorted((ROOT/'z1_checks').glob('*'))
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs if p.is_file()}
    rng=np.random.default_rng(res['seed'])
    gi=GaussianIntegration()
    res['coefficient_table']=coefficient_table()
    res['raw_conditional_checks']=raw_checks(gi,rng,args.raw_samples)
    print('raw checks complete',flush=True)
    rows=[]
    dims=[int(x) for x in args.dims.split(',')]
    # Main parameters chosen independently of observed data. At r=-.5 the
    # Sign old-task 1/d mean coefficient must cancel exactly.
    settings=[(0.,'GD',1.),(0.,'Sign',1.),(0.,'Adam',.1),(0.,'Adam',1.),(0.,'Adam',2.),(-.5,'Sign',1.)]
    alpha,kappa=.6,.25
    for d in dims:
        tic=time.monotonic(); n=round(kappa*d*d)
        sums=np.zeros((len(settings),8)); sq=np.zeros_like(sums)
        max_tail=0.
        controls=[]
        for r,method,e in settings:
            pred=predicted(method,r,alpha,e,kappa)
            h=1e-4
            lo=gi.integrate(np.eye(d)*math.sqrt(1-h),n,r,alpha,method,e,pred['q'])
            hi=gi.integrate(np.eye(d)*math.sqrt(1+h),n,r,alpha,method,e,pred['q'])
            controls.append(((hi[0]-lo[0])/(2*h),(hi[1]-lo[1])/(2*h),pred))
        for rep in range(args.reps):
            bart=np.tril(rng.normal(size=(d,d)),-1)
            np.fill_diagonal(bart,np.sqrt(rng.chisquare(n-np.arange(d))))
            L=bart.T/math.sqrt(n)
            trace_deviation=np.sum(L*L)/d-1
            for j,(r,method,e) in enumerate(settings):
                cm,cv,pred=controls[j]
                m,v,tail=gi.integrate(L,n,r,alpha,method,e,pred['q'])
                second=sampled_second(L,n,r,alpha,method,e,pred['q'],rng)
                # Exact Gaussian density integration for all candidates; Haar
                # fourth moment sampled only for the O(1/n) deleted term.
                v=n/(n-1)*(v-cv*trace_deviation)-second/(n-1)
                m=m-cm*trace_deviation
                a=np.r_[m,v,second,tail,trace_deviation]
                sums[j]+=a; sq[j]+=a*a
                max_tail=max(max_tail,tail)
        for j,(r,method,e) in enumerate(settings):
            means=sums[j]/args.reps
            se=np.sqrt(np.maximum(sq[j]-sums[j]*sums[j]/args.reps,0)/(args.reps-1)/args.reps)
            pred=controls[j][2]
            for k in (0,1):
                for kind,idx,limit,coef in (('mean',k,pred['mu'][k],pred['m'][k]),('wrong_square',k+2,1.,pred['w'][k])):
                    rows.append(dict(d=d,n=n,r=r,method=method,e=e,alpha=alpha,kappa=kappa,task=k,kind=kind,
                                     mc=float(means[idx]),se=float(se[idx]),limit=limit,coefficient=coef,
                                     empirical_coefficient=float(d*(means[idx]-limit)),
                                     scaled_residual=float(d**1.5*(means[idx]-limit-coef/d)),
                                     scaled_residual_se=float(d**1.5*se[idx]),
                                     second_moment=float(means[4+k]),second_moment_se=float(se[4+k]),
                                     max_covariance_tail_bound=max_tail))
        print(json.dumps(dict(d=d,reps=args.reps,seconds=round(time.monotonic()-tic,2),tail=max_tail)),flush=True)
    res['monte_carlo']=rows
    res['input_hashes_before']=before
    res['input_hashes_after']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs if p.is_file()}
    res['inputs_unchanged']=res['input_hashes_before']==res['input_hashes_after']
    res['resources_end']=resources()
    with out.open('x') as f:
        json.dump(res,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(str(out.relative_to(ROOT)),flush=True)


if __name__=='__main__':
    main()
