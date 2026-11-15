"""Independent finite-dimensional trace and moment checks; float diagnostics."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key)=='1'
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'aa1_checks'
cg=Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
cp=Path('/sys/fs/cgroup')/cg.lstrip('/')
assert (cp/'memory.max').read_text().strip()=='3221225472'
assert (cp/'memory.swap.max').read_text().strip()=='0'
sources=['ICML_REVIEW_SIM6.md','paper/appendix/C.tex','paper/sections/main_results.tex',
         'paper/sections/setup.tex','Z2_RECALL_BRIDGE.md','Z2B_BR_PROOF.md']
hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}
rng=np.random.default_rng(202611060102)
results=[]
for rep in range(60):
    d=6; p=2*d; n=100000; m=n-1
    B=np.tril(rng.normal(size=(p,p)),-1)
    B[np.diag_indices(p)]=np.sqrt(rng.chisquare(n-np.arange(p)))
    C=B@B.T/n
    eigen,F=np.linalg.eigh(C)
    delta=float(np.max(np.abs(eigen-1)))
    assert delta<1/8
    F=F@np.diag(np.sqrt(eigen))
    Lx=F[:d]; Ly=F[d:]
    S=C[:d,:d]; Z=math.sqrt(n)*C[d:,:d]
    U=np.zeros((d,d))
    if rep%2==0:
        for _ in range(2): U-=.6*np.sign(U@S-Z)
        method='Sign2'
    else:
        mom=np.zeros_like(U); rms=np.zeros_like(U)
        for t in (1,2):
            residual=U@S-Z
            mom=.9*mom+.1*residual; rms=.99*rms+.01*residual**2
            U-=.6*(mom/(1-.9**t))/(np.sqrt(rms/(1-.99**t))+1)
        method='Adam2'
    norm=np.linalg.norm(U)
    A=Ly.T@U@Lx
    K=(A+A.T)/(2*norm); N=A.T@A/norm**2
    mu=float(np.trace(K)); kn=float(np.trace(K@K)); nn=float(np.trace(N))
    R=float(np.linalg.norm(U,2)/norm)
    assert abs(mu-np.sum(U*Z)/(math.sqrt(n)*norm))<1e-12
    assert 1-2*delta<=2*kn<=1+2*delta+2*delta**2
    assert (1-delta)**2<=nn<=(1+delta)**2
    assert np.linalg.norm(K,2)<=(1+delta)*R+1e-12
    assert np.linalg.norm(N,2)<=(1+delta)**2*R*R+1e-12
    trKN=float(np.trace(K@N)); trKKNN=float(np.trace(K@K@N))
    T20=mu*mu+2*kn
    T02=nn*nn+2*float(np.trace(N@N))
    T21=nn*T20+4*mu*trKN+8*trKKNN
    T40=(mu**4+12*mu*mu*kn+12*kn*kn+32*mu*float(np.trace(K@K@K))
         +48*float(np.trace(K@K@K@K)))
    ED2=n/(n+2)*T20
    ED4=n**3/((n+2)*(n+4)*(n+6))*T40
    v=n/m*nn
    EV=v-ED2/m
    EV2=n**4/m**2*(T02/(n*(n+2))-2*T21/(n*(n+2)*(n+4))
                          +T40/(n*(n+2)*(n+4)*(n+6)))
    centeredV=EV2-2*v*EV+v*v
    varVtilde=2*n**3/(m*m*(n+2))*(float(np.trace(N@N))-nn*nn/n)
    M=3*d/math.sqrt(n)
    D4=8*M**4+480
    assert ED2<=M*M+2
    assert ED4<=D4
    assert varVtilde<=13*R*R
    z=1.28
    eexact=z/(2*math.sqrt(v))*(ED2/m+centeredV/v)
    ebound=z*(2*(M*M+2)/n+52*R*R+16*D4/n**2)
    assert eexact<=ebound
    results.append(dict(method=method,delta=delta,R=R,mu=mu,
        ED2=ED2,ED4=ED4,variance_Vtilde=varVtilde,
        exact_remainder_budget=eexact,coarse_remainder_budget=ebound))
assert hashes=={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}
assert len(os.listdir('/proc/self/task'))<=2
with (OUT/(sys.argv[1] if len(sys.argv)>1 else 'AA1_bound_audit_replay.json')).open('x') as f:
    json.dump(dict(status='PASS',cases=len(results),results=results,
        input_hashes_unchanged=True,source_sha256=hashes,
        arithmetic='float64 independent trace identities; not a proof certificate'),f,indent=2)
print('PASS: 60 independent finite-dimensional trace/moment audits')
