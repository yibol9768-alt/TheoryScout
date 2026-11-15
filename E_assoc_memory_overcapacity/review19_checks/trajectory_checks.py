"""Additional adversarial trajectory and deterministic matrix checks.
No audited code is imported. Small synthetic covariance perturbations isolate
the actual nonlinear updates from asymptotic assertions.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ[name]=='1'
import json
import math
from pathlib import Path
import numpy as np

out=Path(__file__).resolve().parent
rng=np.random.default_rng(202611061920)
cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':')[-1].lstrip('/')
assert (cg/'memory.max').read_text().strip()=='3221225472'
assert (cg/'memory.swap.max').read_text().strip()=='0'

sign_ratios=[]
adam_ratios=[]
matrix_slack=[]
for case in range(120):
    d=12
    orthogonal,_=np.linalg.qr(rng.normal(size=(d,d)))
    vals=1+rng.uniform(-.12,.12,d)
    S=(orthogonal*vals)@orthogonal.T
    Sroot=(orthogonal*np.sqrt(vals))@orthogonal.T
    delta=float(np.max(np.abs(vals-1)))
    G=rng.normal(size=(d,d)); Z=G@Sroot
    assert np.linalg.norm(G)<=2*d
    T=[1,2,3,5,8][case%5]
    alpha=[.01,.6,1.5,4.][(case//5)%4]
    A=2*T*T/math.sqrt(2*math.pi)
    B=sum((j*alpha+2)**2 for j in range(T))
    tau=(2*B*delta*delta/A)**(1/3)
    U=np.zeros((d,d)); H=np.zeros((d,d)); counts=0.
    for j in range(T):
        counts+=np.mean(np.abs(H-G)<=tau)
        U-=alpha*np.sign(U@S-Z)
        H-=alpha*np.sign(H-G)
    lhs=float(np.sum((U-H)**2)/(d*d))
    rhs=4*T*T*alpha*alpha*(counts+B*delta*delta/tau**2)
    assert lhs<=rhs+1e-10
    sign_ratios.append(lhs/rhs)

    beta1=[0.,.9,.999][case%3]
    beta2=[0.,.01,.99,.999999][(case//3)%4]
    e=[.03,1.,10.][(case//12)%3]
    lam=1+2*alpha/e; c0=math.sqrt(1-beta2)
    history_bound=[0.]; endpoint_bound=[0.]
    for j in range(1,T+1):
        previous=e*(lam**(j-1)-1)/(2*c0)
        history_bound.append(previous)
        L=(1+lam**(j-1)/c0)/e
        endpoint_bound.append(endpoint_bound[-1]+alpha*L*math.sqrt(sum(
            (2*endpoint_bound[k]+history_bound[k+1]+2)**2 for k in range(j))))
    U=np.zeros((d,d)); H=np.zeros_like(U)
    mu=np.zeros_like(U); mh=np.zeros_like(U)
    vu=np.zeros_like(U); vh=np.zeros_like(U)
    for j in range(1,T+1):
        ru=U@S-Z; rh=H-G
        mu=beta1*mu+(1-beta1)*ru; mh=beta1*mh+(1-beta1)*rh
        vu=beta2*vu+(1-beta2)*ru*ru; vh=beta2*vh+(1-beta2)*rh*rh
        U-=alpha*(mu/(1-beta1**j))/(np.sqrt(vu/(1-beta2**j))+e)
        H-=alpha*(mh/(1-beta1**j))/(np.sqrt(vh/(1-beta2**j))+e)
    actual=float(np.linalg.norm(U-H)/d)
    upper=endpoint_bound[-1]*delta
    assert actual<=upper+1e-9
    assert np.max(np.abs(H))<=e*(lam**T-1)/(2*c0)+1e-9
    adam_ratios.append(actual/upper)

    # Arbitrary full scatter near I and adaptive actual U; checks (27).
    width=2*d
    O,_=np.linalg.qr(rng.normal(size=(width,width)))
    eigen=1+rng.uniform(-.125,.125,width)
    F=O*np.sqrt(eigen)
    C=F@F.T
    de=float(np.max(np.abs(eigen-1)))
    Lx,Ly=F[:d],F[d:]
    n=100000
    Sc=C[:d,:d]; Zc=math.sqrt(n)*C[d:,:d]
    adaptive=np.zeros((d,d))
    for j in range(T):
        adaptive-=alpha*np.sign(adaptive@Sc-Zc)
    norm=float(np.linalg.norm(adaptive)); R=float(np.linalg.norm(adaptive,2)/norm)
    A=Ly.T@adaptive@Lx
    K=(A+A.T)/(2*norm); N=A.T@A/(norm*norm)
    var=2*float(np.sum(K*K)); trace=float(np.trace(N))
    assert 1-2*de-1e-12<=var<=1+2*de+2*de*de+1e-12
    assert (1-de)**2-1e-12<=trace<=(1+de)**2+1e-12
    assert np.linalg.norm(K,2)<=(1+de)*R+1e-12
    assert np.linalg.norm(N,2)<=(1+de)**2*R*R+1e-12
    matrix_slack.append(dict(variance=var, trace=trace, delta=de))

result=dict(status='PASS', cases=120,
    sign_max_ratio=float(max(sign_ratios)), adam_max_ratio=float(max(adam_ratios)),
    T=[1,2,3,5,8], alpha=[.01,.6,1.5,4.], e=[.03,1.,10.],
    beta1=[0.,.9,.999], beta2=[0.,.01,.99,.999999],
    interpretation='finite diagnostics only; (34) is assessed analytically, not certified by these cases',
    threads=len(list(Path('/proc/self/task').iterdir())),matrix_checks=len(matrix_slack))
assert result['threads']<=2
with (out/'trajectory_results.json').open('x') as f:
    json.dump(result,f,indent=2)
print(json.dumps(result))
