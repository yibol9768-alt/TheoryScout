import numpy as np, sys, itertools, json
alpha, beta, sigma = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
b=16
def run(N,TPP,seed,etas,lams):
    rng=np.random.default_rng(seed)
    i=np.arange(1,N+1); lam_i=i**-alpha; th=np.sqrt(i**(-1-beta)/lam_i)
    tail=sum(k**(-1-beta) for k in range(N+1,200000))
    T=int(TPP*N/b)
    E,L=np.meshgrid(etas,lams,indexing='ij'); E=E.ravel()[:,None]; L=L.ravel()[:,None]
    G=E.shape[0]; w=np.zeros((G,N)); m=np.zeros((G,N)); v=np.zeros((G,N)); b1,b2=0.9,0.95
    for t in range(1,T+1):
        x=rng.standard_normal((b,N))*np.sqrt(lam_i); y=x@th+sigma*rng.standard_normal(b)
        r=w@x.T - y  # G,b
        g=r@x/b
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g
        u=(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+1e-12)
        w=(1-E*L)*w - E*u
    risk=((w-th)**2*lam_i).sum(1)+tail
    return T,E.ravel(),L.ravel(),risk
etas=np.geomspace(3e-4,1e-1,12); lams=np.concatenate([[0],np.geomspace(1e-3,30,16)])
out=[]
for N in [32,64,128]:
  for TPP in [20,80,320]:
    rs=[]; 
    for s in range(2):
        T,E,L,r=run(N,TPP,s,etas,lams); rs.append(r)
    r=np.mean(rs,0); r[(E*L>0.5)|~np.isfinite(r)]=np.inf; k=np.argmin(r)
    tau_ep = 1/(E[k]*L[k]*T) if L[k]>0 else float('inf')
    r0=r[L==0].min(); best_eta_noWD=E[L==0][np.argmin(r[L==0])]
    out.append(dict(N=N,TPP=TPP,T=T,eta=E[k],lam=L[k],tau_epoch=tau_ep,risk=r[k],risk_noWD=r0,eta_noWD=best_eta_noWD))
    print(out[-1],flush=True)
