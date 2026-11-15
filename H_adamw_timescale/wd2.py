# fixed eta (muP-like), sweep only lambda; record optimal tau_epoch=1/(eta*lam*T)
import numpy as np, sys
exec(open('wd.py').read().split('etas=')[0])
eta=float(sys.argv[4])
lams=np.geomspace(1e-3,0.5/eta,24)
res=[]
for N in [32,64,128]:
  for TPP in [10,40,160,640]:
    rs=[run(N,TPP,s,np.array([eta]),lams) for s in range(3)]
    T=rs[0][0]; L=rs[0][2]; r=np.mean([x[3] for x in rs],0); r[~np.isfinite(r)]=np.inf
    k=np.argmin(r); res.append((N,TPP,1/(eta*L[k]*T)))
    print(N,TPP,T,'lam*=%.4g tau_ep*=%.4g risk=%.5g  edge=%s'%(L[k],1/(eta*L[k]*T),r[k],k in (0,len(L)-1)),flush=True)
res=np.array(res); A=np.c_[np.log(res[:,1]),np.ones(len(res))]
print('slope log tau_ep vs log TPP:',np.linalg.lstsq(A,np.log(res[:,2]),rcond=None)[0][0])
