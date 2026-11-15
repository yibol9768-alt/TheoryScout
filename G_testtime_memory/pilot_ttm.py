import numpy as np
rng=np.random.default_rng(0); d=64
def recall(S,K,V):
    sc=(S@K.T).T @ V.T   # n x n: score of candidate j for query i
    corr=sc[np.arange(len(K)),np.arange(len(K))]
    rank=(sc>corr[:,None]).sum(1)
    return (rank==0)
for ratio in [0.5,1,2,4,8,16,32]:
    n=int(ratio*d); K=rng.standard_normal((n,d))/np.sqrt(d); V=rng.standard_normal((n,d))/np.sqrt(d)
    out={}
    S=V.T@K; out['hebb']=recall(S,K,V)
    for beta in [0.05,0.2,0.5,1.0]:
        S=np.zeros((d,d))
        for t in range(n): S+=beta*np.outer(V[t]-S@K[t],K[t])
        out[f'delta{beta}']=recall(S,K,V)
    for a in [0.99,0.95]:
        S=np.zeros((d,d))
        for t in range(n): S=a*S+np.outer(V[t],K[t])
        out[f'gate{a}']=recall(S,K,V)
    # sign-update (elementwise sign of delta gradient), small step
    S=np.zeros((d,d))
    for t in range(n): S+=0.02/d*np.sign(np.outer(V[t]-S@K[t],K[t]))
    out['sign']=recall(S,K,V)
    # LS optimum (batch)
    S=np.linalg.lstsq(K,V,rcond=None)[0].T; out['LS']=recall(S,K,V)
    print(f'n/d={ratio:5}', ' '.join(f'{k}:{v.mean():.2f}(old {v[:n//4].mean():.2f})' for k,v in out.items()),flush=True)
