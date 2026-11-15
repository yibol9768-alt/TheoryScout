"""Exact key coefficients, paired readouts, complete seed-level raw data."""
import argparse
import json
import time
import numpy as np
from design import *
from common import OUT, begin, finish, prior_elapsed, mean_se

def coefficients(K, ids):
    """Backward product: U[b,j,q]=beta*k_j^T A_{j+1}...A_n k_id[q]."""
    n,d = K.shape
    X = np.broadcast_to(K[ids], (len(BETAS),len(ids),d)).copy()
    U = np.empty((len(BETAS),n,len(ids)))
    b = np.array(BETAS)[:,None]
    for j in range(n-1,-1,-1):
        v = (X.reshape(-1,d) @ K[j]).reshape(len(BETAS),len(ids)) * b
        U[:,j,:] = v
        X -= v[:,:,None]*K[j]
    H = K @ K[ids].T
    return np.concatenate((H[None,:,:], U))

def moment_metrics(U, ids, n):
    m = U[ids,np.arange(len(ids))]
    Q = np.einsum('jq,jq->q',U,U)
    wrong = U.copy()
    wrong[ids,np.arange(len(ids))] = 0
    max_wrong = np.max(np.abs(wrong),axis=0)/np.sqrt(Q)*np.sqrt(np.log(n))
    masks = [bins(n,ids)==b for b in range(4)]
    return {'m_bins':[float(m[x].mean()) for x in masks],
            'Q_bins':[float(Q[x].mean()) for x in masks],
            'max_wrong_scaled_bins':[float(max_wrong[x].mean()) for x in masks]}

def readout_metrics(U,V,ids,n):
    score = V @ (V.T @ U)
    good = score[ids,np.arange(len(ids))]
    rank = (score > good[None,:]).sum(axis=0)
    assert np.all((rank>=0)&(rank<n))
    ok = rank==0
    masks = [bins(n,ids)==b for b in range(4)]
    return {'top1':float(ok.mean()),'top10':float((rank < int(np.ceil(.1*n))).mean()),
            'mean_rank':float((rank/(n-1)).mean()),
            'age_top1':[float(ok[x].mean()) for x in masks]}

def keys(seed,d,load,a,family):
    return seed_rng(seed,d,load,a,family,0).standard_normal((d*load,d))*np.sqrt(spectrum(d,a,family)/d)

def smoke():
    # Only d=32 engineering data. No tests on agreement with the conjecture.
    for family in FAMILIES:
        for a in (0.,1.5):
            d,load = 32,2
            K = keys(SMOKE_SEED,d,load,a,family)
            n = len(K)
            ids = query_ids(n)
            V = seed_rng(SMOKE_SEED,d,load,a,family,1).standard_normal((n,19))/np.sqrt(19)
            us = coefficients(K,ids)
            for method,U in zip(METHODS,us):
                if method == 'hebb':
                    C = K.copy()
                    S = V.T@K
                else:
                    beta = float(method.split('_')[1])
                    C = np.zeros_like(K)
                    S = np.zeros((19,d))
                    for j,k in enumerate(K):
                        C -= beta*(C@k)[:,None]*k[None,:]
                        C[j] += beta*k
                        S += beta*np.outer(V[j]-S@k,k)
                np.testing.assert_allclose(U,C@K[ids].T,atol=1e-10,rtol=1e-9)
                np.testing.assert_allclose(V.T@U,S@K[ids].T,atol=1e-10,rtol=1e-9)
                np.testing.assert_allclose(np.sum(U*U,axis=0),
                    np.einsum('qi,ij,qj->q',K[ids],C.T@C,K[ids]),atol=1e-9)
                metrics = readout_metrics(U,V,ids,n)
                assert 0<=metrics['top1']<=1
                m,Q,_ = feedback(d,load,a,family,method)
                ap = recall(m,Q,n,19,96)
                assert np.all((ap>=0)&(ap<=1+1e-12))
            # Independent finite-step moment recursion agrees with direct
            # Gaussian Wick operator on an arbitrary symmetric matrix.
            lam = spectrum(d,a,family)
            D = np.diag(lam/d)
            H = np.diag(np.arange(1,d+1)/d)
            beta = .4
            exact = H-beta*(D@H+H@D)+beta**2*(2*D@H@D+D*np.trace(D@H))
            h = np.diag(H)
            diag = (1-2*beta*lam/d+2*beta**2*lam**2/d**2)*h + beta**2*lam/d**2*(lam@h)
            np.testing.assert_allclose(np.diag(exact),diag)
            # ODE eigensolver vs an augmented exact exponential.
            from scipy.linalg import expm
            M = np.diag(-2*beta*lam)+beta**2*np.outer(lam,lam)/d
            A = np.zeros((d+1,d+1)); A[:d,:d]=M; A[:d,d]=beta**2*lam
            q = expm(A*.7)[:d,d]
            w,R = np.linalg.eigh(M)
            alt = R @ ((np.expm1(.7*w)/w)*(R.T@(beta**2*lam)))
            np.testing.assert_allclose(q,alt,atol=1e-11)
            tiny = 1e-8
            S = np.zeros((19,d))
            for k,v in zip(K,V): S += tiny*np.outer(v-S@k,k)
            assert np.linalg.norm(S/tiny-V.T@K)/np.linalg.norm(V.T@K)<1e-5
    assert np.isclose(mean_se([0.,1.])[1],.5)
    print('SMOKE PASS: d=32 products, direct state, quadratic energies, Wick, ODE, ranks, small beta, seed SE',flush=True)

def formal(state):
    assert json.loads((OUT/'smoke_meta.json').read_text())['complete']
    assert json.loads((OUT/'prepare_meta.json').read_text())['complete']
    assert not (OUT/'raw.jsonl').exists() and not (OUT/'moments.jsonl').exists()
    previous = prior_elapsed('formal')
    count = 0
    with (OUT/'raw.jsonl').open('x',buffering=1) as f, (OUT/'moments.jsonl').open('x',buffering=1) as fm:
        tasks = [('calibration',64,CAL_SEEDS)]+[('validation',d,TEST_SEEDS) for d in DIMS]
        for split,d,seeds in tasks:
            for family in FAMILIES:
                for a in ALPHAS:
                    for load in LOADS:
                        n = d*load
                        ids = query_ids(n)
                        for seed in seeds:
                            if time.monotonic()-state[0]+previous>3100:
                                finish(state,False,sequences=count,reason='cumulative 3100s cutoff')
                                return
                            K = keys(seed,d,load,a,family)
                            us = coefficients(K,ids)
                            assert np.isfinite(us).all(),(split,d,family,a,load,seed)
                            pmax = max(value_dim(n,d,r) for r in READOUTS)
                            Z = seed_rng(seed,d,load,a,family,1).standard_normal((n,pmax))
                            lam = spectrum(d,a,family)
                            diag = {'effective_dim':float(d*d/(lam@lam)),
                                    'key_norm_mean':float(np.sum(K*K,axis=1).mean()),
                                    'key_norm_var':float(np.sum(K*K,axis=1).var()),
                                    'max_key_norm':float(np.sum(K*K,axis=1).max())}
                            base = {'split':split,'family':family,'d':d,'alpha':a,
                                    'load':load,'seed':seed,'queries':len(ids),**diag}
                            for method,U in zip(METHODS,us):
                                fm.write(json.dumps({**base,'method':method,**moment_metrics(U,ids,n)})+'\n')
                                for readout in READOUTS:
                                    p = value_dim(n,d,readout)
                                    V = Z[:,:p]/np.sqrt(p)
                                    row = {**base,'method':method,'readout':readout,'p':p,
                                           **readout_metrics(U,V,ids,n)}
                                    f.write(json.dumps(row)+'\n')
                            count += 1
                        print(f'{split} {family} d={d} a={a} load={load} sequences={count} elapsed={time.monotonic()-state[0]:.1f}s',flush=True)
    finish(state,sequences=count,expected_sequences=1008)
    assert count == 1008

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('smoke','formal'))
    args = parser.parse_args()
    state = begin(args.mode)
    if args.mode == 'smoke': smoke(); finish(state)
    else: formal(state)

if __name__ == '__main__': main()
