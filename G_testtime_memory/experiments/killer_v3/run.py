"""New-seed paired data; primary predictions never use U or value ranks."""
import argparse
import itertools
import json
import time
import numpy as np
from numpy.polynomial.hermite import hermgauss
from design import *
from common import OUT,begin,finish,prior_elapsed,mean_se

def coefficients(K,ids):
    n,d=K.shape
    X=np.broadcast_to(K[ids],(len(BETAS),len(ids),d)).copy()
    U=np.empty((len(BETAS),n,len(ids)))
    b=np.array(BETAS)[:,None]
    for j in range(n-1,-1,-1):
        v=(X.reshape(-1,d)@K[j]).reshape(len(BETAS),len(ids))*b
        U[:,j,:]=v
        X-=v[:,:,None]*K[j]
    return np.concatenate(((K@K[ids].T)[None,:,:],U))

def binned(v,ids,n):
    return [float(v[bins(n,ids)==b].mean()) for b in range(4)]

def moment_metrics(U,ids,n,cond):
    m=U[ids,np.arange(len(ids))]; Q=np.sum(U*U,axis=0)
    pm,pq,vm=cond
    wrong=U.copy(); wrong[ids,np.arange(len(ids))]=0
    return {'m_bins':binned(m,ids,n),'Q_bins':binned(Q,ids,n),
        'm_cond_bins':binned(pm,ids,n),'Q_cond_bins':binned(pq,ids,n),
        'm_variance_pred_bins':binned(vm,ids,n),
        'm_squared_residual_bins':binned((m-pm)**2,ids,n),
        'Q_squared_residual_bins':binned((Q-pq)**2,ids,n),
        'rho_sd_bins':binned((m/np.sqrt(Q)-pm/np.sqrt(pq))**2,ids,n),
        'max_wrong_scaled_bins':binned(np.max(np.abs(wrong),axis=0)/np.sqrt(Q)*np.sqrt(np.log(n)),ids,n)}

def readout_metrics(U,V,ids,n):
    score=V@(V.T@U)
    good=score[ids,np.arange(len(ids))]
    rank=(score>good[None,:]).sum(axis=0)
    ok=rank==0
    return {'top1':float(ok.mean()),'top10':float((rank<int(np.ceil(.1*n))).mean()),
        'mean_rank':float((rank/(n-1)).mean()),'age_top1':binned(ok,ids,n)}

def lifted_smoke():
    """Independent 5-node Gaussian integration checks up to eighth moments.

    d=2,n=4,query i=1 (zero-based). Exhaustive 25^3 key paths and an
    independent lifted second-moment recurrence; no stochastic fit/test.
    """
    d,load,a,family,b=2,2,1.5,'bounded',.4
    ids=np.array([1]); ker=Kernel(d,load,a,family,'delta_0.4',ids)
    delta=ker.delta
    nodes,weight=hermgauss(5)
    nodes=nodes*np.sqrt(2); weight=weight/np.sqrt(np.pi)
    index=np.array(list(itertools.product(range(5),repeat=2)))
    K=nodes[index]*np.sqrt(delta); weights=np.prod(weight[index],axis=1)
    z=np.array([.7,-.4]); cond=ker.conditional(z[None,:])
    e=np.zeros(4); W=np.zeros((4,4))
    for step in range(4):
        ks,ww=([z],[1.]) if step==1 else (K,weights)
        en=np.zeros(4); Wn=np.zeros((4,4))
        for k,wgt in zip(ks,ww):
            A=np.eye(2)-b*np.outer(k,k); L=np.kron(A,A); v=(b*b*np.outer(k,k)).ravel()
            le=L@e
            en+=wgt*(le+v)
            Wn+=wgt*(L@W@L.T+np.outer(le,v)+np.outer(v,le)+np.outer(v,v))
        e,W=en,Wn
    paths=np.array(list(itertools.product(range(25),repeat=3)))
    pw=np.prod(weights[paths],axis=1); H=np.zeros((len(paths),2,2)); c=np.zeros((len(paths),2))
    ptr=0
    for step in range(4):
        k=np.broadcast_to(z,(len(paths),2)) if step==1 else K[paths[:,ptr]]
        if step!=1: ptr+=1
        A=np.eye(2)[None,:,:]-b*k[:,:,None]*k[:,None,:]
        H=A@H@A+b*b*k[:,:,None]*k[:,None,:]
        c=np.einsum('qij,qj->qi',A,c)
        if step==1: c+=b*k
    qs=np.einsum('i,qij,j->q',z,H,z); ms=c@z
    az=np.outer(z,z).ravel()
    np.testing.assert_allclose([pw@ms,pw@qs,pw@(ms*ms)-(pw@ms)**2],
        [cond[0][0],cond[1][0],cond[2][0]],atol=1e-11)
    np.testing.assert_allclose(az@e,pw@qs,atol=1e-11)
    np.testing.assert_allclose(az@(W-np.outer(e,e))@az,pw@(qs*qs)-(pw@qs)**2,atol=1e-11)
    # Independently integrate the query to check the degree-six Wick formula.
    full=Kernel(d,load,a,family,'delta_0.4')
    sm=np.zeros(4); sq=np.zeros(4)
    for k,wgt in zip(K,weights):
        mm,qq,_=full.conditional(np.broadcast_to(k,(4,2)))
        sm+=wgt*mm; sq+=wgt*qq
    um,uq=full.unconditional()
    np.testing.assert_allclose(sm,um,atol=1e-11)
    np.testing.assert_allclose(sq,uq,atol=1e-11)
    return {'conditional_Q_variance_lifted':float(az@(W-np.outer(e,e))@az),
            'conditional_Q_variance_exhaustive':float(pw@(qs*qs)-(pw@qs)**2)}

def smoke():
    for family in FAMILIES:
        for a in (0.,1.5):
            d,load=32,2; K=key_array(SMOKE_SEED,d,load,a,family); n=len(K); ids=query_ids(n)
            V=seed_rng(SMOKE_SEED,d,load,a,family,1).standard_normal((n,19))/np.sqrt(19)
            us=coefficients(K,ids)
            for method,U in zip(METHODS,us):
                if method=='hebb': C=K.copy(); S=V.T@K
                else:
                    b=float(method.split('_')[1]); C=np.zeros_like(K); S=np.zeros((19,d))
                    for j,k in enumerate(K):
                        C-=b*(C@k)[:,None]*k[None,:]; C[j]+=b*k
                        S+=b*np.outer(V[j]-S@k,k)
                np.testing.assert_allclose(U,C@K[ids].T,atol=1e-10,rtol=1e-9)
                np.testing.assert_allclose(V.T@U,S@K[ids].T,atol=1e-10,rtol=1e-9)
                ker=Kernel(d,load,a,family,method)
                mm,qq,vv=ker.conditional(K[ids]); um,uq=ker.unconditional()
                assert np.isfinite(qq).all() and np.all(qq>0) and np.all(vv>=0)
                if method!='hebb':
                    b=ker.beta; D=np.diag(ker.delta); j=21; z=K[j]
                    H=np.diag(ker.rt[j]); Az=np.eye(d)-b*np.outer(z,z)
                    H=Az@H@Az+b*b*np.outer(z,z)
                    for _ in range(n-1-j):
                        H=H-b*(D@H+H@D)+b*b*(2*D@H@D+D*np.trace(D@H))+b*b*D
                    np.testing.assert_allclose(z@H@z,qq[j],atol=2e-10,rtol=1e-9)
    # Algebra of the alternative correlated representation, covariance only.
    w=np.array([.4,.3,np.sqrt(.75)]); rho=w[0]; r=np.sqrt(1-rho*rho); aa=w[1:]; bb=aa/r
    cov=np.zeros((3,3)); cov[0,0]=r*r; cov[0,1:]=-rho*aa; cov[1:,0]=-rho*aa
    cov[1:,1:]=np.eye(2)-np.outer(bb,bb)+rho*rho*np.outer(aa,aa)/(r*r)
    np.testing.assert_allclose(cov,np.eye(3)-np.outer(w,w),atol=1e-14)
    assert np.isclose(mean_se([0.,1.])[1],.5)
    lifted=lifted_smoke()
    (OUT/'smoke_checks.json').write_text(json.dumps(lifted,indent=2))
    print('SMOKE PASS: d=32 products/state, finite conditional Wick recurrence; d=2 exhaustive vs lifted Q variance, unconditional degree-six formula; correlated covariance',flush=True)

def oracle(us,ids,n,d,load,a,family,seed,readout):
    chosen={m:us[METHODS.index(m)] for m in ORACLE_METHODS}
    ww={m:U/np.linalg.norm(U,axis=0) for m,U in chosen.items()}
    totals={(m,r):0 for m in ORACLE_METHODS for r in READOUTS}
    rng=seed_rng(seed,d,load,a,family,3)
    for start in range(0,len(ids),8):
        stop=min(start+8,len(ids)); size=stop-start
        G=rng.standard_normal((size,ORACLE_DRAWS,n))
        for r in READOUTS:
            p=value_dim(n,d,r); R=np.sqrt(rng.chisquare(p,(size,ORACLE_DRAWS)))
            for m in ORACLE_METHODS:
                W=ww[m][:,start:stop].T
                h=np.einsum('qbn,qn->qb',G,W,optimize=True)
                T=G+W[:,None,:]*(R-h)[:,:,None]
                good=T[np.arange(size)[:,None],np.arange(ORACLE_DRAWS)[None,:],ids[start:stop,None]]
                totals[m,r]+=int(np.sum(np.max(T,axis=2)<=good))
    rows=[]
    for m,U in chosen.items():
        rho=ww[m][ids,np.arange(len(ids))]
        for r in READOUTS:
            p=value_dim(n,d,r)
            bound=loose_readout_bound(U,ids,p)
            rows.append({'method':m,'readout':r,'oracle_w':totals[m,r]/(len(ids)*ORACLE_DRAWS),
                'oracle_rho':float(readout(rho,n,p).mean()),
                'coupling_bound_mean':float(bound.mean()),'coupling_bound_max':float(bound.max()),
                'coupling_nonvacuous_fraction':float(np.mean(bound<1.)),
                'MC_samples':len(ids)*ORACLE_DRAWS})
    np.savez(OUT/f'oracle_U_d{d}_seed{seed}.npz',ids=ids,**chosen)
    return rows

def formal(state):
    assert json.loads((OUT/'prepare_meta.json').read_text())['complete']
    assert json.loads((OUT/'smoke_meta.json').read_text())['complete']
    readout=Readout(OUT/'readout_tables.npz'); count=0; oracle_count=0
    previous=prior_elapsed('formal')
    for name in ('raw.jsonl','moments.jsonl','oracle.jsonl'): assert not (OUT/name).exists()
    with (OUT/'raw.jsonl').open('x',buffering=1) as f, (OUT/'moments.jsonl').open('x',buffering=1) as fm, (OUT/'oracle.jsonl').open('x',buffering=1) as fo:
        for family in FAMILIES:
            for d in DIMS:
                for a in ALPHAS:
                    for load in LOADS:
                        if time.monotonic()-state[0]+previous>3150:
                            finish(state,False,sequences=count,reason='cumulative 3150s cutoff'); return
                        n=d*load; ids=query_ids(n)
                        ks=[key_array(seed,d,load,a,family) for seed in TEST_SEEDS]
                        pred={}
                        # Only queries enter the predictor; all other key coordinates
                        # are integrated out, and no values/U/ranks have been computed.
                        for method in METHODS:
                            ker=Kernel(d,load,a,family,method)
                            for seed,K in zip(TEST_SEEDS,ks): pred[seed,method]=ker.conditional(K[ids])
                            del ker
                        for seed,K in zip(TEST_SEEDS,ks):
                            if time.monotonic()-state[0]+previous>3150:
                                finish(state,False,sequences=count,reason='cumulative 3150s cutoff'); return
                            us=coefficients(K,ids); assert np.isfinite(us).all()
                            pmax=max(value_dim(n,d,r) for r in READOUTS)
                            Z=seed_rng(seed,d,load,a,family,1).standard_normal((n,pmax))
                            base={'family':family,'d':d,'alpha':a,'load':load,'seed':seed,'queries':len(ids)}
                            actual={}
                            for method,U in zip(METHODS,us):
                                cond=pred[seed,method]; pm,pq,_=cond
                                fm.write(json.dumps({**base,'method':method,**moment_metrics(U,ids,n,cond)})+'\n')
                                for r in READOUTS:
                                    p=value_dim(n,d,r); V=Z[:,:p]/np.sqrt(p)
                                    metrics=readout_metrics(U,V,ids,n)
                                    prediction=readout(pm/np.sqrt(pq),n,p)
                                    actual[method,r]=metrics['top1']
                                    f.write(json.dumps({**base,'method':method,'readout':r,'p':p,
                                        'predicted':float(prediction.mean()),'predicted_bins':binned(prediction,ids,n),**metrics})+'\n')
                            if oracle_cell(family,d,load,a):
                                for row in oracle(us,ids,n,d,load,a,family,seed,readout):
                                    row['observed']=actual[row['method'],row['readout']]
                                    fo.write(json.dumps({**base,**row})+'\n'); oracle_count+=1
                            count+=1
                        del ks,pred
                        print(f'formal {family} d={d} a={a} load={load} sequences={count} elapsed={time.monotonic()-state[0]:.1f}s',flush=True)
    assert count==864 and oracle_count==384
    finish(state,sequences=count,expected_sequences=864,oracle_records=oracle_count)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=('smoke','formal')); args=parser.parse_args()
    state=begin(args.mode)
    if args.mode=='smoke': smoke(); finish(state)
    else: formal(state)

if __name__=='__main__': main()
