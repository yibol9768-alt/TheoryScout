"""EXP1: CPU-only independent calibration and frozen retrieval experiments."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import math, time, resource, json
import numpy as np
from scipy.optimize import linprog
from e0_checks import budget_dp
K=4

def rng(seed,g,stream): return np.random.default_rng(np.random.SeedSequence([seed,g,stream]))
def norm(x): return (x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)).astype('float32')
def entropy(p): return float(-np.sum(p[p>0]*np.log2(p[p>0])))
def budgets(g): return sorted({g//2,g,g+1,g+3,5*g//2,4*g,*range(g,g+5)})
def probability(g,dist):
    p=np.ones(g) if dist==0 else np.arange(1,g+1,dtype=float)**(-dist)
    return p/p.sum()
def make_keys(seed,g,tau):
    r=rng(seed,g,30); u=r.normal(size=(g,32)); d=r.normal(size=(g,K,32))
    raw=u[:,None,:]+tau*d
    return u,raw,np.concatenate([norm(u),norm(raw).reshape(-1,32)])
def query_scores(r,u,raw,keys,i,j,sigma): return norm(raw[i,j]+sigma*r.normal(size=(len(i),32)))@keys.T

def coefficients(lam,ws,vs,vr):
    return np.array([0,1,K]),np.array([0,ws,K]),np.array([0,vs*(lam+ws),vr*(lam+1)])
def value(p,z,lam,ws,vs,vr):
    c,w,u=coefficients(lam,ws,vs,vr)
    return float(p@u[z]/(lam+w[z].sum()))
def fractional(p,b,lam,ws,vs,vr):
    c,w,u=coefficients(lam,ws,vs,vr); g=len(p)
    ae=np.zeros((g,3*g)); ae[np.arange(g)[:,None],3*np.arange(g)[:,None]+np.arange(3)]=1
    cost=np.tile(c,g); load=np.tile(w,g); num=(p[:,None]*u).ravel(); t=0.
    for _ in range(30):
        sol=linprog(-(num-t*load),A_ub=cost[None,:],b_ub=[b],A_eq=ae,b_eq=np.ones(g),bounds=(0,None),method='highs',options={'threads':1})
        if not sol.success: raise RuntimeError(sol.message)
        x=sol.x; new=float(num@x/(lam+load@x))
        if abs(new-t)<1e-12: break
        t=new
    x=x.reshape(g,3); z=np.array([np.flatnonzero(row>1e-8)[0] for row in x]); mixed=int(((x>1e-8).sum(1)>1).sum())
    # A solver optimal basic solution must obey the one-fractional-group theorem.
    assert mixed<=1 and c[z].sum()<=b+1e-8
    if 0<u[1]<u[2] and 0<w[1]<w[2]:
        assert new-value(p,z,lam,ws,vs,vr)<=p.max()*u[2]/(lam+float((x@w).sum()))+1e-8
    return z

def make_policies(p,b,lam,ws,vs,vr,r):
    g=len(p); order=np.argsort(-p,kind='stable'); out={}
    def counts(n,s):
        z=np.zeros(g,dtype='int8'); z[order[:n]]=2; z[order[n:n+s]]=1; return z
    out['summary']=counts(0,min(g,b)); n=min(g,b//K)
    out['raw']=counts(n,0); out['raw_fill']=counts(n,min(g-n,b-K*n))
    z=np.zeros(g,dtype='int8'); remaining=b
    for idx in r.permutation(g):
        modes=np.array([0,1,2]); modes=modes[np.array([0,1,K])<=remaining]
        z[idx]=r.choice(modes); remaining-=int([0,1,K][z[idx]])
    out['random']=z
    c,w,u=coefficients(lam,ws,vs,vr)
    out['dp']=budget_dp(p,c,u,w,b,lam,tolerance=1e-11)[1]
    out['fractional_down']=fractional(p,b,lam,ws,vs,vr)
    return out

def calibration_index(seed,g,episode,field,target_type,m,raw_fraction,variant):
    target=episode if target_type==0 else g+K*episode+field
    if m==1: return np.array([target]),int(target_type==0),int(target_type==1)
    own=np.array([episode]) if target_type==0 else g+K*episode+np.arange(K)
    m=min(m,len(own)+K*(g-1)); remaining=m-len(own)
    candidates=[(n,remaining-K*n) for n in range(g) if 0<=remaining-K*n<=g-1-n]
    nraw,nsum=min(candidates,key=lambda pair:abs((K*pair[0]+(K if target_type else 0))/m-raw_fraction))
    sub=rng(seed,g,1000+episode*12+field*3+variant)
    order=sub.permutation(np.delete(np.arange(g),episode))
    rawids=g+(K*order[:nraw,None]+np.arange(K)).ravel()
    ids=np.r_[own,rawids,order[nraw:nraw+nsum]]
    ids=np.r_[target,ids[ids!=target]]
    assert len(ids)==m and len(np.unique(ids))==m
    summary=ids[ids<g]; rawgroups=np.unique((ids[ids>=g]-g)//K)
    assert not np.intersect1d(summary,rawgroups).size
    return ids,int(nsum+(target_type==0)),int(K*nraw+(K if target_type else 0))

def calibrate(layer,seed,g,channel,n):
    """4096 total trials; forced valid target. Fit and validation are disjoint."""
    r=rng(seed,g,40); ms=[m for m in (1,8,16,32,64,128,256) if m<=5*g]
    if layer=='E2':
        tau,sigma=channel; u,raw,keys=make_keys(seed,g,tau)
    else: signal,ws,wr=channel
    specs=[(m,t,frac,v) for m in ms for t in (0,1) for frac in (0.,.5,1.) for v in (0,1)]
    allcells=[]
    for part in range(2):
        cells={}
        for start in range(0,n//2,128):
            batch=min(128,n//2-start); ii=r.integers(g,size=batch); jj=r.integers(K,size=batch)
            if layer=='E2': scores=query_scores(r,u,raw,keys,ii,jj,sigma)
            else: scores=r.gumbel(size=(batch,5*g))+np.log(np.r_[np.full(g,ws),np.full(K*g,wr)])
            for row in range(batch):
                m,t,frac,v=specs[(start+row)%len(specs)]; target=int(ii[row] if t==0 else g+K*ii[row]+jj[row])
                # Sample mutually exclusive episode operators, as in a real index.
                # M=1 is a forced-single-evidence probe (raw cannot otherwise have M=1).
                chosen,ns,nr=calibration_index(seed,g,int(ii[row]),int(jj[row]),t,m,frac,v)
                vals=scores[row,chosen].copy()
                if layer=='E1': vals[0]+=math.log((signal+(ws if t==0 else wr))/(ws if t==0 else wr))
                winner=int(chosen[np.lexsort((chosen,-vals))[0]])
                key=(t,ns,nr,v); a=cells.setdefault(key,[0,0]); a[0]+=int(winner==target); a[1]+=1
        allcells.append([dict(t=k[0],ns=k[1],nr=k[2],variant=k[3],hits=v[0],n=v[1]) for k,v in sorted(cells.items())])
    fit,valid=allcells
    def arrays(cells): return [np.array([x[k] for x in cells]) for k in ('t','ns','nr','hits','n')]
    t,ns,nr,h,nn=arrays(fit); grid=np.logspace(-3,4,161)
    def loss(lams,weight):
        target=np.where(t==0,weight,1.)
        pp=(lams[:,None]+target)/(lams[:,None]+ns*weight+nr)
        pp=np.clip(pp,1e-12,1-1e-12)
        return -(h*np.log(pp)+(nn-h)*np.log1p(-pp)).sum(1)
    ix=int(loss(grid,1.).argmin()); hom=(float(grid[ix]),1.)
    best=(math.inf,None)
    for weight in np.logspace(-1,1,41):
        losses=loss(grid,weight); idx=int(losses.argmin())
        if losses[idx]<best[0]: best=(float(losses[idx]),(float(grid[idx]),float(weight)))
    def diagnostics(par):
        la,we=par; t,ns,nr,h,nn=arrays(valid)
        pred=(la+np.where(t==0,we,1.))/(la+ns*we+nr)
        # Prespecified cell maximum, without filtering low-count calibration failures.
        residual=float(np.max(np.abs(h/nn-pred)))
        differences=[]
        for tt in (0,1):
            for m in sorted(set((ns+nr).tolist())):
                rates=[]
                for v in (0,1):
                    ix=(t==tt)&((ns+nr)==m)&(np.array([x['variant'] for x in valid])==v)
                    if nn[ix].sum(): rates.append(float(h[ix].sum()/nn[ix].sum()))
                if len(rates)==2: differences.append(abs(rates[0]-rates[1]))
        subset=max(differences,default=0.)
        return dict(lambda_hat=la,ws_hat=we,max_residual=residual,subset_difference=subset,domain_pass=residual<=.03 and subset<=.04)
    return dict(homogeneous=diagnostics(hom),extended=diagnostics(best[1]),fit=fit,validation=valid)

def prepare(seed,g,dist,eps,n,ns,nr):
    p=probability(g,dist); h=rng(seed,g,11).choice(g,size=n,p=p); ph=np.bincount(h,minlength=g)/n
    r=rng(seed,g,10); z=r.integers(2,size=(g,32),dtype='int8'); e=r.random((g,K,32))<eps; e[:,0,:]=False
    source=z[:,None,:]^e
    rt=rng(seed,g,50); ii=rt.choice(g,size=n,p=p); jj=rt.integers(K,size=n); tt=rt.integers(32,size=n)
    summary=z[ii,tt]; truth=source[ii,jj,tt]
    flips=rng(seed,g,51).random(n); correct_s=(summary^(flips<ns))==truth; correct_r=flips>=nr
    guess=rng(seed,g,52).integers(2,size=n)==truth
    rc=rng(seed,g,41); ci=rc.integers(g,size=4096); cj=rc.integers(K,size=4096); ct=rc.integers(32,size=4096); ff=rc.random(4096)
    vs=float(2*np.mean((z[ci,ct]^(ff<ns))==source[ci,cj,ct])-1); vr=float(2*np.mean(ff>=nr)-1)
    return p,ph,ii,jj,correct_s,correct_r,guess,vs,vr

def active_indices(z):
    g=len(z); return np.r_[np.flatnonzero(z==1),g+(K*np.flatnonzero(z==2)[:,None]+np.arange(K)).ravel()]

def retrieve(scores,z,ii,jj,layer,channel,pad=None,route=False):
    g=len(z); n=len(ii); mode=z[ii]; target=np.where(mode==1,ii,g+K*ii+jj)
    ids=np.arange(g) if route else active_indices(z)
    if len(ids)==0: return np.zeros(n,dtype=bool)
    vals=scores[:,ids]; winner=ids[np.argmax(vals,axis=1)]; mx=vals.max(1)
    if pad is not None: mx=np.maximum(mx,pad.max(1))
    if route: target=ii; mode=np.full(n,2)
    targetscore=scores[np.arange(n),target].copy()
    if layer=='E1':
        lam,ws,wr=channel; ww=np.where(mode==1,ws,wr)
        if route: ww=np.full(n,ws)
        targetscore+=np.log((lam+ww)/ww)
    return (mode>0)&((targetscore>mx)|((targetscore==mx)&(target<=winner)))

def scan_counts(scores,ii,jj,cs,cr,guess,permutation):
    """All count pairs via raw prefix maxima and summary interval maxima."""
    g=len(permutation); batch=len(ii); nscores=scores[:,permutation]
    rids=g+(K*permutation[:,None]+np.arange(K)).ravel(); rvals=scores[:,rids].reshape(batch,g,K)
    ridmat=np.broadcast_to(rids.reshape(g,K),(batch,g,K)); rbest=rvals.max(2)
    rarg=np.where(rvals==rbest[:,:,None],ridmat,10*g).min(2)
    sums=[]; counts=[]
    mxr=np.full(batch,-np.inf); ir=np.full(batch,10*g)
    for n in range(g+1):
        if n:
            vv=rbest[:,n-1]; ids=rarg[:,n-1]; better=(vv>mxr)|((vv==mxr)&(ids<ir)); mxr=np.maximum(mxr,vv); ir=np.where(better,ids,ir)
        mxs=np.full(batch,-np.inf); iss=np.full(batch,10*g)
        for s in range(g-n+1):
            if s:
                vv=nscores[:,n+s-1]; idx=permutation[n+s-1]; better=(vv>mxs)|((vv==mxs)&(idx<iss)); mxs=np.maximum(mxs,vv); iss=np.where(better,idx,iss)
            winner=np.where((mxs>mxr)|((mxs==mxr)&(iss<ir)),iss,ir)
            hs=winner==ii; hr=winner==(g+K*ii+jj)
            good=np.where(hs,cs,np.where(hr,cr,guess))
            sums.append(int(good.sum())); counts.append((n,s))
    return np.array(sums,dtype='int64'),counts

def scan_counts_fast(scores,ii,jj,cs,cr,guess,permutation):
    g=len(permutation); batch=len(ii); width=scores.shape[1]
    order=width-1-np.argsort(scores[:,::-1],axis=1,kind='stable')
    rank=np.empty_like(order); np.put_along_axis(rank,order,np.broadcast_to(np.arange(width),order.shape),axis=1)
    sr=rank[:,permutation]; rr=rank[:,g+(K*permutation[:,None]+np.arange(K)).ravel()].reshape(batch,g,K).max(2)
    pref=np.concatenate([np.full((batch,1),-1),np.maximum.accumulate(rr,axis=1)],axis=1)
    sums=[]; counts=[]; rows=np.arange(batch)[:,None]
    for n in range(g+1):
        sm=np.concatenate([np.full((batch,1),-1),np.maximum.accumulate(sr[:,n:],axis=1)],axis=1) if n<g else np.full((batch,1),-1)
        best=np.maximum(pref[:,n,None],sm); winner=order[rows,np.maximum(best,0)]
        hs=(best>=0)&(winner==ii[:,None]); hr=(best>=0)&(winner==(g+K*ii+jj)[:,None])
        correct=np.where(hs,cs[:,None],np.where(hr,cr[:,None],guess[:,None]))
        sums.extend(correct.sum(0).tolist()); counts.extend((n,s) for s in range(g-n+1))
    return np.array(sums,dtype='int64'),counts

def run_condition(config,cal,outdir,n=8192,permutations=10):
    start=time.monotonic(); layer,seed,g,dist,eps,ns,nr,channel=config
    p,ph,ii,jj,cs,cr,guess,vs,vr=prepare(seed,g,dist,eps,n,ns,nr)
    model=cal['homogeneous']; lam=model['lambda_hat']; ws=model['ws_hat']
    ps=p if dist==0 else ph
    policies=[]; r=rng(seed,g,60)
    for b in budgets(g):
        for name,z in make_policies(ps,b,lam,ws,vs,vr,r).items(): policies.append((b,name,z))
    fulls=np.ones(g,dtype=int); fullr=np.full(g,2,dtype=int)
    policies.extend([(4*g,'full_summary',fulls),(4*g,'full_raw',fullr),(4*g,'padded_summary',fulls),(4*g,'cold_archive',fulls),(4*g,'route_raw',fullr)])
    if dist:
        # Budget-feasible two-threshold grid and fixed random policies.
        order=np.argsort(-ph,kind='stable')
        for b in budgets(g):
            for nraw in sorted(set(np.linspace(0,min(g,b//K),7,dtype=int))):
                for nsum in sorted(set(np.linspace(0,min(g-nraw,b-K*nraw),7,dtype=int))):
                    z=np.zeros(g,dtype=int); z[order[:nraw]]=2; z[order[nraw:nraw+nsum]]=1
                    policies.append((b,'threshold_candidate',z))
            for _ in range(32): policies.append((b,'random_candidate',make_random(g,b,r)))
    size=len(policies); results=np.zeros((size,n),dtype='bool'); hits=np.zeros(size,dtype='int64'); rb_gain=np.zeros(size); covered=np.zeros(size,dtype='int64')
    enum=np.zeros((permutations,(g+1)*(g+2)//2),dtype='int64') if layer=='E2' and dist==0 else None
    perms=[np.arange(g)]+[rng(seed,g,200+i).permutation(g) for i in range(1,permutations)]
    rt=rng(seed,g,53)
    if layer=='E2':
        tau,sigma=channel; u,raw,keys=make_keys(seed,g,tau)
        # Frozen independent distractor keys for the equal-count intervention.
        padkeys=norm(rng(seed,g,54).normal(size=(3*g,32)))
    for at in range(0,n,128):
        en=min(at+128,n); ix=ii[at:en]; jx=jj[at:en]; batch=en-at
        if layer=='E1':
            true_lam,true_ws,true_wr=channel
            scores=rt.gumbel(size=(batch,5*g))+np.log(np.r_[np.full(g,true_ws),np.full(4*g,true_wr)])
            pad=rt.gumbel(size=(batch,3*g))+math.log(true_wr)
        else:
            q=norm(raw[ix,jx]+sigma*rt.normal(size=(batch,32))); scores=q@keys.T; pad=q@padkeys.T
        for a,(b,name,z) in enumerate(policies):
            route=name=='route_raw'; hh=retrieve(scores,z,ix,jx,layer,channel,pad=pad if name=='padded_summary' else None,route=route)
            mode=z[ix]; correct=np.where(mode==1,cs[at:en],cr[at:en])
            results[a,at:en]=np.where(hh,correct,guess[at:en]); hits[a]+=hh.sum(); covered[a]+=(mode>0).sum()
            rb_gain[a]+=np.sum(hh*(2*correct.astype(int)-1))
        if enum is not None:
            for pi,perm in enumerate(perms):
                sums,pairs=scan_counts_fast(scores,ix,jx,cs[at:en],cr[at:en],guess[at:en],perm); enum[pi]+=sums
    candidate_best={}
    for b in budgets(g):
        eligible=[a for a,(bb,name,z) in enumerate(policies) if bb==b and name not in ('padded_summary','cold_archive','route_raw')]
        best=float(results[eligible].mean(1).max())
        if enum is not None:
            feasible=np.array([nn*K+ss<=b for nn,ss in pairs]); best=max(best,float(enum[:,feasible].max()/n))
        candidate_best[b]=best
    rows=[]
    for a,(b,name,z) in enumerate(policies):
        if name.endswith('_candidate'): continue
        route=name=='route_raw'; padcount=3*g if name=='padded_summary' else 0
        w=float(np.array([0,ws,K])[z].sum()+padcount)
        if route: pred=.5*(1+vr*(lam+1)/(lam+g))
        else: pred=.5*(1+float(ps@np.array([0,vs*(lam+ws),vr*(lam+1)])[z])/(lam+w))
        acc=float(results[a].mean()); nraw=int((z==2).sum()); nsum=int((z==1).sum()); packets=nraw*K+nsum
        # Rao-Blackwell only the fair guess; keep source, retrieval and reader realizations.
        actual_gain=float(2*acc-1)
        row=dict(budget=b,policy=name,n=nraw,s=nsum,W=w,accuracy=acc,predicted_accuracy=pred,abs_error=abs(acc-pred),se=math.sqrt(acc*(1-acc)/n),recall=float(hits[a]/n),conditional_recall=float(hits[a]/covered[a]) if covered[a] else None,missing_rate=float(1-covered[a]/n),coverage_mass=float(p[z>0].sum()),regret=max(0.,candidate_best[b]-acc),stored_bytes=136*(packets+padcount+(4*g if name=='cold_archive' else 0)),active_key_bytes=128*(g if route else packets+padcount),cold_bytes=136*4*g if name=='cold_archive' else 0,route_offset_bytes=8*g if route else 0,extra_random_reads=int(route),actual_gain=actual_gain,expected_gain_no_guess=float(rb_gain[a]/n))
        if layer=='E1':
            ll,ww,wr=channel; load=np.array([0,ww,K*wr])[z].sum()+padcount*wr
            row['oracle_accuracy']=.5*(1+(float(p@np.array([0,(1-2*ns)*(1-1.5*eps)*(ll+ww),(1-2*nr)*(ll+wr)])[z])/(ll+load) if not route else (1-2*nr)*(ll+ww)/(ll+g*ww)))
        ext=cal['extended']; le,we=ext['lambda_hat'],ext['ws_hat']; row['extended_predicted_accuracy']=.5*(1+(vr*(le+we)/(le+g*we) if route else value(ps,z,le,we,vs,vr)*(le+np.array([0,we,K])[z].sum())/(le+np.array([0,we,K])[z].sum()+padcount)))
        rows.append(row)
    endpoints={name:a for a,(_,name,_) in enumerate(policies) if name in ('full_summary','full_raw','padded_summary','cold_archive','route_raw')}
    endpointbits=np.array([results[endpoints[x]] for x in endpoints],dtype='int8')
    assert np.array_equal(results[endpoints['full_summary']],results[endpoints['cold_archive']])
    paired={}
    rb=rng(seed,g,70)
    for left,right in [('full_summary','full_raw'),('padded_summary','full_raw'),('route_raw','full_summary'),('cold_archive','full_summary')]:
        diff=results[endpoints[left]].astype('int8')-results[endpoints[right]].astype('int8'); samples=[]
        # Exact paired bootstrap for mean ternary differences via multinomial counts.
        pp=np.bincount(diff+1,minlength=3)/n
        for offset in range(0,2000,128):
            cc=rb.multinomial(n,pp,size=min(128,2000-offset)); samples.extend(((cc[:,2]-cc[:,0])/n).tolist())
        paired[left+'-'+right]=dict(mean=float(diff.mean()),ci95=np.quantile(samples,[.025,.975]).tolist())
    cid=config_id(config)
    np.savez_compressed(outdir/(cid+'.npz'),endpoint_names=np.array(list(endpoints)),endpoint_correct=endpointbits,counts=np.array(pairs) if enum is not None else np.empty((0,2)),candidate_correct_counts=enum if enum is not None else np.empty((0,0)))
    return dict(id=cid,layer=layer,seed=seed,G=g,dist=dist,epsilon=eps,nu_s=ns,nu_r=nr,channel=channel,n_test=n,p=p.tolist(),p_hat=ph.tolist(),entropy=entropy(p),estimated_entropy=entropy(ph),v_s_hat=vs,v_r_hat=vr,v_s_true=(1-2*ns)*(1-1.5*eps),v_r_true=1-2*nr,calibration=cal,rows=rows,paired=paired,permutations_completed=permutations if enum is not None else 0,elapsed_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)

def make_random(g,b,r):
    z=np.zeros(g,dtype=int)
    for idx in r.permutation(g):
        mode=int(r.choice(np.flatnonzero(np.array([0,1,K])<=b))); z[idx]=mode; b-=int([0,1,K][mode])
    return z

def config_id(c):
    layer,seed,g,dist,eps,ns,nr,ch=c
    return f'{layer}_s{seed}_G{g}_p{dist}_e{eps:.6f}_reader{ns}_{nr}_ch'+('_'.join(str(x) for x in ch))
