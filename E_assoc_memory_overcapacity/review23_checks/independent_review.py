"""Review 23: independent checks; never imports or executes reviewed programs.

Run only with the specified Python inside the memory-limited systemd scope.
All outputs are new files in this directory, opened exclusively.
"""
import os
import json
import time
import math
import hashlib
import resource
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm, binom

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
START = time.monotonic()
SEED = 2306112026
DESIGN = {64: 64, 128: 48, 256: 32, 512: 16}
METHODS = ('GD3', 'Sign2', 'Adam2')

def emit(name, obj):
    with (OUT / name).open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)

def progress(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)

def input_hashes():
    names = ['AC2_CONCENTRATION.md', 'AC2_TRANSFER_PRINCIPLE.md',
             'AC2_AC1_FIXES.md', 'AC2_NUMERICAL_RESULTS.md',
             'AC1_TOPL_PROOF.md', 'AC1_RESULTS.md', 'AC1_NOVELTY.md',
             'PROOF_REVIEW22.md', 'AA1_QUANT_RANK.md', 'ICML_REVIEW_SIM8.md']
    paths = [ROOT / x for x in names]
    for directory in ('ac2_checks', 'paper'):
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file())
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}

def raw_reaggregation():
    raw = [json.loads(s) for s in (ROOT/'ac2_checks/full/AC2_datasets.jsonl').read_text().splitlines()]
    published = json.loads((ROOT/'ac2_checks/full/AC2_results.json').read_text())
    grouped = defaultdict(list)
    for r in raw:
        grouped[(r['method'],r['d'])].append(r)
    answer = []
    maxdiff = 0.
    for pub in published['rows']:
        rr = grouped[(pub['method'], pub['d'])]
        assert len(rr) == pub['datasets']
        assert sorted(r['rep'] for r in rr) == list(range(len(rr)))
        assert all(r['n'] == pub['n'] for r in rr)
        assert all(0 <= r['counts'][0] <= r['counts'][1] <= r['n'] for r in rr)
        assert all(r['recall'] == [k/r['n'] for k in r['counts']] for r in rr)
        # Compute with integer totals and math.fsum, not the report's NumPy path.
        avg = [math.fsum(r['counts'][j]/r['n'] for r in rr)/len(rr) for j in (0,1)]
        var = [math.fsum((r['counts'][j]/r['n']-avg[j])**2 for r in rr)/(len(rr)-1)
               for j in (0,1)]
        se = [math.sqrt(v/len(rr)) for v in var]
        for key, val in [('mean', avg), ('SE', se), ('dataset_variance', var)]:
            maxdiff = max(maxdiff, max(abs(x-y) for x,y in zip(pub[key],val)))
        answer.append(dict(method=pub['method'],d=pub['d'],n=pub['n'],B=len(rr),
                           mean=avg,variance=var,SE=se,n_variance=[pub['n']*v for v in var]))
    trends=[]
    for method in METHODS:
        rr=sorted((r for r in answer if r['method']==method),key=lambda r:r['d'])
        for j,L in enumerate((1,5)):
            vv=[r['variance'][j] for r in rr]
            trends.append(dict(method=method,L=L,variances=vv,
                strictly_decreasing=all(vv[i]>vv[i+1] for i in range(3)),ratio=vv[0]/vv[-1]))
    assert len(raw)==720 and len(answer)==12 and maxdiff<1e-14
    return dict(datasets=len(raw),queries=sum(r['n'] for r in raw),max_summary_difference=maxdiff,
                rows=answer,trends=trends,resources_claimed=published['resources'])

def endpoint(z,s,method):
    """Independent recurrence, accepts scalars or matrices."""
    u=np.zeros_like(z)
    first=np.zeros_like(z); second=np.zeros_like(z)
    steps=3 if method=='GD3' else 2
    for step in range(1,steps+1):
        residual = z - (u*s if np.ndim(z)==0 else u@s)
        if method=='GD3':
            increment=.7*residual
        elif method=='Sign2':
            increment=.6*np.sign(residual)
        else:
            first += .1*(residual-first)
            second += .01*(residual**2-second)
            increment=.3*(first/(1-.9**step))/(np.sqrt(second/(1-.99**step))+1.)
        u=u+increment
    return u

def scalar_moments():
    moments={}
    for method in METHODS:
        # Integrate on both half-lines; Sign2 also checked by closed Gaussian tails.
        cuts=[-np.inf,-.6,0.,.6,np.inf] if method=='Sign2' else [-np.inf,0.,np.inf]
        a=math.fsum(quad(lambda x: x*float(endpoint(np.array(x),1.,method))*norm.pdf(x),
                        lo,hi,epsabs=2e-12)[0] for lo,hi in zip(cuts[:-1],cuts[1:]))
        q=math.fsum(quad(lambda x: float(endpoint(np.array(x),1.,method))**2*norm.pdf(x),
                        lo,hi,epsabs=2e-12)[0] for lo,hi in zip(cuts[:-1],cuts[1:]))
        moments[method]=dict(a=a,q=q,c=a/math.sqrt(q))
    exact_a=2.4*norm.pdf(.6); exact_q=2.88*norm.sf(.6)
    moments['Sign2_closed_form'] = dict(a=exact_a,q=exact_q,c=exact_a/math.sqrt(exact_q))
    assert abs(moments['Sign2']['c']-moments['Sign2_closed_form']['c'])<1e-12
    return moments

def choose_load(d,c):
    # Continuous root, then evaluate the two neighboring integers.
    f=lambda x:c*d/math.sqrt(x)-norm.isf(1/(x-1))
    real=brentq(f,4.,4.*d*d)
    n=min((math.floor(real),math.ceil(real)),key=lambda k:abs(f(k)))
    return n, f(n)

def full_ranks(x,y,u):
    """All queries and all candidates; no stopping and no rank proxy."""
    # This fresh-data check records ordinary ranks only. Fail explicitly if the
    # zero-endpoint atom occurs; the separate edge check tests its tie rule.
    assert np.any(u), 'zero endpoint requires an explicit tie rule'
    transformed=u@x
    n=x.shape[1]
    wins=[0,0]
    for offset in range(0,n,128):
        stop=min(offset+128,n)
        # An entire candidate column is computed at once.
        matrix=y.T@transformed[:,offset:stop]
        ids=np.arange(offset,stop)
        diag=matrix[ids,np.arange(stop-offset)].copy()
        # Same arithmetic path for the correct score and all comparisons.
        matrix[ids,np.arange(stop-offset)]=-np.inf
        greater=np.count_nonzero(matrix>diag,axis=0)
        wins[0]+=int(np.count_nonzero(greater==0))
        wins[1]+=int(np.count_nonzero(greater<5))
    return wins

def transpose_check(rng):
    records=[]
    d,n=9,85
    for method in METHODS:
        x=rng.standard_normal((d,n))/math.sqrt(d)
        y=rng.standard_normal((d,n))/math.sqrt(d)
        u=endpoint((d/math.sqrt(n))*(y@x.T),(d/n)*(x@x.T),method)
        data=np.concatenate((x,y),axis=0)
        f=np.linalg.cholesky((d/n)*(data@data.T))
        q=np.linalg.solve(f,math.sqrt(d/n)*data)
        b=f[d:,:].T@u@f[:d,:]/np.linalg.norm(u)
        scores=(d/np.linalg.norm(u))*(y.T@u@x)
        correct=scores.diagonal().copy()
        direct=(np.einsum('ij,ij->j',scores,scores)-correct**2)/(n-1)
        column=(n*n*np.einsum('ij,ij->j',b@q,b@q)-correct**2)/(n-1)
        row=(n*n*np.einsum('ij,ij->j',b.T@q,b.T@q)-correct**2)/(n-1)
        k=(b+b.T)/2
        pencil=max(float(np.max(abs(np.linalg.eigvalsh(t*k+s*b.T@b)-
                                     np.linalg.eigvalsh(t*k+s*b@b.T))))
                   for t,s in ((2.,-.7),(-.3,1.1),(0.,2.)))
        records.append(dict(method=method,column_error=float(max(abs(direct-column))),
                            transpose_error=float(max(abs(direct-row))),pencil_error=pencil))
        assert max(abs(direct-column))<1e-12
    return records

def haar_pair_check(rng):
    result=[]
    trials=12000
    for n in (64,256,1024,4096):
        # A diagonal trace-zero unit-variance quadratic form, including all n coordinates.
        lam=np.r_[np.ones(n//2),-np.ones(n//2)]/math.sqrt(2*n)
        ds=[]; errors=[]
        for offset in range(0,trials,120):
            size=min(120,trials-offset)
            original=rng.standard_normal((size,n,2))
            first=original[:,:,0]/np.linalg.norm(original[:,:,0],axis=1)[:,None]
            second=original[:,:,1]-first*np.sum(first*original[:,:,1],axis=1)[:,None]
            second/=np.linalg.norm(second,axis=1)[:,None]
            w=np.stack((first,second),axis=2)
            d=n*np.einsum('bnj,n,bnj->bj',w,lam,w)
            raw=np.einsum('bnj,n,bnj->bj',original,lam,original)
            errors.append(abs(d-raw))
            ds.append(d)
        d=np.concatenate(ds); error=np.concatenate(errors)
        grid=(-1.,0.,1.)
        discrepancy=max(abs(np.mean((d[:,0]<=a)&(d[:,1]<=b))-ndtr(a)*ndtr(b))
                        for a in grid for b in grid)
        p=np.mean(d>0,axis=0)
        result.append(dict(n=n,trials=trials,coupling_L1=error.mean(axis=0).tolist(),
            coupling_bound=8/math.sqrt(n),variance=d.var(axis=0,ddof=1).tolist(),
            exact_variance=n/(n+2),empirical_covariance=float(np.cov(d.T)[0,1]),
            exact_covariance=-n/((n+2)*(n-1)),half_rectangle_error=float(discrepancy),
            sign_event_covariance=float(np.mean((d[:,0]>0)&(d[:,1]>0))-p[0]*p[1])))
    return result

def joint_extreme_check(rng):
    """A legal fixed-C endpoint, plus one common complement Stiefel frame.

    C=[I rI; rI I], U=I, d=n/4. B has two scalar identity blocks.
    Simulates both true columns, their shared candidates and the two special candidates.
    """
    rows=[]
    trials=2400
    for n in (256,1024,4096):
        d=n//4; k=n-2; b=norm.isf(1/(n-1)); r=b/math.sqrt(d)
        a=math.sqrt(1-r*r)
        obs=[]
        for repeat in range(trials):
            w,_=np.linalg.qr(rng.standard_normal((n,2)),mode='reduced')
            w*=rng.choice((-1.,1.),size=2)
            # Multiplication by the explicit block B; rank d, embedded in n dimensions.
            bw=np.zeros((n,2))
            bw[:d]=r*w[:d]/math.sqrt(d)
            bw[d:2*d]=a*w[:d]/math.sqrt(d)
            correct=n*np.sum(w*bw,axis=0)
            cross=n*(w.T@bw)
            v=bw-w@(w.T@bw)
            gram=v.T@v
            values,vectors=np.linalg.eigh(gram)
            square=(vectors*np.sqrt(np.maximum(values,0)))@vectors.T
            h,_=np.linalg.qr(rng.standard_normal((k,2)),mode='reduced')
            # QR convention does not affect H's invariant law once signs are randomized.
            h*=rng.choice((-1.,1.),size=2)
            scores=n*h@square
            col0=np.r_[cross[1,0],scores[:,0]]
            col1=np.r_[cross[0,1],scores[:,1]]
            top=[]
            for L in (1,5):
                top.extend((np.partition(col0,-L)[-L],np.partition(col1,-L)[-L]))
            obs.append(np.r_[correct-b,top])
        z=np.array(obs)
        entries=[]
        for j,L in enumerate((1,5)):
            maxima=z[:,2+2*j:4+2*j]
            event=z[:,:2]+b>=maxima
            p=event.mean(axis=0)
            scaled=b*(maxima-b)
            entries.append(dict(L=L,success=p.tolist(),success_covariance=float(
                np.mean(event[:,0]&event[:,1])-p[0]*p[1]),
                scaled_extreme_correlation=float(np.corrcoef(scaled.T)[0,1]),
                cross_correct_extreme_correlations=np.corrcoef(np.c_[z[:,:2],scaled].T)[:2,2:].tolist()))
        rows.append(dict(n=n,d=d,trials=trials,mu=b,delta=r,R=1/math.sqrt(d),
                         b2_delta=b*b*r,entries=entries))
    return rows

def root_and_rate_checks():
    # Independent integration by splitting the Gaussian line at polynomial roots.
    # This covers either sign of the threshold instead of reusing the reviewed larger root.
    cases=[]
    parameters=[(2.,.1,50.,900.,3.,1000),(2.,.1,50.,900.,-1.,1000),
                (2.,.999,50.,10.,3.,100),(0.,1.,2.,0.,1.,10)]
    for ell,h,p,j,t,n in parameters:
        c=math.sqrt(1-h)
        poly=[(n-1)*c*c-t*t*h,2*ell*c*(n-1+t*t),
              (n-1+t*t)*ell*ell-t*t*(p+j)]
        rr=np.roots(poly) if abs(poly[0])>1e-15 else np.roots(poly[1:])
        cuts=sorted(x.real for x in rr if abs(x.imag)<1e-9)
        limits=[-np.inf]+cuts+[np.inf]
        prob=0.
        for lo,hi in zip(limits[:-1],limits[1:]):
            point=(0. if lo==-np.inf and hi==np.inf else
                   hi-1 if lo==-np.inf else lo+1 if hi==np.inf else (lo+hi)/2)
            q=ell+c*point
            energy=(p+point*point+j-q*q)/(n-1)
            if q>=t*math.sqrt(max(0.,energy)):
                prob+=ndtr(hi)-ndtr(lo)
        H=(n-1+t*t)*(1-h)-t*t
        cases.append(dict(parameters=[ell,h,p,j,t,n],H=H,success_probability=float(prob),
                          exception=(t<=0 or H<=0 or h>=1)))
    threshold=brentq(lambda x:64*math.log(x)/x-1,100.,1000.)
    budgets=[]
    for logd in (math.log(64),math.log(512),math.log(10**6),100.,1000.,10000.):
        # Use logs to avoid representing astronomical d or n.
        b=2*math.sqrt(logd)
        logn=2*logd-math.log(4*logd)
        lam=math.exp((logd-logn)/2)
        ed=lam**(1/3)
        h=64*math.log(b)/b
        term=(b*b/math.log(b))*ed
        budgets.append(dict(log_d=logd,b=b,h=h,Sign_energy_term=term,
                            target_order=math.log(logd)**2/logd))
    return dict(root_cases=cases,Hoeffding_radius=math.sqrt(math.log(144/.01)/256),
                h_le_1_requires_b_at_least=threshold,
                corresponding_leading_log_d=threshold**2/4,rate_diagnostics=budgets)

def fresh_full_datasets(rng,moments):
    rows=[]; totals=0; summaries=[]
    with (OUT/'fresh_datasets.jsonl').open('x',buffering=1) as handle:
        for d,B in DESIGN.items():
            for method in METHODS:
                n,s=choose_load(d,moments[method]['c'])
                vals=[]
                for rep in range(B):
                    x=rng.standard_normal((d,n))/math.sqrt(d)
                    y=rng.standard_normal((d,n))/math.sqrt(d)
                    u=endpoint((d/math.sqrt(n))*(y@x.T),(d/n)*(x@x.T),method)
                    counts=full_ranks(x,y,u)
                    val=np.array(counts)/n
                    vals.append(val)
                    handle.write(json.dumps(dict(d=d,n=n,method=method,rep=rep,
                        counts=counts,recall=val.tolist(),s_eff=s))+'\n')
                    totals+=n
                    if (rep+1)%8==0:
                        progress(dict(stage='fresh',d=d,method=method,done=rep+1,B=B,
                                      elapsed=time.monotonic()-START))
                vals=np.array(vals)
                row=dict(d=d,n=n,method=method,B=B,mean=vals.mean(axis=0).tolist(),
                    variance=vals.var(axis=0,ddof=1).tolist(),
                    SE=(vals.std(axis=0,ddof=1)/math.sqrt(B)).tolist())
                summaries.append(row); progress(row)
    return dict(seed=SEED,design=DESIGN,datasets=sum(DESIGN.values())*3,
                all_original_queries=totals,rows=summaries)

def resources():
    cg=Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
    base=Path('/sys/fs/cgroup')/cg.lstrip('/')
    info=dict(wall_seconds=time.monotonic()-START,cpu_seconds=time.process_time(),
              peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              threads=next(s for s in Path('/proc/self/status').read_text().splitlines()
                           if s.startswith('Threads:')))
    for name in ('memory.max','memory.swap.max','memory.peak'):
        try: info[name]=(base/name).read_text().strip()
        except OSError: pass
    return info

if __name__=='__main__':
    before=input_hashes(); emit('inputs_before_sha256.json',before)
    emit('design.json',dict(seed=SEED,fresh_datasets=DESIGN,
        methods=METHODS,lists=[1,5],ranking='all queries/all candidates; no early stopping',
        independent_unit='one whole X,Y dataset',threads_limit=2))
    rng=np.random.default_rng(SEED)
    moments=scalar_moments(); emit('moments.json',moments)
    emit('raw_reaggregation.json',raw_reaggregation()); progress({'stage':'raw_verified'})
    emit('transpose.json',transpose_check(rng))
    emit('haar_pairs.json',haar_pair_check(rng)); progress({'stage':'haar_pairs_done'})
    emit('joint_extremes.json',joint_extreme_check(rng)); progress({'stage':'joint_extremes_done'})
    emit('root_and_rates.json',root_and_rate_checks())
    fresh=fresh_full_datasets(rng,moments); emit('fresh_results.json',fresh)
    after=input_hashes()
    changed=[p for p in before if before[p]!=after.get(p)]
    emit('integrity.json',dict(checked_files=len(before),changed=changed,
                              added_inputs=sorted(set(after)-set(before)),after=after))
    assert not changed
    info=resources(); emit('resources.json',info); progress(info)
