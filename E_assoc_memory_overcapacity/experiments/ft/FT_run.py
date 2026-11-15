import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key) == '2', (key, os.environ.get(key))
import sys, json, time, math, hashlib, resource, csv
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.optimize import brentq
from scipy.stats import t as student

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/ft'
METHODS=['GD','Sign','Adam_e0.1','Adam_e1','AdamW_e0.1','Muon']
COORD=METHODS[:4]
CASES=[('independent',0.,0.),('samekey_random',1.,0.),('aligned',1.,.5),
       ('conflict',1.,-.5),('correlated',math.sqrt(.5),math.sqrt(.5))]
DS=[64,128]
NS=20
NQ=64
TS=[1,4]
TARGET=.75
START=time.monotonic()
gx,gw=np.polynomial.legendre.leggauss(512)
QN=(gx+1)*6
QW=gw*6*2*np.exp(-QN**2/2)/math.sqrt(2*math.pi)

def dump(path,obj):
    with open(path,'x') as f: json.dump(obj,f,indent=2,allow_nan=False)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze_check():
    for line in (ROOT/'FT_FREEZE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert sha(ROOT/name)==digest, name

def metadata(mode,extra=None):
    return dict(mode=mode,seconds=time.monotonic()-START,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        executable=sys.executable,numpy=np.__version__,threads={k:os.environ[k] for k in
        ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        resource_command='systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0',
        **(extra or {}))

def delta(method,T,step,sigma,z):
    z=np.asarray(z)
    if method=='GD': return (1-(1-step)**T)*sigma*z
    u=np.zeros_like(z); m=np.zeros_like(z); v=np.zeros_like(z)
    for it in range(1,T+1):
        res=u-sigma*z
        if method=='Sign': direction=np.sign(res)
        else:
            e=.1 if '0.1' in method else 1.
            m=.9*m+.1*res; v=.99*v+.01*res**2
            direction=(m/(1-.9**it))/(np.sqrt(v/(1-.99**it))+e*sigma)
        u-=step*sigma*direction
    return u

def moments(method,T,step,r):
    sigma=math.sqrt(2*(1-r))
    if method=='GD':
        k=1-(1-step)**T; b=k*sigma*sigma; v=k*k*sigma*sigma
    elif method=='Sign':
        # All thresholds are integer multiples of alpha, integrated exactly.
        edges=np.r_[np.arange(T+1)*step,np.inf]
        mid=np.where(np.isfinite(edges[1:]),(edges[:-1]+edges[1:])/2,edges[:-1]+1)
        ds=delta(method,T,step,sigma,mid)
        phi=lambda x:np.exp(-x*x/2)/math.sqrt(2*math.pi)
        b=float(2*sigma*np.sum(ds*(phi(edges[:-1])-phi(edges[1:]))))
        v=float(2*np.sum(ds**2*(ndtr(edges[1:])-ndtr(edges[:-1]))))
    else:
        ds=delta(method,T,step,sigma,QN)
        b=float(np.sum(QW*sigma*QN*ds)); v=float(np.sum(QW*ds*ds))
    q=1-b+v; a0=1-b/2; a1=r+b/2; loss=sigma*sigma-2*b+v
    out=dict(b=b,v=v,cross=-b/2,q=q,a0=a0,a1=a1,loss=loss,relative_loss=loss/(sigma*sigma))
    for name,a in [('old',a0),('new',a1)]:
        c=a/math.sqrt(q)
        out[name+'_msr']=c
        out[name+'_rank']=float(ndtr(-c/math.sqrt(2)))
        out[name+'_top10']=float(ndtr(c-ndtri(.9)))
    return out

def first_bracket(fn,T):
    prev=0.; fp=fn(0.)
    for cur in np.linspace(0,.6/T,33)[1:]:
        fc=fn(float(cur))
        if fp>=0 and fc<=0: return prev,float(cur)
        prev=float(cur); fp=fc
    return None

def prepare():
    forecasts=[]
    for case,rx,ry in CASES:
        r=rx*ry
        for T in TS:
            for method in COORD:
                fn=lambda a:moments(method,T,a,r)['relative_loss']-TARGET
                lo,hi=first_bracket(fn,T)
                step=brentq(fn,lo,hi,xtol=1e-13)
                forecasts.append(dict(case=case,r=r,T=T,method=method,step=step,
                    **moments(method,T,step,r)))
    dump(OUT/'FT_forecasts.json',dict(rows=forecasts,quadrature='512-point Legendre on [0,12], sign exact intervals'))
    dump(OUT/'FT_prepare_meta.json',metadata('prepare'))

def data(d,case,rx,ry,seed,tag):
    n=d*d
    entropy=[20261106,864219,d,case,seed,tag]
    def rand(j): return np.random.default_rng(np.random.SeedSequence(entropy+[j]))
    x0=rand(0).standard_normal((d,n))/math.sqrt(d)
    y0=rand(1).standard_normal((d,n))/math.sqrt(d)
    x1=rx*x0+math.sqrt(1-rx*rx)*rand(2).standard_normal((d,n)) / math.sqrt(d)
    y1=ry*y0+math.sqrt(1-ry*ry)*rand(3).standard_normal((d,n)) / math.sqrt(d)
    ids=rand(4).choice(n,min(NQ,n),replace=False)
    return x0,y0,x1,y1,ids,entropy

def train(method,T,step,sigma,u0,s,z):
    u=u0.copy(); m=np.zeros_like(u); v=np.zeros_like(u)
    for it in range(1,T+1):
        res=u@s-z
        if method=='GD': u-=step*res; continue
        if method=='Sign': direction=np.sign(res)
        elif method=='Muon':
            m=.9*m+.1*res
            aa,ss,bb=np.linalg.svd(m/(1-.9**it),full_matrices=False)
            direction=math.sqrt(len(u))*(aa@bb)
        else:
            e=.1 if '0.1' in method else 1.
            m=.9*m+.1*res; v=.99*v+.01*res**2
            direction=(m/(1-.9**it))/(np.sqrt(v/(1-.99**it))+e*sigma)
        if method.startswith('AdamW'): u*=1-step*sigma*.02
        u-=step*sigma*direction
    return u

def excess(u,opt,s):
    diff=u-opt
    return float(np.sum((diff@s)*diff)/u.size)

def matched(method,T,sigma,u0,s,z,opt,base):
    cache={}
    def evaluate(a):
        if a not in cache:
            u=train(method,T,a,sigma,u0,s,z)
            cache[a]=(excess(u,opt,s)/base,u)
        return cache[a][0]-TARGET
    bracket=first_bracket(evaluate,T)
    if bracket is None: return None
    lo,hi=bracket
    for _ in range(24):
        mid=(lo+hi)/2
        if evaluate(mid)>0: lo=mid
        else: hi=mid
    best=min(cache,key=lambda a:abs(cache[a][0]-TARGET))
    rel,u=cache[best]
    return best,u,rel

def measurements(u,x,y,ids):
    d,n=x.shape
    ux=u@x
    diag=np.sum(y*ux,axis=0)
    wrongmean=(y.sum(axis=1)@ux-diag)/(n-1)
    # ||Y^T U X||_F^2 = tr(U XX^T U^T YY^T), no n by n array.
    total=float(np.sum((ux@ux.T)*(y@y.T)))
    variance=(total-float(diag@diag))/(n*(n-1))-float(np.mean(wrongmean**2))
    margin=float(np.mean(diag-wrongmean))
    uv=ux[:,ids]
    truth=diag[ids]
    greater=np.zeros(len(ids),dtype=np.int64)
    equal=np.zeros(len(ids),dtype=np.int64)
    for lo in range(0,n,2048):
        hi=min(lo+2048,n)
        scores=y[:,lo:hi].T@uv
        # Correct score comes from a different BLAS path; exclude it explicitly.
        for j,idx in enumerate(ids):
            if lo<=idx<hi: scores[idx-lo,j]=-np.inf
        greater+=np.sum(scores>truth,axis=0)
        equal+=np.sum(scores==truth,axis=0)
    assert equal.max()==0, 'Unexpected exact tie: preserve output and amend separately'
    ranks=greater+1
    return dict(margin=margin,variance=variance,correct_mean=float(diag.mean()),
        correct_variance=float(diag.var()),msr=margin/math.sqrt(variance),
        rank=float(np.mean(greater/(n-1))),top10=float(np.mean(ranks<=math.ceil(.1*n))),
        ids=ids.tolist(),ranks=ranks.tolist(),correct_scores=truth.tolist())

def configuration(d,ic,seed,tag,smoke=False):
    case,rx,ry=CASES[ic]; r=rx*ry; sigma=math.sqrt(2*(1-r))
    x0,y0,x1,y1,ids,ent=data(d,ic,rx,ry,seed,tag)
    n=d*d
    s=d/n*(x1@x1.T); z0=d/math.sqrt(n)*(y0@x0.T); z1=d/math.sqrt(n)*(y1@x1.T)
    opt=np.linalg.solve(s,z1.T).T
    base=excess(z0,opt,s)
    rows=[]
    specs=[('baseline',0,'pre',0.,z0,1.)]
    for T in TS:
        for method in METHODS:
            step=.08
            u=train(method,T,step,sigma,z0,s,z1)
            specs.append((method,T,'fixed',step,u,excess(u,opt,s)/base))
            match=matched(method,T,sigma,z0,s,z1,opt,base)
            if match is None:
                specs.append((method,T,'matched',None,None,None))
            else:
                step,u,rel=match
                specs.append((method,T,'matched',step,u,rel))
    for method,T,mode,step,u,rel in specs:
        row=dict(d=d,n=n,case=case,rx=rx,ry=ry,r=r,seed=seed,entropy=ent,
            method=method,T=T,mode=mode,step=step,relative_loss=rel,base_excess=base)
        if u is None:
            row['missing']='target not bracketed'; rows.append(row); continue
        diff=u-z0
        row.update(q=float(np.mean(u*u)),cross=float(np.mean(z0*diff)),v=float(np.mean(diff*diff)))
        row['old']=measurements(u,x0,y0,ids)
        row['new']=measurements(u,x1,y1,ids)
        if method in COORD:
            pred=moments(method,T,step,r)
            proxy=z0+delta(method,T,step,sigma,(z1-z0)/sigma)
            row.update(pred=pred,proxy_mse=float(np.mean((u-proxy)**2)))
        if smoke:
            # Independently form dense scores at d=12 only.
            for name,x,y in [('old',x0,y0),('new',x1,y1)]:
                scores=y.T@u@x; dd=np.diag(scores)
                wm=(scores.sum(axis=0)-dd)/(n-1)
                vv=(np.sum(scores*scores,axis=0)-dd*dd)/(n-1)-wm*wm
                assert abs(vv.mean()-row[name]['variance'])<1e-10
                cnt=(scores[:,ids]>dd[ids]).sum(axis=0)
                # Explicit correct candidate removal before rank audit.
                for j,idx in enumerate(ids): cnt[j]-=int(scores[idx,idx]>dd[idx])
                assert np.array_equal(cnt+1,row[name]['ranks'])
            if method=='GD':
                mat=np.linalg.matrix_power(np.eye(d)-step*s,T)
                exact=z0@mat+opt@(np.eye(d)-mat)
                assert np.max(abs(exact-u))<1e-10
            if method.startswith('Adam') and T==1 and mode=='fixed':
                res=z0@s-z1; e=.1 if '0.1' in method else 1.
                original=z0/math.sqrt(n)
                if method.startswith('AdamW'): original*=1-step*sigma*.02
                original-=step*sigma/math.sqrt(n)*(res/math.sqrt(n))/(abs(res)/math.sqrt(n)+e*sigma/math.sqrt(n))
                assert np.max(abs(original*math.sqrt(n)-u))<1e-10
        rows.append(row)
    return rows

def run(mode):
    freeze_check()
    path=OUT/f'FT_{mode}_rows.jsonl'
    assert not path.exists()
    tasks=[(12,i,0,99) for i in [0,2,3]] if mode=='smoke' else [(d,i,se,10) for se in range(NS) for d in DS for i in range(len(CASES))]
    if mode=='formal':
        assert json.loads((OUT/'FT_smoke_meta.json').read_text())['passed']
    completed=0
    with path.open('x') as f:
        for d,ic,se,tag in tasks:
            if time.monotonic()-START>3000: break
            rows=configuration(d,ic,se,tag,mode=='smoke')
            for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
            f.flush(); completed+=1
            print(json.dumps(dict(mode=mode,d=d,case=CASES[ic][0],seed=se,completed=completed,
                seconds=time.monotonic()-START)),flush=True)
    freeze_check()
    dump(OUT/f'FT_{mode}_meta.json',metadata(mode,dict(completed=completed,expected=len(tasks),
        passed=completed==len(tasks),records=completed*25)))

def analyze():
    freeze_check()
    rows=[json.loads(s) for s in (OUT/'FT_formal_rows.jsonl').read_text().splitlines()]
    groups={}
    for row in rows:
        key=(row['d'],row['case'],row['method'],row['T'],row['mode'])
        groups.setdefault(key,[]).append(row)
    gates=[]; summaries=[]; contrasts=[]
    def stat(vals):
        vals=np.asarray(vals,dtype=float)
        return float(vals.mean()),float(vals.std(ddof=1)/math.sqrt(len(vals)))
    def gate(group,key,vals,pred,tol,count):
        mn,se=stat(vals); crit=float(student.ppf(1-.05/(2*count),len(vals)-1))
        bound=abs(mn-pred)+crit*se
        gates.append(dict(gate=group,key=str(key),mean=mn,se=se,pred=pred,tolerance=tol,bound=bound,
            passed=bool(len(vals)==NS and bound<=tol)))
    n_g1=2*5*4*2*2*4
    n_g2=2*5*4*2*2*3
    for key,rs in sorted(groups.items()):
        d,case,method,T,mode=key
        ok=[r for r in rs if 'missing' not in r]
        if len(ok)!=NS:
            summaries.append(dict(key=str(key),missing=NS-len(ok))); continue
        entry=dict(d=d,case=case,method=method,T=T,mode=mode,n=len(rs))
        for field in ['step','relative_loss','q','cross','v']:
            mn,se=stat([r[field] for r in ok]); entry[field]=dict(mean=mn,se=se)
        for task in ['old','new']:
            entry[task]={}
            for field in ['margin','variance','correct_mean','correct_variance','msr','rank','top10']:
                mn,se=stat([r[task][field] for r in ok]); entry[task][field]=dict(mean=mn,se=se)
        summaries.append(entry)
        if method in COORD:
            # Paired per-seed predictions use the observed NEW-loss selected step,
            # the map from that step to predictions was frozen before new data.
            for name,tol in [('q',.06),('cross',.04),('old_margin',.06),('old_variance',.08)]:
                vals=[]
                for row in ok:
                    obs=row[name] if name in ['q','cross'] else row['old'][name[4:]]
                    target=row['pred'][name] if name in ['q','cross'] else row['pred']['a0' if name=='old_margin' else 'q']
                    vals.append(obs-target)
                gate('G1',key+(name,),vals,0.,tol,n_g1)
            for task in ['old','new']:
                for name,tol in [('msr',.07),('rank',.04),('top10',.06)]:
                    vals=[row[task][name]-row['pred'][task+'_'+name] for row in ok]
                    gate('G2',key+(task,name),vals,0.,tol,n_g2*2)
    # G3 cross term: predeclared independent task, matched T4 d128 GD.
    key=(128,'independent','GD',4,'matched')
    if key in groups:
        rr=groups[key]
        mn,se=stat([r['cross'] for r in rr]); lower=mn+float(student.ppf(.99,NS-1))*se
        err,se2=stat([r['old']['variance']-(1+r['v']) for r in rr])
        upper=err+float(student.ppf(.99,NS-1))*se2
        gates.append(dict(gate='G3',key='negative_cross_and_reject_independent_noise',
            cross=mn,cross_upper99=lower,variance_error=err,error_upper99=upper,
            passed=bool(len(rr)==NS and lower<-.05 and upper<-.1)))
    # G4 is matching coverage and precision, not a directional finding.
    matched_rows=[r for r in rows if r['mode']=='matched']
    failures=[(r['d'],r['case'],r['seed'],r['method'],r['T']) for r in matched_rows
        if 'missing' in r or abs(r['relative_loss']-TARGET)>.002]
    gates.append(dict(gate='G4',key='matched_excess_loss',passed=len(matched_rows)==2400 and not failures,
        count=len(matched_rows),failures=failures))
    # G5 four T1 analytic order contrasts, old MSR at matched loss, d128 independent.
    order=[('GD','Adam_e1'),('Adam_e1','Adam_e0.1'),('Adam_e0.1','Sign')]
    for T in TS:
        pairs=order if T==1 else [('GD','Sign'),('GD','Adam_e0.1'),('GD','Adam_e1')]
        for aa,bb in pairs:
            ra={r['seed']:r for r in groups[(128,'independent',aa,T,'matched')] if 'missing' not in r}
            rb={r['seed']:r for r in groups[(128,'independent',bb,T,'matched')] if 'missing' not in r}
            seeds=sorted(set(ra)&set(rb))
            vals=[ra[s]['old']['msr']-rb[s]['old']['msr'] for s in seeds]
            mn,se=stat(vals); lower=mn-float(student.ppf(1-.05/6,len(seeds)-1))*se
            gates.append(dict(gate='G5',key=str((T,aa,bb)),mean=mn,se=se,lower=lower,
                passed=len(seeds)==NS and lower>0))
    # Complete paired contrast table, including negative and unresolved cases.
    for d in DS:
        for case,_,_ in CASES:
            for T in TS:
                for mode in ['fixed','matched']:
                    gd={r['seed']:r for r in groups[(d,case,'GD',T,mode)] if 'missing' not in r}
                    for method in METHODS[1:]:
                        other={r['seed']:r for r in groups[(d,case,method,T,mode)] if 'missing' not in r}
                        seeds=sorted(set(gd)&set(other))
                        for task in ['old','new']:
                            for field in ['msr','rank','top10']:
                                vals=[other[s][task][field]-gd[s][task][field] for s in seeds]
                                mn,se=stat(vals); crit=float(student.ppf(.975,len(seeds)-1))
                                contrasts.append(dict(d=d,case=case,T=T,mode=mode,method=method,
                                    task=task,metric=field,mean=mn,se=se,lo=mn-crit*se,hi=mn+crit*se,n=len(seeds)))
    expected=2*5*20*25
    complete=len(rows)==expected and len(groups)==2*5*25 and all(len(rs)==NS for rs in groups.values())
    formal=json.loads((OUT/'FT_formal_meta.json').read_text())
    gates.insert(0,dict(gate='G0',key='integrity_coverage_resources',passed=complete and formal['passed'],
        records=len(rows),expected=expected,groups=len(groups)))
    dump(OUT/'FT_gates.json',gates)
    dump(OUT/'FT_summary.json',summaries)
    with (OUT/'FT_contrasts.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(contrasts[0])); w.writeheader(); w.writerows(contrasts)
    counts={g:dict(passed=sum(v['passed'] for v in gates if v['gate']==g),total=sum(v['gate']==g for v in gates)) for g in sorted({v['gate'] for v in gates})}
    dump(OUT/'FT_analysis_meta.json',metadata('analyze',dict(counts=counts)))
    print(json.dumps(counts),flush=True)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare': prepare()
    elif mode in ['smoke','formal']: run(mode)
    elif mode=='analyze': analyze()
    else: raise ValueError(mode)
