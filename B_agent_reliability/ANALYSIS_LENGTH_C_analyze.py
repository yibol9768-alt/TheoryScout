"""Offline raw-trace length diagnostics. No runner imports, network or model calls."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import json, math, hashlib, csv, collections, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/real/analysis_length'
SEED=640811
rng=np.random.default_rng(SEED)
SOURCES={}
def read(p):
    data=p.read_bytes(); SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(data).hexdigest(); return data.decode()
def oracle(s,op):
    s=s.copy()
    for x in op['instructions']:
        a,b,c=[f'r{x[z]}' for z in ('a','b','c')]
        s[a]=(s[a]+x['m']*s[b]-s[c]+x['d'])%997
        s[c]=(s[c]+s[a])%997
    s['F0']=(s['r0']+2*s['r1']-s['r2'])%997
    s['F1']=(3*s['F0']+s['r3']-s['r4'])%997
    s['F2']=(s['F1']+2*s['r5'])%997
    return s
TASKS=[]; ACTIONS=[]; STEPS=[]; checks=0
for batch in ('calib1','holdout_b','holdout_c'):
    folder=ROOT/'experiments/real'/f'{batch}_real'
    summary=json.loads(read(folder/'summary.json'))
    for result in summary['results']:
        ev=[json.loads(x) for x in read(folder/(result['id']+'.jsonl')).splitlines()]
        t=ev[0]['task']; assert t['seed']==result['seed']; s=t['initial'].copy(); prefix=[s.copy()]
        for op in t['operations']: s=oracle(s,op); prefix.append(s)
        assert prefix==t['prefix']
        req={e['call']:e for e in ev if e['event']=='request'}
        resp={e['call']:e for e in ev if e['event']=='response'}
        acts=[e for e in ev if e['event']=='action']; actual=t['initial'].copy(); expstep=0; exprev=0
        prevbad=0; prior_errors=0; firsterr=None; local=[]
        for a in acts:
            st,rv=a['step'],a['review']; assert (st,rv)==(expstep,exprev)
            assert a['before']==actual and a['expected']==prefix[st+1]
            if rv==0: assert actual==prefix[st]
            parsed=json.loads(a['content']) if a['action'] else None
            assert parsed==a['action']
            if parsed: actual.update(parsed['writes'])
            wrong=sum(actual.get(k)!=v for k,v in prefix[st+1].items())+int(parsed is None)
            assert actual==a['after'] and wrong==a['n_after'] and a['correct']==(wrong==0)
            assert a['n_before']==(0 if rv==0 else sum(a['before'].get(k)!=v for k,v in prefix[st+1].items()))
            request=req[a['call']]; response=resp[a['call']]['response']
            assert len(request['messages'])==(2 if rv==0 else 4)
            assert a['content']==response['choices'][0]['message']['content']
            assert not any('length' in json.loads(m['content']) for m in request['messages'] if m['role']=='user')
            row=dict(batch=batch,id=result['id'],length=t['length'],step=st,review=rv,correct=int(a['correct']),
                     error=int(not a['correct']),detect=int(bool(parsed and parsed['detect'])),prior_errors=prior_errors,
                     previous_error=prevbad,n_before=a['n_before'],n_after=a['n_after'],
                     prompt_tokens=response['usage']['prompt_tokens'],prompt_chars=sum(len(m['content']) for m in request['messages']),
                     checkpoint_F=int('F0' in prefix[st]),numeric_error=int(a['error_type']=='numeric'))
            ACTIONS.append(row); local.append(row)
            if rv==0:
                episode=dict(row,fatal=0,resolved=1,r1=None,r2=None,error_fields='|'.join(k for k,v in prefix[st+1].items() if actual.get(k)!=v))
                STEPS.append(episode)
                prevbad=int(not a['correct'])
                if prevbad:
                    if firsterr is None: firsterr=st
                    prior_errors+=1
            else:
                episode[f'r{rv}']=int(a['correct'])
            if a['correct']: expstep+=1; exprev=0
            else: exprev+=1
            checks+=1
        assert ev[-1]['event']=='result' and ev[-1]['result']==result
        if result['status']=='unrepaired_after_2_reviews': assert exprev==3; episode['fatal']=1
        if result['status']=='success': assert expstep==t['length']
        if exprev and exprev<3: episode['resolved']=0
        advances=[a for a in local if a['review']==0]
        assert len(advances)==result['advances'] and sum(a['error'] for a in advances)==result['seed_errors']
        TASKS.append(dict(batch=batch,id=result['id'],length=t['length'],status=result['status'],n=len(advances),
                          errors=sum(a['error'] for a in advances),d=int(result['status']=='unrepaired_after_2_reviews'),
                          success=int(result['status']=='success'),first_error=firsterr,tail_n=sum(a['step']>0 for a in advances),
                          tail_d=int(result['status']=='unrepaired_after_2_reviews' and advances[-1]['step']>0),
                          fatal_step=advances[-1]['step'] if result['status']=='unrepaired_after_2_reviews' else None,
                          censor=int(result['status'] not in ('success','unrepaired_after_2_reviews'))))
for name,rows in [('tasks',TASKS),('actions',ACTIONS),('steps',STEPS)]:
    with (OUT/f'ANALYSIS_LENGTH_C_{name}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def rate(rows,field):
    n=len(rows); k=sum(x[field] for x in rows); return dict(k=k,n=n,rate=k/n if n else None)
def metrics(steps,actions,tasks):
    blocks=[]
    for lo,hi in [(0,1),(1,8),(8,16),(16,32),(32,48),(48,64)]:
        ss=[s for s in steps if lo<=s['step']<hi]; aa=[a for a in actions if lo<=a['step']<hi]
        blocks.append(dict(lo=lo,hi=hi,error=rate(ss,'error'),fatal=rate([s for s in ss if s['resolved']],'fatal'),
                           r1=rate([a for a in aa if a['review']==1],'correct'),r2=rate([a for a in aa if a['review']==2],'correct')))
    hist={}
    for rv in (0,1,2):
        for name,fn in [('no_prior',lambda a:a['prior_errors']==(int(a['review']>0))),
                        ('prior',lambda a:a['prior_errors']>int(a['review']>0)),
                        ('prev_clean',lambda a:not a['previous_error']),('prev_error',lambda a:bool(a['previous_error']))]:
            rows=[a for a in actions if a['review']==rv and fn(a) and (a['step']>0 or name.startswith('no_') or name=='prior')]
            hist[f'{rv}_{name}']=rate(rows,'error' if rv==0 else 'correct')
    dep={}
    for rv in (1,2):
        rows=[a for a in actions if a['review']==rv]
        table=[[sum(a['detect']==d and a['correct']==c for a in rows) for c in (0,1)] for d in (0,1)]
        dep[str(rv)]=dict(table_detect_by_correct=table,fisher_p=fisher(table))
    toks={}
    for rv in (0,1,2):
        rows=[a for a in actions if a['review']==rv]
        if rows:
            toks[str(rv)]={k:[float(x) for x in np.percentile([a[k] for a in rows],[0,25,50,75,100])] for k in ('prompt_tokens','prompt_chars')}
    return dict(tasks=len(tasks),status=dict(collections.Counter(t['status'] for t in tasks)),blocks=blocks,history=hist,
                detection_repair=dep,tokens=toks,fatal_steps=[t['fatal_step'] for t in tasks if t['d']],
                first_error_steps=[t['first_error'] for t in tasks if t['first_error'] is not None])
def fisher(tab):
    a,b=tab[0];c,d=tab[1]; n=a+b+c+d; r=a+b;col=a+c
    if not n: return None
    def prob(x): return math.comb(col,x)*math.comb(n-col,r-x)/math.comb(n,r)
    observed=prob(a)
    return min(1.,sum(prob(x) for x in range(max(0,r-(n-col)),min(r,col)+1) if prob(x)<=observed*(1+1e-10)))
DIAG={}
for batch in ('calib1','holdout_b','holdout_c','all'):
    sel=lambda x:batch=='all' or x['batch']==batch
    DIAG[batch]=metrics([s for s in STEPS if sel(s)],[a for a in ACTIONS if sel(a)],[t for t in TASKS if sel(t)])
print('RAW',checks,len(TASKS),len(STEPS),flush=True)
(OUT/'ANALYSIS_LENGTH_C_raw_diagnostics.json').write_text(json.dumps(DIAG,ensure_ascii=False,indent=2)+'\n')
(OUT/'ANALYSIS_LENGTH_C_inputs_sha256.json').write_text(json.dumps(SOURCES,indent=2)+'\n')

# Small deterministic optimizer, avoiding external/scipy dependencies.
def nelder(fun,x,step=.7,maxiter=240,tol=1e-7):
    x=np.asarray(x,float); dim=len(x); simplex=np.tile(x,(dim+1,1))
    for j in range(dim): simplex[j+1,j]+=step
    vals=np.array([fun(v) for v in simplex]); it=0
    for it in range(maxiter):
        order=np.argsort(vals); simplex=simplex[order];vals=vals[order]
        if np.max(np.abs(simplex[1:]-simplex[0]))<tol: break
        mid=simplex[:-1].mean(0); xr=mid+(mid-simplex[-1]); fr=fun(xr)
        if fr<vals[0]:
            xe=mid+2*(xr-mid);fe=fun(xe)
            simplex[-1],vals[-1]=(xe,fe) if fe<fr else (xr,fr)
        elif fr<vals[-2]: simplex[-1],vals[-1]=xr,fr
        else:
            xc=mid+.5*((xr if fr<vals[-1] else simplex[-1])-mid);fc=fun(xc)
            if fc<min(fr,vals[-1]):simplex[-1],vals[-1]=xc,fc
            else:
                simplex[1:]=simplex[0]+.5*(simplex[1:]-simplex[0]);vals[1:]=[fun(v) for v in simplex[1:]]
    j=np.argmin(vals);return simplex[j],float(vals[j]),it+1

def xlog(x,p):
    if x==0:return 0.
    if p<=0:return -math.inf
    return x*math.log(p)
def fit_beta(n,e,starts=True):
    n=np.asarray(n,int);e=np.asarray(e,int);m=e.sum()/n.sum() if n.sum() else 0
    if not 0<m<1: return dict(mean=float(m),k=None,rho=0.,alpha=None,beta=None,boundary=True,ll=0.)
    maxn=int(max(n));grid=np.arange(maxn,dtype=float)
    def loglike(mean,k):
        a=mean*k;b=(1-mean)*k
        ca=np.r_[0.,np.cumsum(np.log(a+grid))];cb=np.r_[0.,np.cumsum(np.log(b+grid))]
        ck=np.r_[0.,np.cumsum(np.log(k+grid))]
        return float(np.sum(ca[e]+cb[n-e]-ck[n]))
    hom=xlog(int(e.sum()),m)+xlog(int((n-e).sum()),1-m)
    def obj(x):
        if not(-15<x[0]<6 and -6<x[1]<16):return 1e20+float(np.sum(x*x))
        return -loglike(1/(1+math.exp(-x[0])),math.exp(x[1]))
    solutions=[nelder(obj,[math.log(m/(1-m)),v],maxiter=180) for v in ([0,2,5,10] if starts else [3,11])]
    xx,loss,it=min(solutions,key=lambda z:z[1]);mean=1/(1+math.exp(-xx[0]));k=math.exp(xx[1])
    if -loss<=hom+1e-6 or k>1e6:
        return dict(mean=float(m),k=None,rho=0.,alpha=None,beta=None,boundary=True,ll=hom)
    return dict(mean=mean,k=k,rho=1/(k+1),alpha=mean*k,beta=(1-mean)*k,boundary=False,ll=-loss)

def beta_ll(n,e,b):
    if b['boundary']:
        return sum(xlog(int(y),b['mean'])+xlog(int(x-y),1-b['mean']) for x,y in zip(n,e))
    a=b['alpha'];bb=b['beta'];k=a+bb
    return sum(math.lgamma(a+y)+math.lgamma(bb+x-y)-math.lgamma(k+x)-math.lgamma(a)-math.lgamma(bb)+math.lgamma(k) for x,y in zip(n,e))

def beta_surv(b,L):
    if b['boundary']:return (1-b['mean'])**L
    return math.exp(sum(math.log1p(-b['alpha']/(b['k']+j)) for j in range(L)))

cal=[t for t in TASKS if t['batch']=='calib1']
cs=[s for s in STEPS if s['batch']=='calib1'];ca=[a for a in ACTIONS if a['batch']=='calib1']
LENGTHS=[8,16,32,64,128]
# The p-mixture models heterogeneity of natural mistakes, and holds repair rates shared.
quadx,quadw=np.polynomial.legendre.leggauss(192);quadp=(quadx+1)/2;quadw=quadw/2

def get_counts(tasks,steps):
    # CALIB has no unresolved service episode; exact telescoping is valid here.
    return np.array([len(steps),sum(s['error'] for s in steps),sum(s['r1']==1 for s in steps),
                     sum(s['r1'] is not None for s in steps),sum(s['r2']==1 for s in steps),
                     sum(s['r2'] is not None for s in steps)],float)
def fit_models(tasks,steps,fast=False):
    n=np.array([t['n'] for t in tasks]);d=np.array([t['d'] for t in tasks]);e=np.array([t['errors'] for t in tasks])
    counts=get_counts(tasks,steps);N,E,R1,N1,R2,N2=counts
    p=E/N;r1=R1/N1 if N1 else 1.;r2=R2/N2 if N2 else 1.;f=(1-r1)*(1-r2);h=p*f
    out={'A':dict(h=h,p=p,r1=r1,r2=r2,f=f), 'B_fatal':fit_beta(n,d,not fast),
         'B_error':dict(beta=fit_beta(n,e,not fast),p=p,r1=r1,r2=r2,f=f)}
    for cut in (1,8):
        eN=sum(min(t['n'],cut) for t in tasks);eD=sum(t['d'] and t['fatal_step']<cut for t in tasks)
        tN=sum(max(0,t['n']-cut) for t in tasks);tD=sum(t['d'] and t['fatal_step']>=cut for t in tasks)
        h0=eD/eN if eN else 0.;hp=tD/tN if tN else 0.
        out[f'C{cut}']=dict(cut=cut,h0=h0,hplus=hp,early_n=eN,early_d=int(eD),tail_n=tN,tail_d=int(tD))
        tn=np.array([max(0,t['n']-cut) for t in tasks]);td=np.array([int(t['d'] and t['fatal_step']>=cut) for t in tasks])
        out[f'D{cut}']=dict(cut=cut,h0=h0,tail=fit_beta(tn,td,not fast))
    return out

def pred(model,param,L):
    if model=='A':return (1-param['h'])**L
    if model=='B_fatal':return beta_surv(param,L)
    if model=='B_error':
        b=param['beta'];f=param['f']
        if b['boundary']:return (1-b['mean']*f)**L
        lw=np.log(quadw)+(b['alpha']-1)*np.log(quadp)+(b['beta']-1)*np.log1p(-quadp)
        w=np.exp(lw-lw.max());w/=w.sum()
        return float(w@((1-f*quadp)**L))
    cut=param['cut'];surv=(1-param['h0'])**min(cut,L)
    return surv*((1-param['hplus'])**max(0,L-cut) if model[0]=='C' else beta_surv(param['tail'],max(0,L-cut)))

def traj_ll(model,param,tasks):
    # Marginal likelihood for observed terminal time or right-censored exposure.
    total=0.
    for t in tasks:
        # An incomplete repair episode cannot be labelled survived; exclude that final step.
        n=t['n']
        if t['censor']:
            ss=[s for s in STEPS if s['id']==t['id'] and s['batch']==t['batch']]
            if ss and not ss[-1]['resolved']:n-=1
        s=pred(model,param,n)
        prob=pred(model,param,n-1)-s if t['d'] else s
        total+=math.log(max(prob,1e-300))
    return total

def endpoint_score(model,param,tasks):
    ts=[t for t in tasks if not t['censor']]
    terms=[];brier=[]
    for t in ts:
        p=pred(model,param,t['length']);y=t['success']
        terms.append(xlog(y,p)+xlog(1-y,1-p));brier.append((y-p)**2)
    return dict(n=len(ts),successes=sum(t['success'] for t in ts),ll=sum(terms),logloss=-sum(terms)/len(ts) if ts else None,
                brier=sum(brier)/len(ts) if ts else None)
MODELS=fit_models(cal,cs)
EVAL={}
for name,param in MODELS.items():
    scores={}
    for batch in ('calib1','holdout_b','holdout_c','holdout_bc'):
        ts=[t for t in TASKS if (t['batch']!='calib1' if batch=='holdout_bc' else t['batch']==batch)]
        scores[batch]=dict(trajectory_ll=traj_ll(name,param,ts),endpoint=endpoint_score(name,param,ts),
                          by_length={str(L):endpoint_score(name,param,[t for t in ts if t['length']==L]) for L in sorted({t['length'] for t in ts})})
    EVAL[name]=dict(parameters=param,predictions={str(L):pred(name,param,L) for L in LENGTHS},scores=scores)
print('FITS',json.dumps({m:dict(p=x['parameters'],s=x['predictions'],hold_ll=x['scores']['holdout_bc']['trajectory_ll']) for m,x in EVAL.items()}),flush=True)
(OUT/'ANALYSIS_LENGTH_C_models.json').write_text(json.dumps(EVAL,indent=2)+'\n')

# Cluster-robust logistic associations; exploratory, not causal or multiplicity adjusted.
def logistic(rows,outcome,features):
    if len(rows)<15 or len({r[outcome] for r in rows})<2:return dict(n=len(rows),insufficient=True)
    X=np.array([[1.]+[fn(r) for _,fn in features] for r in rows]);y=np.array([r[outcome] for r in rows],float)
    names=['intercept']+[name for name,_ in features];beta=np.zeros(X.shape[1])
    for _ in range(70):
        eta=np.clip(X@beta,-30,30);p=1/(1+np.exp(-eta));H=X.T@((p*(1-p))[:,None]*X)
        update=np.linalg.pinv(H)@(X.T@(y-p));beta+=update
        if np.max(np.abs(update))<1e-8:break
    meat=np.zeros_like(H);groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[(r['batch'],r['id'])].append(i)
    for ids in groups.values():
        u=X[ids].T@(y[ids]-p[ids]);meat+=np.outer(u,u)
    bread=np.linalg.pinv(H);cov=bread@meat@bread
    G=len(groups);cov*=G/(G-1)*(len(rows)-1)/(len(rows)-X.shape[1]) if G>1 else 1
    se=np.sqrt(np.maximum(0,np.diag(cov)))
    return dict(n=len(rows),clusters=G,coefficients={name:dict(beta=float(b),se_cluster=float(s),OR=math.exp(float(np.clip(b,-50,50))),
               CI95_OR=[math.exp(float(np.clip(b-1.96*s,-50,50))),math.exp(float(np.clip(b+1.96*s,-50,50)))],
               p_normal=math.erfc(abs(b/s)/math.sqrt(2)) if s else None) for name,b,s in zip(names,beta,se)})
REG={}
for batch in ('calib1','holdout_b','holdout_c','all'):
    for rv in (0,1,2):
        rows=[a for a in ACTIONS if a['review']==rv and (batch=='all' or a['batch']==batch)]
        base=[('log2_step_plus1',lambda a:math.log2(a['step']+1)),('prior_errors',lambda a:a['prior_errors']-int(a['review']>0))]
        if batch=='all':base += [('batch_b',lambda a:int(a['batch']=='holdout_b')),('batch_c',lambda a:int(a['batch']=='holdout_c'))]
        REG[f'{batch}_r{rv}']=logistic(rows,'error' if rv==0 else 'correct',base)
    rows=[a for a in ACTIONS if a['review']==0 and a['step']>0 and (batch=='all' or a['batch']==batch)]
    REG[f'{batch}_context']=logistic(rows,'error',[('log2_step_plus1',lambda a:math.log2(a['step']+1)),('prompt_token_centered',lambda a:a['prompt_tokens']-364)])
(OUT/'ANALYSIS_LENGTH_C_regressions.json').write_text(json.dumps(REG,indent=2)+'\n')

# Simulation null repeats failure-dependent stopping, unlike fixed-n Pearson chi-square.
def dispersion(n,e):
    n=np.array(n);e=np.array(e);keep=n>0;n=n[keep];e=e[keep];p=e.sum()/n.sum()
    return float(np.sum((e-n*p)**2/(n*p*(1-p)))/(len(n)-1))
def hetero_test(tasks,steps,B=10000):
    n0=np.array([t['n'] for t in tasks]);e0=np.array([t['errors'] for t in tasks]);counts=get_counts(tasks,steps)
    N,E,R1,N1,R2,N2=counts;p=E/N;r1=R1/N1;r2=R2/N2;f=(1-r1)*(1-r2)
    # Administrative caps for B service failures are treated as externally fixed sensitivity.
    caps=np.array([t['n'] if t['censor'] else t['length'] for t in tasks]);m=len(tasks)
    alive=np.ones((B,m),bool);nn=np.zeros((B,m),int);ee=np.zeros((B,m),int)
    for j in range(int(max(caps))):
        active=alive & (j<caps)[None,:];bad=(rng.random((B,m))<p)&active
        nn+=active;ee+=bad;alive &= ~(bad & (rng.random((B,m))<f))
    keep=caps>0;nn=nn[:,keep];ee=ee[:,keep];phat=ee.sum(1)/nn.sum(1)
    stat=((ee-nn*phat[:,None])**2/(nn*phat[:,None]*(1-phat[:,None]))).sum(1)/(keep.sum()-1)
    obs=dispersion(n0,e0);beta=fit_beta(n0,e0)
    return dict(Pearson_per_df=obs,simulation_p_upper=(int(sum(stat>=obs))+1)/(B+1),null95=np.quantile(stat,[.025,.975]).tolist(),
                simulation_replicates=B,beta_error_fit=beta,observed_n_range=[int(min(n0)),int(max(n0))],
                caveat='B uses observed service-censor advance cap as fixed; pooled null ignores batch variation')
HET={}
for batch in ('calib1','holdout_b','holdout_c','all'):
    ts=[t for t in TASKS if batch=='all' or t['batch']==batch];ss=[s for s in STEPS if batch=='all' or s['batch']==batch]
    HET[batch]=hetero_test(ts,ss)
print('HET',json.dumps(HET),flush=True)
(OUT/'ANALYSIS_LENGTH_C_heterogeneity.json').write_text(json.dumps(HET,indent=2)+'\n')

# KM curves include B service censoring at last completely resolved advance.
def km(tasks):
    caps=[]
    for t in tasks:
        n=t['n'];last=[s for s in STEPS if s['id']==t['id'] and s['batch']==t['batch']]
        if t['censor'] and last and not last[-1]['resolved']:n-=1
        caps.append((n,t['d']))
    S=1.;rows=[]
    for k in range(1,max(n for n,d in caps)+1):
        risk=sum(n>=k for n,d in caps);death=sum(n==k and d for n,d in caps)
        if risk:S*=1-death/risk
        rows.append(dict(step_1based=k,risk=risk,deaths=death,KM=S))
    return rows
KM={}
POSITION={}
for batch in ('calib1','holdout_b','holdout_c','all'):
    for L in (16,32,64,'all'):
        ts=[t for t in TASKS if (batch=='all' or t['batch']==batch) and (L=='all' or t['length']==L)]
        if ts:
            key=f'{batch}_{L}';KM[key]=km(ts)
            ss=[s for s in STEPS if (batch=='all' or s['batch']==batch) and (L=='all' or s['length']==L)]
            aa=[a for a in ACTIONS if (batch=='all' or a['batch']==batch) and (L=='all' or a['length']==L)]
            POSITION[key]={}
            for label,fn in [('early16',lambda a:a['step']<16),('late16',lambda a:a['step']>=16)]:
                z=[s for s in ss if fn(s)];r=[a for a in aa if fn(a)]
                POSITION[key][label]=dict(error=rate(z,'error'),fatal=rate([s for s in z if s['resolved']],'fatal'),
                                         r1=rate([a for a in r if a['review']==1],'correct'),r2=rate([a for a in r if a['review']==2],'correct'))
(OUT/'ANALYSIS_LENGTH_C_survival.json').write_text(json.dumps(dict(KM=KM,position=POSITION),indent=2)+'\n')

# CALIB-only model uncertainty. Repeated resampling of whole tasks preserves dependencies.
# 1200 refits with optimizer boundary diagnostics, not action-level IID intervals.
BOOT_N=1200
boot={name:{L:[] for L in LENGTHS} for name in MODELS};boundaries=collections.Counter()
boot_parameters=[]
byid={t['id']:[s for s in cs if s['id']==t['id']] for t in cal}
start=time.monotonic()
for b in range(BOOT_N):
    idx=rng.integers(0,len(cal),len(cal));ts=[cal[i] for i in idx];ss=[s for t in ts for s in byid[t['id']]]
    mm=fit_models(ts,ss,fast=True)
    boot_parameters.append(mm)
    for name,par in mm.items():
        for L in LENGTHS:boot[name][L].append(pred(name,par,L))
        if name=='B_fatal' and par['boundary']:boundaries[name]+=1
        if name=='B_error' and par['beta']['boundary']:boundaries[name]+=1
        if name[0]=='D' and par['tail']['boundary']:boundaries[name]+=1
    if (b+1)%200==0:print('BOOT',b+1,round(time.monotonic()-start,1),flush=True)

def mixture_interval(probs,n,alpha=.05):
    p=np.clip(np.array(probs),0,1);pmf=[]
    for k in range(n+1):pmf.append(float(np.mean(math.comb(n,k)*p**k*(1-p)**(n-k))))
    cum=np.cumsum(pmf)
    return [int(np.searchsorted(cum,alpha/2)),int(np.searchsorted(cum,1-alpha/2))]
for name in EVAL:
    EVAL[name]['uncertainty']={str(L):dict(calibration95=np.quantile(boot[name][L],[.025,.975]).tolist(),
                                      predictive_n20_975=mixture_interval(boot[name][L],20,.025)) for L in LENGTHS}
    EVAL[name]['bootstrap']=dict(n=BOOT_N,boundary_fits=boundaries.get(name,0),seed=SEED)
(OUT/'ANALYSIS_LENGTH_C_models.json').write_text(json.dumps(EVAL,indent=2)+'\n')
np.savez_compressed(OUT/'ANALYSIS_LENGTH_C_bootstrap.npz',**{name:np.array([boot[name][L] for L in LENGTHS]).T for name in boot})
(OUT/'ANALYSIS_LENGTH_C_bootstrap_parameters.json').write_text(json.dumps(boot_parameters)+'\n')
print('BOOT_DONE',flush=True)
