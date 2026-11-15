"""Review22: independent full-array and algebra checks. No reviewed code imported."""
import os, json, time, resource, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.integrate import quad
from scipy.stats import binom, norm

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
rng = np.random.default_rng(22061107)
start = time.monotonic()

def train(S, Z, method):
    W = np.zeros(Z.shape)
    first = np.zeros(Z.shape)
    second = np.zeros(Z.shape)
    for step in range(3 if method == 'GD3' else 2):
        descent = Z - W @ S
        if method == 'GD3':
            W = W + 0.7 * descent
        elif method == 'Sign2':
            W = W + 0.6 * np.sign(descent)
        else:
            first = 0.9 * first + 0.1 * descent
            second = 0.99 * second + 0.01 * descent**2
            W = W + 0.3 * (first / (1 - 0.9**(step+1))) / (1 + np.sqrt(second / (1 - 0.99**(step+1))))
    return W

def scalar_adam(x):
    w = m = v = 0.0
    for t in (1, 2):
        grad = w - x
        m = .9*m + .1*grad
        v = .99*v + .01*grad*grad
        w -= .3*(m/(1-.9**t))/(1+np.sqrt(v/(1-.99**t)))
    return w

report = {}
# Independent quadrature / closed form, not the reviewed integrand implementation.
aa = quad(lambda x: 2*norm.pdf(x)*x*scalar_adam(x), 0, np.inf, epsabs=1e-12)[0]
qq = quad(lambda x: 2*norm.pdf(x)*scalar_adam(x)**2, 0, np.inf, epsabs=1e-12)[0]
report['moments'] = {'GD3': 1.0, 'Sign2': 2*1.2*norm.pdf(.6)/np.sqrt(2*1.2**2*norm.sf(.6)), 'Adam2': aa/np.sqrt(qq)}

raw = [json.loads(s) for s in (PROJECT/'ac1_checks/serial/AC1_groups.jsonl').read_text().splitlines()]
old = json.loads((PROJECT/'ac1_checks/serial/AC1_simulation.json').read_text())
groups = defaultdict(list)
for r in raw:
    groups[(r['d'],r['method'],r['s'])].append(r)
maxmean = maxse = 0.0
for r in old['rows']:
    rr = groups[(r['d'],r['method'],r['s'])]
    assert len(rr)==128 and len({v['group'] for v in rr})==128
    for field in ('p','sphere_gauss','shift_pred','residual_smooth','T_minus_b'):
        values = np.array([v[field] for v in rr])
        maxmean = max(maxmean, float(np.max(abs(values.mean(0)-r[field]))))
        maxse = max(maxse, float(np.max(abs(values.std(0,ddof=1)/np.sqrt(len(rr))-r[field+'_SE']))))
report['aggregation'] = {'records':len(raw),'cells':len(groups),'max_mean_error':maxmean,'max_group_SE_error':maxse}

# All scores, D and V from actual full X,Y; compare both matrix orientations.
algebra = []
for method in ('GD3','Sign2','Adam2'):
    d,n = 8,90
    X = rng.normal(size=(d,n))/np.sqrt(d)
    Y = rng.normal(size=(d,n))/np.sqrt(d)
    S = d/n*(X@X.T)
    Z = d/np.sqrt(n)*(Y@X.T)
    W = train(S,Z,method)
    C = d/n*(np.vstack((X,Y))@np.vstack((X,Y)).T)
    F = np.linalg.cholesky(C)
    Q = np.sqrt(d/n)*np.linalg.solve(F,np.vstack((X,Y)))
    Lx,Ly = F[:d],F[d:]
    A = Ly.T@W@Lx
    u = Q[:,0]
    scores = d/np.linalg.norm(W)*(Y.T@W@X[:,0])
    D = n*(u@A@u)/np.linalg.norm(W)
    vtrue = np.mean(scores[1:]**2)
    vforward = (n*n*np.linalg.norm(A@u)**2/np.linalg.norm(W)**2-D*D)/(n-1)
    vreverse = (n*n*np.linalg.norm(A.T@u)**2/np.linalg.norm(W)**2-D*D)/(n-1)
    algebra.append({'method':method,'D_error':float(abs(D-scores[0])), 'V_direct':float(vtrue), 'V_Au':float(vforward),'V_Atranspose_u':float(vreverse), 'V_correct_error':float(abs(vtrue-vforward)), 'V_transpose_error':float(abs(vtrue-vreverse))})
report['orientation'] = algebra

# Exact marginal tail probabilities: no extreme-value sampling approximation.
tails=[]
for m in (100,10000,1000000,100000000):
    b=norm.isf(1/m)
    for L in (1,5):
        quantiles=[]
        for p in (.1,.5,.9):
            lo,hi=-10.,10.
            for _ in range(70):
                z=(lo+hi)/2
                cdf=binom.cdf(L-1,m,norm.sf(b+z/b))
                if cdf<p: lo=z
                else: hi=z
            quantiles.append((lo+hi)/2)
        tails.append({'m':m,'L':L,'b':float(b),'quantiles_b_H_minus_b':quantiles,'b2_over_sqrt_m':float(b*b/np.sqrt(m))})
report['extreme_quantiles']=tails

# Actual full training arrays and all candidate rankings: no residual roots,
# no Bartlett sampling, no Stiefel threshold sampler.
direct=[]
for d,B,Qn in ((64,512,32),(128,256,24),(256,80,20)):
    for method in ('GD3','Sign2','Adam2'):
        base=next(r for r in old['rows'] if r['d']==d and r['method']==method and r['s']==0)
        n=base['n']
        vv=[]
        for rep in range(B):
            X=rng.normal(size=(d,n))/np.sqrt(d)
            Y=rng.normal(size=(d,n))/np.sqrt(d)
            W=train(d/n*(X@X.T),d/np.sqrt(n)*(Y@X.T),method)
            ids=rng.choice(n,size=Qn,replace=False)
            scores=Y.T@(W@X[:,ids])
            diag=scores[ids,np.arange(Qn)]
            rank=1+(scores>diag).sum(axis=0)
            vv.append([(rank<=L).mean() for L in (1,5)])
        vv=np.array(vv)
        se=vv.std(0,ddof=1)/np.sqrt(B)
        diff=vv.mean(0)-base['p']
        z=diff/np.sqrt(se**2+np.array(base['p_SE'])**2)
        result={'d':d,'method':method,'n':n,'B':B,'queries':Qn,'direct_mean':vv.mean(0).tolist(),'direct_dataset_SE':se.tolist(),'difference_from_AC1':diff.tolist(),'difference_z':z.tolist()}
        direct.append(result)
        print(json.dumps(result),flush=True)
report['full_array_direct']=direct

# Random finite-history checks, including beta2=0 and beta1 > sqrt(beta2).
history=[]
for beta1,beta2 in ((.9,.99),(.99,0.),(0.,0.),(.99,.01)):
    alpha,e,T=.3,1.,6
    g=rng.normal(size=3000)*20
    w=np.zeros_like(g)
    refs=[]
    maxrat=0.
    for t in range(1,T+1):
        refs.append(w-g)
        s=np.array(refs)
        weights1=(1-beta1)*beta1**np.arange(t-1,-1,-1)/(1-beta1**t)
        weights2=(1-beta2)*beta2**np.arange(t-1,-1,-1)/(1-beta2**t)
        r=s+rng.normal(size=s.shape)*.3
        ms=weights1@s; us=np.sqrt(weights2@(s*s))
        mr=weights1@r; ur=np.sqrt(weights2@(r*r))
        J=(1+2*alpha/e)**(t-1)/np.sqrt(1-beta2)
        L=(1+J)/e
        ratio=abs(mr/(ur+e)-ms/(us+e))/(L*np.linalg.norm(r-s,axis=0))
        maxrat=max(maxrat,float(ratio.max()))
        w-=alpha*ms/(us+e)
    history.append({'beta1':beta1,'beta2':beta2,'max_lipschitz_bound_ratio':maxrat})
report['adam_history']=history
report['input_hashes_unchanged']={}
for line in (ROOT/'input_hashes.sha256').read_text().splitlines():
    digest,name=line.split('  ',1)
    report['input_hashes_unchanged'][name]=(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==digest)
cg=Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
cp=Path('/sys/fs/cgroup')/cg.lstrip('/')
report['resources']={'wall_seconds':time.monotonic()-start,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'threads':next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Threads:'))}
for key in ('memory.max','memory.swap.max','memory.peak'):
    report['resources'][key]=(cp/key).read_text().strip()
with (ROOT/'independent_results.json').open('x') as f:
    json.dump(report,f,indent=2,ensure_ascii=False)
print(json.dumps({'aggregation':report['aggregation'],'orientation':algebra,'resources':report['resources']}),flush=True)
