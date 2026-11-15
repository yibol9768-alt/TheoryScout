"""Review 13: independent deterministic checks; no project imports or MR draws.

Run with the systemd/resource prefix in PROOF_REVIEW13.md. All outputs exclusive.
Only RNG SeedSequence state expansion is permitted; no training RNG is sampled.
"""
import ast
import csv
import ctypes
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import nct, t

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
START = time.process_time()
resource.setrlimit(resource.RLIMIT_CPU, (240, 250))
assert str(ROOT) == '/home/liuyibo/auto-llm-driving/TheoryScout/E_assoc_memory_overcapacity'
assert sys.executable == '/home/liuyibo/venvs/theory/bin/python'
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    assert os.environ[key] == '2'
cg = next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
CP = Path('/sys/fs/cgroup') / cg.lstrip('/')
assert (CP/'memory.max').read_text().strip() == '3221225472'
assert (CP/'memory.swap.max').read_text().strip() == '0'
pools = []
for libpath in sorted({x.split()[-1] for x in Path('/proc/self/maps').read_text().splitlines() if 'openblas' in x and '.so' in x}):
    lib = ctypes.CDLL(libpath)
    for name in ['scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads', 'openblas_get_num_threads64_', 'openblas_get_num_threads']:
        if hasattr(lib, name):
            fn=getattr(lib,name);fn.restype=ctypes.c_int
            pools.append({'library':libpath,'threads':fn()});break
    else: raise RuntimeError(libpath)
assert pools and all(x['threads']<=2 for x in pools)

def save(name, value):
    with (OUT/name).open('x') as f: json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
def table(name, rows):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def readcsv(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

# Snapshot all pre-existing regular files; active *.log files are recorded separately.
before={}; logs={}; symlinks=[]
for path in sorted(ROOT.rglob('*')):
    if path.is_symlink():symlinks.append(str(path.relative_to(ROOT)));continue
    if not path.is_file() or OUT in path.parents or path.name=='PROOF_REVIEW13.md':continue
    rel=str(path.relative_to(ROOT))
    if path.suffix=='.log':logs[rel]={'size':path.stat().st_size};continue
    before[rel]=sha(path)
save('existing_nonlog_sha256.json',before)
manifest={}
for line in (ROOT/'Y3_FROZEN.sha256').read_text().splitlines():
    h,name=line.split('  ',1);manifest[name]=(sha(ROOT/name)==h)
assert all(manifest.values())

# Recompute scalar response via the physical h(z), then differentiate the variance.
# Adam beta uses E[h'(Z)] (ordinary derivative exists off 0), rather than
# the author's g*h(g) density-polynomial implementation.
pop=json.loads((ROOT/'experiments/x1/X1_population.json').read_text())
z=float(ndtri(.9)); kappa=.25
methods=['GD','Adam2','Adam1','Adam.1','Sign']
pairs=[('GD','Sign'),('Adam2','Adam.1'),('Adam.1','Sign')]
coeff={}; rows=[]; maxcsv=0.; max_block=0.; max_deriv=0.
def responses(method,basevar,var):
    if method=='GD':return np.array([1.,var])
    if method=='Sign':return np.array([math.sqrt(2/math.pi/var),1.])
    eps={'Adam2':2.,'Adam1':1.,'Adam.1':.1}[method]*math.sqrt(basevar)
    def integ(fun):return quad(lambda x:math.sqrt(2/math.pi/var)*math.exp(-x*x/(2*var))*fun(x),0,np.inf,epsabs=2e-12,epsrel=2e-12)[0]
    return np.array([integ(lambda x:eps/(x+eps)**2),integ(lambda x:(x/(x+eps))**2)])
def covariance(method,basevar,var,cov):
    if method=='GD':return cov
    if method=='Sign':return 2/math.pi*math.asin(cov/var)
    eps={'Adam2':2.,'Adam1':1.,'Adam.1':.1}[method]*math.sqrt(basevar)
    rho=cov/var; den=1-rho*rho; scale=math.sqrt(var)
    def inner(a):
        ha=scale*a/(scale*a+eps)
        def fun(b):
            hb=scale*b/(scale*b+eps)
            v1=math.exp(-(a*a+b*b-2*rho*a*b)/(2*den))
            v2=math.exp(-(a*a+b*b+2*rho*a*b)/(2*den))
            return ha*hb*(v1-v2)/(math.pi*math.sqrt(den))
        return quad(fun,0,11,epsabs=2e-11,epsrel=2e-10)[0]
    return quad(inner,0,11,epsabs=2e-11,epsrel=2e-10)[0]
frozen={(int(a['ir']),a['method'],a['task']):a for a in readcsv('y3_checks/Y3_coefficients.csv')}
for p in pop['rows']:
    ir,r,method,alpha=p['ir'],p['r'],p['method'],p['amplitude']
    v0=2*(1-r); beta,Q=responses(method,v0,v0)
    hh=v0*2e-4
    deriv=(responses(method,v0,v0-2*hh)-8*responses(method,v0,v0-hh)+8*responses(method,v0,v0+hh)-responses(method,v0,v0+2*hh))/(12*hh)
    bp,qp=deriv
    q=1-alpha*v0*beta+alpha*alpha*Q
    b=3-2*r
    ell=(1+2*alpha*(beta*(r-3)+(r-1)*b*bp)+alpha**2*(2*v0*beta**2+b*qp))/(kappa*q)
    # Independent exact 2x2 covariance family: tr(E)=diag(E)=0, tr(E^2)/2=delta.
    # This family has the same first-order trace contractions as the Wishart law.
    def exact_block(delta):
        S=np.array([[1.,math.sqrt(delta)],[math.sqrt(delta),1.]])
        K=S@S@S-2*r*S@S+S; J=r*S-S@S
        bb,qq=responses(method,v0,float(K[0,0]))
        off=covariance(method,v0,float(K[0,0]),float(K[0,1]))
        Mh=np.array([[qq,off],[off,qq]])
        MU=S+2*alpha*bb*J+alpha*alpha*Mh
        energy=float(np.trace(S@MU)/2/q)
        means=[]
        for r0,r1 in [(1.,r),(r,1.)]:
            means.append(float(np.trace(r0*S+alpha*bb*(r1*S-r0*S@S))/2/math.sqrt(kappa*q)))
        return np.array(means+[energy])
    base=np.array([(1+alpha*(r-1)*beta)/math.sqrt(kappa*q),(r+alpha*(1-r)*beta)/math.sqrt(kappa*q),1.])
    delta=2e-4
    slope=2*(exact_block(delta/2)-base)/(delta/2)-(exact_block(delta)-base)/delta
    for ti,(task,r0,r1) in enumerate([('old',1.,r),('new',r,1.)]):
        a=r0+alpha*(r1-r0)*beta;mu=a/math.sqrt(kappa*q)
        m=alpha*((r1-r0)*b*bp-r0*beta)/(kappa**1.5*math.sqrt(q))
        w=ell+mu*mu; tt=z-mu
        BR=math.exp(-tt*tt/2)/math.sqrt(2*math.pi)*(m-mu*w/2-mu/2-mu*mu*tt/2)
        c={'mu':mu,'m':m,'ell':ell,'w':w,'q':q,'beta':float(beta),'beta_prime':float(bp),'Q':float(Q),'Q_prime':float(qp),'B_recall':BR,'recall_limit':float(ndtr(mu-z))}
        coeff[ir,method,task]=c
        errors={key:abs(float(frozen[ir,method,task][key])-value) for key,value in c.items()}
        maxcsv=max(maxcsv,max(errors.values()));max_deriv=max(max_deriv,errors['beta_prime'],errors['Q_prime'])
        em=abs(slope[ti]/kappa-m);ee=abs(slope[2]/kappa-ell)
        max_block=max(max_block,em,ee)
        rows.append(dict(ir=ir,r=r,method=method,task=task,**c,block_m=slope[ti]/kappa,block_ell=slope[2]/kappa,max_csv_error=max(errors.values()),block_error=max(em,ee)))
table('scalar_and_block_checks.csv',rows)
assert maxcsv<1e-8,(maxcsv,max_deriv)
assert max_block<2e-4,max_block

# Polynomial integration on the sphere: moment of each coordinate monomial,
# independently of Gaussian trace-Wick formulae used by the author.
def add(a,b):
    c=a.copy()
    for k,v in b.items():c[k]=c.get(k,0.)+v
    return c
def mul(a,b):
    c={}
    for k,v in a.items():
        for l,w in b.items():
            h=tuple(x+y for x,y in zip(k,l));c[h]=c.get(h,0.)+v*w
    return c
def power(a,j):
    out={(0,)*6:1.}
    for _ in range(j):out=mul(out,a)
    return out
def quadratic(M):
    out={}
    for i,j in itertools.product(range(6),repeat=2):
        k=[0]*6;k[i]+=1;k[j]+=1;k=tuple(k);out[k]=out.get(k,0.)+M[i,j]
    return out
def sphere_integral(poly,n):
    s=0.
    for exps,c in poly.items():
        if any(j%2 for j in exps):continue
        a=sum(exps)//2
        numerator=math.prod(math.prod(range(1,j,2)) for j in exps)
        s+=c*numerator/math.prod(n+2*j for j in range(a))
    return s
d,n=2,23
T=np.arange(36,dtype=float).reshape(6,6)
R=np.eye(6)+.017*np.cos(T+T.T)
C=R@R.T
ev,E=np.linalg.eigh(C); root=(E*np.sqrt(ev))@E.T
lx,ly,le=root[:2],root[2:4],root[4:]
# Deterministic orthonormal Stiefel rows; these are test matrices, not MR draws.
jj=np.arange(n)+.5
QQ=np.array([np.ones(n)/math.sqrt(n)]+[math.sqrt(2/n)*np.cos(math.pi*k*jj/n) for k in range(1,6)])
assert np.max(abs(QQ@QQ.T-np.eye(6)))<1e-14
x,y0,ye=np.split(math.sqrt(n/d)*root@QQ,3)
joint=[]
for method in methods:
    rr=.5;y1=rr*y0+math.sqrt(1-rr*rr)*ye
    S=d/n*x@x.T;Z0=d/math.sqrt(n)*y0@x.T;Z1=d/math.sqrt(n)*y1@x.T;H=Z1-Z0@S
    if method=='GD':h=H
    elif method=='Sign':h=np.sign(H)
    else:h=H/(abs(H)+{'Adam2':2.,'Adam1':1.,'Adam.1':.1}[method])
    alpha=next(p['amplitude'] for p in pop['rows'] if p['r']==rr and p['method']==method)
    U=Z0+alpha*h; q=coeff[0,method,'old']['q']; c=n/(d*math.sqrt(q))
    for task,y,L in [('old',y0,ly),('new',y1,rr*ly+math.sqrt(1-rr*rr)*le)]:
        A=L.T@U@lx;M=(A+A.T)/2;N=A.T@A
        scores=y.T@U@x/math.sqrt(q)
        directD=np.diag(scores);directV=(np.sum(scores*scores,axis=0)-directD*directD)/(n-1)
        qm=np.einsum('ai,ab,bi->i',QQ,M,QQ);qn=np.einsum('ai,ab,bi->i',QQ,N,QQ)
        identity_error=max(np.max(abs(directD-c*qm)),np.max(abs(directV-c*c/(n-1)*(qn-qm*qm))))
        pm,pn=quadratic(M),quadratic(N)
        sm={};tm={}
        tr=lambda v:float(np.trace(v))
        m1,n1=tr(M),tr(N)
        formula={(1,0):m1,(0,1):n1,(2,0):m1*m1+2*tr(M@M),
            (1,1):m1*n1+2*tr(M@N),(3,0):m1**3+6*m1*tr(M@M)+8*tr(M@M@M),
            (0,2):n1*n1+2*tr(N@N),(2,1):n1*(m1*m1+2*tr(M@M))+4*m1*tr(M@N)+8*tr(M@M@N),
            (4,0):m1**4+12*m1*m1*tr(M@M)+12*tr(M@M)**2+32*m1*tr(M@M@M)+48*tr(M@M@M@M)}
        for ab,value in formula.items():
            a,b=ab; sm[ab]=sphere_integral(mul(power(pm,a),power(pn,b)),n)
            tm[ab]=value/math.prod(n+2*j for j in range(a+b))
        moment_error=max(abs(sm[k]-tm[k]) for k in sm)
        joint.append(dict(method=method,task=task,identity_max_error=float(identity_error),sphere_moment_max_error=moment_error,
            ED=c*sm[1,0],ED2=c*c*sm[2,0],EV=c*c/(n-1)*(sm[0,1]-sm[2,0]),EDV=c**3/(n-1)*(sm[1,1]-sm[3,0]),EV2=c**4/(n-1)**2*(sm[0,2]-2*sm[2,1]+sm[4,0])))
assert max(r['identity_max_error'] for r in joint)<1e-10
assert max(r['sphere_moment_max_error'] for r in joint)<1e-10
table('joint_identity_checks.csv',joint)

# Check the displayed long recall coefficient is the same polynomial in mu,t.
import sympy as sp
mm,mu,ww,tt=sp.symbols('m mu w t');zz=tt+mu
long=mm-zz*ww/2+tt*(ww+2*mu**2)/2+2*mu*(tt**2-1)+(tt**3-3*tt)/2-2*mu*zz*tt-zz*(tt**2-1)+zz*(1+zz*tt)/2
short=mm-mu*ww/2-mu/2-mu**2*tt/2
assert sp.expand(long-short)==0

# X1 means are recomputed from individual records, never from Y3 comparison files.
formal=[json.loads(s) for s in (ROOT/'experiments/x1/X1_formal_rows.jsonl').read_text().splitlines()]
obs={}
for row in formal:obs.setdefault((row['d'],row['ir'],row['method']),[]).append(row)
errs0=[];errs1=[];new0=[];new1=[]
for dim,ir in itertools.product([128,256],range(3)):
    for a,b in pairs:
        ca,cb=coeff[ir,a,'old'],coeff[ir,b,'old']
        observed=np.mean([x['old'] for x in obs[dim,ir,a]])-np.mean([x['old'] for x in obs[dim,ir,b]])
        limit=ca['recall_limit']-cb['recall_limit'];corr=(ca['B_recall']-cb['B_recall'])/dim
        errs0.append(float(observed-limit));errs1.append(float(observed-limit-corr))
    for method in methods:
        observed=np.mean([x['new'] for x in obs[dim,ir,method]])-pop['target']
        new0.append(float(observed));new1.append(float(observed-coeff[ir,method,'new']['B_recall']/dim))
rmse=lambda a:float(100*np.sqrt(np.mean(np.array(a)**2)))
x1_result={'old_pp_before':rmse(errs0),'old_pp_after':rmse(errs1),'new_pp_before':rmse(new0),'new_pp_after':rmse(new1)}

# Sampling variance adjustment uses within-dataset paired query covariance.
# Law: Sigma_512 = Sigma_2048 + (1/512 - 1/2048) E[within-query covariance].
# This is an X1-only planning estimate. d512 extrapolates the d256 covariance;
# it is not new-data validation or a claim of known d512 power.
covs={}; var_rows=[]
for dim,ir in itertools.product([128,256],range(3)):
    vals={'old':[],'new':[]};within={'old':[],'new':[]}
    for seed in range(192):
        with np.load(ROOT/f'experiments/x1/X1_formal_d{dim}_r{ir}_s{seed}.npz',allow_pickle=False) as f:
            old=np.array([f[m+'_old_success'] for m in methods],dtype=float).T
            new=np.array([f[m+'_new_success'] for m in methods],dtype=float).T
            dif=np.column_stack([old[:,methods.index(a)]-old[:,methods.index(b)] for a,b in pairs])
            for task,v in [('old',dif),('new',new)]:
                vals[task].append(v.mean(axis=0));within[task].append(np.cov(v,rowvar=False,ddof=1))
    for task in ['old','new']:
        C2048=np.cov(vals[task],rowvar=False,ddof=1);W=np.mean(within[task],axis=0)
        C512=C2048+(1/512-1/2048)*W
        assert np.linalg.eigvalsh(C512).min()>0
        covs[dim,ir,task]=C512
        for j,label in enumerate([a+'-'+b for a,b in pairs] if task=='old' else methods):
            var_rows.append(dict(d=dim,ir=ir,task=task,label=label,sd2048=float(math.sqrt(C2048[j,j])),within_variance=float(W[j,j]),projected_sd512=float(math.sqrt(C512[j,j]))))
table('x1_query_variance.csv',var_rows)
power_rows=[]
for ir,task in itertools.product(range(3),['old','new']):
    C=covs[256,ir,task]
    for j,label in enumerate([a+'-'+b for a,b in pairs] if task=='old' else methods):
        B=(coeff[ir,pairs[j][0],'old']['B_recall']-coeff[ir,pairs[j][1],'old']['B_recall']) if task=='old' else coeff[ir,label,'new']['B_recall']
        shift=B/512;sd=math.sqrt(C[j,j])
        for count in [24,64]:
            nc=abs(shift)*math.sqrt(count)/sd;crit=t.ppf(1-.05/(2*72),count-1)
            power72=float(nct.sf(crit,count-1,nc)+nct.cdf(-crit,count-1,nc))
            crit1=t.ppf(.975,count-1)
            power1=float(nct.sf(crit1,count-1,nc)+nct.cdf(-crit1,count-1,nc))
            # Per-cell squared loss improvement requires one-sided separation at midpoint.
            power_loss=float(nct.sf(t.ppf(.9875,count-1),count-1,nc/2))
            power_rows.append(dict(ir=ir,r=[.5,0.,-.5][ir],task=task,label=label,B=B,shift_pp=100*shift,N=count,
                projected_sd512_queries_pp=100*sd,SE_pp=100*sd/math.sqrt(count),halfwidth72_pp=float(100*crit*sd/math.sqrt(count)),
                approximate_two_sided_power72=power72,approximate_two_sided_power_unadjusted=power1,
                approximate_positive_loss_power_family2=power_loss,
                N_normal_80_family72=float((3.391+0.842)**2*sd*sd/shift**2) if shift else None))
table('d512_power.csv',power_rows)
aggregate=[]
for task,K in [('old',27),('new',45)]:
    for dims in [[128,256,512],[512]]:
        kactual=K if len(dims)==3 else K//3
        mean=0.;var=0.
        for dim,ir in itertools.product(dims,range(3)):
            if task=='old':delta=np.array([(coeff[ir,a,'old']['B_recall']-coeff[ir,b,'old']['B_recall'])/dim for a,b in pairs])
            else:delta=np.array([coeff[ir,m,'new']['B_recall']/dim for m in methods])
            C=covs[min(dim,256),ir,task]
            mean+=float(delta@delta)/kactual
            var+=float(4*delta@C@delta)/kactual**2
        for count in [24,64]:
            nc=mean/math.sqrt(var/count)
            aggregate.append(dict(task=task,dimensions=dims,N=count,assumed_truth='frozen_corrected_prediction',
                expected_loss_improvement=mean,projected_block_SD=math.sqrt(var),
                approximate_power_family2=float(nct.sf(t.ppf(.9875,count-1),count-1,nc))))
save('aggregate_power.json',aggregate)

# Enumerate all planned SeedSequence states (not random data), and compare
# recorded old experiment entropy tuples and all X1 used phases.
new_entropy=[(20261106,30917401,71,d,ir,s,stream) for d,ir,s,stream in itertools.product([128,256,512],range(3),range(64),range(4))]
def signature(ent):return tuple(int(v) for v in np.random.SeedSequence(list(ent)).generate_state(8))
new_states={signature(ent):ent for ent in new_entropy}
assert len(new_states)==len(new_entropy)==2304
old={};entropy_sources=[]
def register(ent,source):
    if isinstance(ent,list) and ent and all(isinstance(x,int) and x>=0 for x in ent):
        ent=tuple(ent);old.setdefault(ent,set()).add(source)
def walk(obj,source):
    if isinstance(obj,dict):
        for key,v in obj.items():
            if 'entropy' in key.lower():
                register(v,source)
                if isinstance(v,list):
                    for j in range(6):register(v+[j],source+' /append stream conservative')
            walk(v,source)
    elif isinstance(obj,list):
        for v in obj:
            if isinstance(v,(list,dict)):walk(v,source)
for path in ROOT.rglob('*'):
    if OUT in path.parents or path.is_symlink() or not path.is_file():continue
    if path.suffix not in ['.json','.jsonl'] or path.name.startswith('Y3_'):continue
    rel=str(path.relative_to(ROOT))
    try:
        if path.suffix=='.jsonl':
            with path.open() as f:
                for line in f:
                    if 'entropy' in line:walk(json.loads(line),rel)
        elif path.stat().st_size<25_000_000:
            text=path.read_text()
            if 'entropy' in text:walk(json.loads(text),rel)
    except (ValueError,UnicodeError):entropy_sources.append({'unparsed':rel})
for phase,d,ir,s,stream in itertools.product([11,22,33],[16,128,256,512],range(3),range(192),range(4)):
    register([20261106,913752,phase,d,ir,s,stream],'X1 all phases, conservative superset')
# Archive every random initializer occurrence, including reviewer experiments,
# and extract constant integer initializers without executing any old code.
initializer_lines=[]
for path in ROOT.rglob('*.py'):
    if OUT in path.parents or path.is_symlink():continue
    rel=str(path.relative_to(ROOT));txt=path.read_text()
    for no,line in enumerate(txt.splitlines(),1):
        if any(s in line for s in ['SeedSequence','default_rng','RandomState','manual_seed','random.seed','Sobol(']):initializer_lines.append({'file':rel,'line':no,'text':line.strip()})
    try:tree=ast.parse(txt)
    except SyntaxError:continue
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['default_rng','SeedSequence','RandomState'] and node.args:
            try:v=ast.literal_eval(node.args[0])
            except (ValueError,TypeError):continue
            if isinstance(v,int):register([v],rel)
            elif isinstance(v,list):register(v,rel)
collisions=[]
for ent,sources in old.items():
    sig=signature(ent)
    if sig in new_states:collisions.append({'old':ent,'new':new_states[sig],'sources':sorted(sources)})
assert not collisions
save('rng_initializer_inventory.json',initializer_lines)
save('seed_state_audit.json',{'planned_streams':2304,'unique_generated_256bit_states':len(new_states),
    'old_entropy_tuples_including_conservative_supersets':len(old),'old_vs_new_state_collisions':collisions,
    'entropy_parse_notes':entropy_sources,'old_entropy_sources':sorted({s for ss in old.values() for s in ss}),
    'limitation':'Static initializer review plus recorded entropy reconstruction. No claim of mathematical RNG independence or exhaustive undocumented executions. No random MR arrays or query IDs generated.'})

after={name:sha(ROOT/name) for name in before}
changed=[name for name in before if before[name]!=after[name]]
assert not changed,changed
summary={'checks':'independent code; deterministic quadrature and test tensors; old X1 queries only; no new MR data',
    'frozen_manifest':manifest,'existing_nonlog_files_hashed':len(before),'changed_existing_nonlog_files':changed,
    'logs_excluded_from_immutability':logs,'symlinks_skipped':symlinks,
    'max_scalar_csv_error':maxcsv,'max_response_derivative_error':max_deriv,'max_exact_block_slope_error':max_block,
    'max_direct_joint_identity_error':max(x['identity_max_error'] for x in joint),
    'max_sphere_polynomial_moment_error':max(x['sphere_moment_max_error'] for x in joint),
    'recall_algebra_identity':True,'recall_remainder_proved':False,'x1_rmse':x1_result,
    'planned_rng_streams':2304,'old_entropy_tuples_checked':len(old),'state_collisions':len(collisions),
    'cpu_seconds':time.process_time()-START,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    'python':sys.version,'numpy':np.__version__,'threadpools':pools,
    'memory_max':(CP/'memory.max').read_text().strip(),'memory_swap_max':(CP/'memory.swap.max').read_text().strip()}
save('audit_summary.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ['logs_excluded_from_immutability','frozen_manifest']},ensure_ascii=False),flush=True)
print(json.dumps(aggregate,ensure_ascii=False),flush=True)
