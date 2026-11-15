"""Frozen, CPU-only X2 experiment; no external project imports or network."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import t as student_t
import ctypes


def threadpool_info():
    libraries = sorted(set(s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines()
                           if 'openblas' in s and '/' in s))
    pools = []
    for path in libraries:
        lib = ctypes.CDLL(path)
        for symbol in ['scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads',
                       'openblas_get_num_threads', 'openblas_get_num_threads64_']:
            try:
                fn = getattr(lib, symbol)
            except AttributeError:
                continue
            fn.restype = ctypes.c_int
            pools.append(dict(filepath=path, num_threads=fn(), symbol=symbol))
            break
        else:
            raise RuntimeError(('No BLAS thread-count symbol', path))
    assert len(pools) >= 2, pools
    return pools

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/ext'
DIMS = [128, 256]
GAMMAS = [4, 16, 64, 256]
NSEEDS = 16
ES = [0., .01, .1, 1.]
TAUS = [0., 1., 2.]
B = 8 / (3 * math.pi)
BLOCK = 4096


def put(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resources():
    # cgroup files are system resource controls, not another project.
    line = next(s for s in Path('/proc/self/cgroup').read_text().splitlines()
                if s.startswith('0::'))
    group = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
    vals = {k: (group/k).read_text().strip() for k in
            ['memory.max', 'memory.swap.max', 'memory.peak', 'memory.events']}
    pools = threadpool_info()
    assert vals['memory.max'] == '2147483648', vals
    assert vals['memory.swap.max'] == '0', vals
    assert all(p['num_threads'] <= 2 for p in pools), pools
    return dict(cgroup=str(group), controls=vals, threadpools=pools,
                rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                interpreter=sys.executable)


def moments(e):
    if e == 0:
        return dict(a=math.sqrt(2/math.pi), q=1., c=math.sqrt(2/math.pi))
    density = lambda z: math.sqrt(2/math.pi)*math.exp(-z*z/2)
    a = quad(lambda z: density(z)*z*z/(z+e), 0, math.inf,
             epsabs=2e-13, epsrel=2e-13)[0]
    q = quad(lambda z: density(z)*z*z/(z+e)**2, 0, math.inf,
             epsabs=2e-13, epsrel=2e-13)[0]
    r = quad(lambda z: density(z)*z*z/(z+e)**3, 0, math.inf,
             epsabs=2e-13, epsrel=2e-13)[0]
    return dict(a=a, q=q, c=a/math.sqrt(q), derivative=(a*r-q*q)/q**1.5)


def spectral_mean(gamma):
    # Li F.2, Eq.18, p=z^2 to remove endpoint singularity.
    return quad(lambda z: (2*math.sqrt(1-z*z/gamma)*(1+z*z)
                          + 2*z*z*(1-z*z)/(gamma*math.sqrt(1-z*z/gamma)))/math.pi,
                0, 1, epsabs=2e-13, epsrel=2e-13)[0]


def prepare():
    st = time.monotonic()
    resources()
    ms = {str(e): moments(e) for e in ES}
    ms['GD'] = dict(a=1., q=1., c=1.)
    ec = brentq(lambda e: moments(e)['c']-B, .000001, 1., xtol=5e-15)
    bracket = [ec-1e-10, ec+1e-10]
    grid = []
    for gamma in GAMMAS:
        mu = spectral_mean(gamma)
        delta = 52/gamma**1.5
        for name, m in ms.items():
            c = m['c']
            var = 1+2*c*c/gamma
            approx = c/B*(1+(.1-c*c)/gamma)
            grid.append(dict(gamma=gamma, method=name, mean=c, variance=var,
                             msr=c/math.sqrt(var), ratio_expansion=approx,
                             ratio_spectral_center=c/(math.sqrt(var)*mu),
                             naive_scaled_gd=c/math.sqrt(1+2/gamma)))
    thresholds = []
    for gamma in [16, 64, 256, 512, 1024, 4096]:
        delta = 52/gamma**1.5
        low = B*(1-.1/gamma-1/gamma**2)/math.sqrt(1+delta)
        high = B*(1-.1/gamma+1/gamma**2)/math.sqrt(1-delta)
        thresholds.append(dict(gamma=gamma, lower=low, upper=high,
                               decisions={name: ('above' if m['c']/math.sqrt(1+2*m['c']**2/gamma)>high
                                                else 'below' if m['c']/math.sqrt(1+2*m['c']**2/gamma)<low
                                                else 'unresolved') for name,m in ms.items()}))
    c = ms['0.1']['c']; mu = B*(1-.1/16)
    tau2 = (mu*mu*(1+2*c*c/16)-c*c)/(c*c-mu*mu)
    forecasts = dict(b=B, ecrit=ec, ecrit_bracket=bracket,
                     bracket_values=[moments(e)['c']-B for e in bracket],
                     ecrit_derivative=moments(ec)['derivative'],
                     ncrit_eps1e8=(ec/1e-8)**2, moments=ms, grid=grid,
                     certified_envelopes=thresholds,
                     noise_center_gamma16=dict(mu=mu, variance_muon=1., taucrit=math.sqrt(tau2)),
                     asymptotic_advantages={k: v['c']/B-1 for k,v in ms.items()})
    put(OUT/'EXT_forecasts.json', forecasts)
    paths = ['EXT_CHOICE.md', 'EXT_FORMULATION.md', 'EXT_PREREG.md',
             'EXT_lemmas/L1.tex','EXT_lemmas/L2.tex','EXT_lemmas/L3.tex','EXT_lemmas/L4.tex',
             'experiments/ext/EXT_run.py','experiments/ext/EXT_run_v2.py','EXT_PREDATA_AMENDMENT.md','experiments/ext/EXT_forecasts.json']
    put(OUT/'EXT_frozen.json', {p: digest(ROOT/p) for p in paths})
    # sha256 file is created after this command; start one shared wall budget now.
    put(OUT/'EXT_budget.json', dict(start_unix=time.time(), numerical_limit_seconds=3400,
                                  planning_seconds_upper=5., prepare_seconds=time.monotonic()-st))
    put(OUT/'EXT_prepare_metadata.json', dict(seconds=time.monotonic()-st, resource=resources()))
    print(json.dumps(dict(ecrit=ec, methods=ms, taucrit_center=math.sqrt(tau2))))


def frozen():
    expected = json.loads((OUT/'EXT_frozen.json').read_text())
    for p, sha in expected.items():
        assert digest(ROOT/p) == sha, ('frozen input changed',p)
    line = (ROOT/'EXT_PREREG.sha256').read_text().split()
    assert line[0] == digest(ROOT/'EXT_PREREG.md')


def remaining():
    b = json.loads((OUT/'EXT_budget.json').read_text())
    return b['numerical_limit_seconds'] - (time.time()-b['start_unix'])


def data(d, gamma, seed, tag):
    n = gamma*d
    ent = [20261106, 730021, d, n, seed, tag]
    x = np.random.default_rng(np.random.SeedSequence(ent+[0])).standard_normal((d,n))
    y = np.random.default_rng(np.random.SeedSequence(ent+[1])).standard_normal((d,n))
    noise = np.random.default_rng(np.random.SeedSequence(ent+[2])).standard_normal((d,d))
    z = (y@x.T-np.outer(y.sum(axis=1),x.sum(axis=1))/n)/math.sqrt(n)
    return x,y,z,noise,ent


def evaluate(u, x, y, gx, gy):
    d,n=x.shape
    diag=np.empty(n); wrong=np.empty(n)
    sy=y.sum(axis=1)
    for lo in range(0,n,BLOCK):
        hi=min(n,lo+BLOCK)
        pred=u@x[:,lo:hi]
        diag[lo:hi]=np.einsum('ij,ij->j', y[:,lo:hi],pred)/d
        wrong[lo:hi]=(sy@pred/d-diag[lo:hi])/(n-1)
    total=np.einsum('ij,ji->',u.T@gy@u,gx)/(d*d)
    variance=(total-float(diag@diag))/(n*(n-1))-float(wrong@wrong)/n
    return dict(mean=float(diag.mean()-wrong.mean()), variance=float(variance),
                norm=float(np.linalg.norm(u)/d),
                diagonal_second=float(diag@diag/n), wrong_mean_second=float(wrong@wrong/n)),diag,wrong


def covariance(u,h,gx,gy,du,dh,wu,wh):
    d=u.shape[0];n=len(du)
    total=np.einsum('ij,ji->',u.T@gy@h,gx)/(d*d)
    return float((total-du@dh)/(n*(n-1))-wu@wh/n)


def one(d,gamma,seed,tag,forecast,noise_enabled=False,smoke=False):
    x,y,z,noise,ent=data(d,gamma,seed,tag)
    gx=x@x.T;gy=y@y.T
    a,_,bt=np.linalg.svd(z,full_matrices=False)
    matrices={'GD':z,'polar':math.sqrt(d)*(a@bt)}
    for e in ES:
        raw=np.sign(z) if e==0 else z/(np.abs(z)+e)
        matrices[str(e)]=raw/math.sqrt(forecast['moments'][str(e)]['q'])
    rows=[]; stored={}
    for name,u in matrices.items():
        r,diag,wrong=evaluate(u,x,y,gx,gy)
        r.update(method=name,d=d,gamma=gamma,n=d*gamma,seed=seed,entropy=ent,
                 mean_scaled=r['mean']/math.sqrt(d/gamma),
                 msr_scaled=r['mean']/math.sqrt(r['variance'])/math.sqrt(d/gamma),tau=0.)
        rows.append(r);stored[name]=(diag,wrong)
        if smoke:
            scores=y.T@u@x/d
            ix=np.arange(d*gamma)
            explicit_diag=scores[ix,ix].copy()
            scores[ix,ix]=0
            explicit_wrong=scores.sum(axis=0)/(d*gamma-1)
            explicit_var=(scores*scores).sum(axis=0)/(d*gamma-1)-explicit_wrong**2
            assert np.allclose(diag,explicit_diag,rtol=2e-12,atol=2e-12)
            assert np.allclose(wrong,explicit_wrong,rtol=2e-12,atol=2e-12)
            assert abs(r['variance']-explicit_var.mean())<2e-11
            identity=np.sum(u*z)*math.sqrt(d*gamma)/(d*(d*gamma-1))
            assert abs(r['mean']-identity)<2e-11
    if noise_enabled:
        rn,dn,wn=evaluate(noise,x,y,gx,gy)
        for name in ['0.1','polar']:
            base=next(r for r in rows if r['method']==name)
            du,wu=stored[name]
            cov=covariance(matrices[name],noise,gx,gy,du,dn,wu,wn)
            for tau in TAUS[1:]:
                mean=base['mean']+tau*rn['mean']
                var=base['variance']+2*tau*cov+tau*tau*rn['variance']
                r=dict(base,mean=mean,variance=var,mean_scaled=mean/math.sqrt(d/gamma),
                       msr_scaled=mean/math.sqrt(var)/math.sqrt(d/gamma),tau=tau,
                       noise_mean=rn['mean'],noise_variance=rn['variance'],cross_covariance=cov)
                rows.append(r)
                if smoke:
                    re,_,_=evaluate(matrices[name]+tau*noise,x,y,gx,gy)
                    assert abs(re['variance']-var)<2e-11
                    assert abs(re['mean']-mean)<2e-11
    if smoke:
        # Independent raw-gradient, bias-corrected AdamW implementation.
        n=x.shape[1]
        probabilities=np.full((n,n),1/n)
        neggrad=y@(np.eye(n)-probabilities)@x.T/n
        assert np.allclose(neggrad,z/math.sqrt(n),atol=2e-12)
        for beta1,beta2 in [(0.,0.),(.9,.999),(.9,0.)]:
            m=(1-beta1)*(-neggrad);v=(1-beta2)*neggrad**2
            raw=-(m/(1-beta1))/(np.sqrt(v/(1-beta2))+.1/math.sqrt(n))
            assert np.allclose(raw,z/(np.abs(z)+.1),atol=2e-12)
        uncentered=y@x.T/math.sqrt(n)
        assert np.linalg.norm(z-uncentered)>1e-6
    return rows


def run(mode):
    st=time.monotonic();frozen();start_resources=resources()
    forecast=json.loads((OUT/'EXT_forecasts.json').read_text())
    if mode=='smoke':
        tasks=[(16,4,0),(24,16,1)];tag=91
    else:
        assert (OUT/'EXT_smoke_metadata.json').exists()
        assert json.loads((OUT/'EXT_smoke_metadata.json').read_text())['all_checks_passed']
        tasks=[(d,g,s) for s in range(NSEEDS) for g in GAMMAS for d in DIMS];tag=11
    path=OUT/f'EXT_{mode}_rows.jsonl'
    completed=[];stopped=False
    with path.open('x') as f:
        for d,gamma,seed in tasks:
            if remaining()<180:
                stopped=True;break
            t=time.monotonic()
            rows=one(d,gamma,seed,tag,forecast,noise_enabled=(gamma==16 or mode=='smoke'),smoke=mode=='smoke')
            for row in rows:
                f.write(json.dumps(row,allow_nan=False)+'\n')
            f.flush();completed.append([d,gamma,seed])
            print(json.dumps(dict(mode=mode,d=d,gamma=gamma,seed=seed,
                                  seconds=time.monotonic()-t,remaining=remaining())),flush=True)
    frozen()
    put(OUT/f'EXT_{mode}_metadata.json',dict(seconds=time.monotonic()-st,completed=completed,
        expected_configurations=len(tasks),completed_configurations=len(completed),stopped=stopped,
        all_checks_passed=(not stopped),start_resource=start_resources,end_resource=resources(),
        data_file_sha256=digest(path),remaining_seconds=remaining()))


def analyze():
    st=time.monotonic();frozen();resources()
    import csv
    rows=[json.loads(s) for s in (OUT/'EXT_formal_rows.jsonl').read_text().splitlines()]
    fc=json.loads((OUT/'EXT_forecasts.json').read_text())
    groups={}
    for r in rows:
        key=(r['d'],r['gamma'],r['method'],r['tau'])
        groups.setdefault(key,[]).append(r)
    def stat(values):
        return float(np.mean(values)),float(np.std(values,ddof=1)/math.sqrt(len(values)))
    summary=[];gates=[]
    def addgate(family,context,passed,**extra):
        gates.append(dict(family=family,context=context,passed=bool(passed),**extra))
    # Full cross product, not just observed records, determines expected coverage.
    expected_keys={(d,g,m,0.) for d in DIMS for g in GAMMAS
                   for m in ['GD','polar']+[str(e) for e in ES]}
    expected_keys|={(d,16,m,tau) for d in DIMS for m in ['0.1','polar'] for tau in [1.,2.]}
    coverage=set(groups)==expected_keys and all(len(rs)==NSEEDS and sorted(r['seed'] for r in rs)==list(range(NSEEDS)) for rs in groups.values())
    addgate('G0','coverage',coverage,groups=len(groups),rows=len(rows))
    tc=float(student_t.ppf(1-.05/(2*80),NSEEDS-1))
    for key,rs in sorted(groups.items()):
        d,g,m,tau=key
        sm={metric:dict(zip(['mean','se'],stat([r[metric] for r in rs])))
            for metric in ['mean_scaled','variance','msr_scaled','norm']}
        summary.append(dict(d=d,gamma=g,method=m,tau=tau,n_seeds=len(rs),statistics=sm))
        if tau==0 and m!='polar':
            c=fc['moments'][m]['c']
            for metric,target,tol in [('mean_scaled',c,.04),('variance',1+2*c*c/g,.06)]:
                v=sm[metric];err=v['mean']-target
                bound=abs(err)+tc*v['se']
                addgate('G1',str(key)+':'+metric,len(rs)==NSEEDS and bound<=tol,
                        measured=v['mean'],se=v['se'],prediction=target,error=err,
                        simultaneous_equivalence_bound=bound,tolerance=tol)
    # Six predeclared directional contrasts; paired seeds are the units.
    contrasts=[('0.0',256,0.,-1),('0.01',256,0.,-1),('0.1',256,0.,1),('1.0',256,0.,1),
               ('0.1',16,0.,-1),('0.1',16,2.,1)]
    tdir=float(student_t.ppf(1-.05/6,NSEEDS-1))
    for m,g,tau,sgn in contrasts:
        aa={r['seed']:r for r in groups.get((256,g,m,tau),[])}
        bb={r['seed']:r for r in groups.get((256,g,'polar',tau),[])}
        seeds=sorted(set(aa)&set(bb))
        dif=[aa[s]['msr_scaled']/bb[s]['msr_scaled']-1 for s in seeds]
        mu,se=stat(dif);bound=sgn*mu-tdir*se
        addgate('G2' if g==256 else 'G4',str((256,g,m,tau)),len(seeds)==NSEEDS and bound>0,
                paired_relative_gap=mu,se=se,signed_lower_bound=bound,expected_sign=sgn)
    # Expansion is an approximation, not an exact fixed-gamma law.
    n3=2*2*5
    t3=float(student_t.ppf(1-.05/(2*n3),NSEEDS-1))
    for d in DIMS:
        for g in [64,256]:
            polar={r['seed']:r for r in groups.get((d,g,'polar',0.),[])}
            for m in ['GD']+[str(e) for e in ES]:
                rs=groups.get((d,g,m,0.),[])
                vals=[r['msr_scaled']/polar[r['seed']]['msr_scaled'] for r in rs]
                mu,se=stat(vals);c=fc['moments'][m]['c']
                pred=c/B*(1+(.1-c*c)/g)
                bound=abs(mu-pred)+t3*se
                addgate('G3',str((d,g,m)),len(rs)==NSEEDS and bound<=.025,
                        ratio=mu,se=se,prediction=pred,equivalence_bound=bound,tolerance=.025)
    # Independent-Gaussian perturbation moment law, at d256/g16 only.
    tn=float(student_t.ppf(1-.05/(2*4),NSEEDS-1))
    c=fc['moments']['0.1']['c']
    for tau in [1.,2.]:
        rs=groups.get((256,16,'0.1',tau),[])
        for metric,pred,tol in [('mean_scaled',c,.04),('variance',1+2*c*c/16+tau*tau,.10)]:
            mu,se=stat([r[metric] for r in rs]);bound=abs(mu-pred)+tn*se
            addgate('G5',str((256,16,'0.1',tau,metric)),len(rs)==NSEEDS and bound<=tol,
                    measured=mu,se=se,prediction=pred,equivalence_bound=bound,tolerance=tol)
    # Preregistered diagnostic to distinguish the two finite-budget corrections.
    c=fc['moments']['0.0']['c'];truth=c/math.sqrt(1+2*c*c/4);naive=c/math.sqrt(1+2/4)
    rs=groups.get((256,4,'0.0',0.),[])
    mu,se=stat([r['msr_scaled'] for r in rs])
    addgate('G6','sign_gamma4_reject_naive',len(rs)==NSEEDS and
            abs(mu-truth)+float(student_t.ppf(.975,15))*se<=.015 and
            mu-float(student_t.ppf(.99,15))*se>naive+.015,
            measured=mu,se=se,new_prediction=truth,naive_prediction=naive)
    counts={fam:dict(passed=sum(g['passed'] for g in gates if g['family']==fam),
                     total=sum(g['family']==fam for g in gates)) for fam in sorted(set(g['family'] for g in gates))}
    put(OUT/'EXT_summary.json',dict(gate_counts=counts,gates=gates,groups=summary,
                                   formal_rows_sha256=digest(OUT/'EXT_formal_rows.jsonl')))
    fields=sorted(set().union(*(g.keys() for g in gates)))
    with (OUT/'EXT_gates.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(gates)
    for g in gates:
        if not g['passed']:print('FAILED',json.dumps(g))
    print(json.dumps(counts))
    put(OUT/'EXT_analysis_metadata.json',dict(seconds=time.monotonic()-st,resource=resources(),
        remaining_seconds=remaining(),frozen_verified=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','smoke','formal','analyze'])
    mode=p.parse_args().mode
    assert Path.cwd().resolve()==ROOT
    if mode=='prepare':prepare()
    elif mode=='analyze':analyze()
    else:run(mode)
