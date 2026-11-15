"""Deterministic quadrature only; never generates a training dataset.

The recall coefficient is a proposed Edgeworth closure, not a proved theorem.
All outputs use exclusive creation. Run with the resource wrapper in the report.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key) == '2'
import csv
import ctypes
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'y3_checks'
START = time.monotonic()
INPUTS = ['ICML_REVIEW_SIM4.md', 'X1_DESIGN.md', 'X1_RESULTS.md', 'X1_PARAMETERS.md',
          'X1_PREREG.md', 'X2_FINITE_D.md', 'W8_FINITE_D.md',
          'experiments/x1/X1_population.json', 'experiments/x1/X1_primary.csv',
          'experiments/x1/X1_new_matching.csv', 'experiments/x1/X1_formal_rows.jsonl',
          'experiments/x1/X1_run_v3.py']
HASHES = {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in INPUTS}
POP = json.loads((ROOT / 'experiments/x1/X1_population.json').read_text())
Z = float(ndtri(.9))
KAPPA = .25
PAIRS = [('GD', 'Sign'), ('Adam2', 'Adam.1'), ('Adam.1', 'Sign')]

def write_json(name, obj):
    with (OUT / name).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def write_csv(name, rows):
    with (OUT / name).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def expect(fun, tol=2e-12):
    return quad(lambda g: math.sqrt(2 / math.pi) * math.exp(-g * g / 2) * fun(g),
                0, np.inf, epsabs=tol, epsrel=tol)[0]

def response(method, sigma, tol=2e-12):
    # beta'(v) and Q'(v) differentiate the Gaussian density, not h.
    if method == 'GD':
        return dict(beta=1., beta_prime=0., Q=sigma*sigma, Q_prime=1.)
    if method == 'Sign':
        beta = math.sqrt(2 / math.pi) / sigma
        return dict(beta=beta, beta_prime=-beta / (2 * sigma*sigma), Q=1., Q_prime=0.)
    e = {'Adam.1': .1, 'Adam1': 1., 'Adam2': 2.}[method]
    a = expect(lambda g: g*g / (g+e), tol)
    a3 = expect(lambda g: g**4 / (g+e), tol)
    q = expect(lambda g: (g/(g+e))**2, tol)
    j = expect(lambda g: g*g * (g/(g+e))**2, tol)
    return dict(beta=a/sigma, beta_prime=(a3-3*a)/(2*sigma**3),
                Q=q, Q_prime=(j-q)/(2*sigma*sigma))

def coefficients(p, tol=2e-12):
    r, method, alpha = p['r'], p['method'], p['amplitude']
    sigma = math.sqrt(2*(1-r))
    h = response(method, sigma, tol)
    beta, bp, qh, qp = (h[k] for k in ['beta', 'beta_prime', 'Q', 'Q_prime'])
    b = 3-2*r
    q = 1 - alpha*sigma*sigma*beta + alpha*alpha*qh
    ell_num = 1 + 2*alpha*(beta*(r-3)+(r-1)*b*bp) + alpha*alpha*(2*sigma*sigma*beta*beta+b*qp)
    ell = ell_num/(KAPPA*q)
    rows=[]
    for task, r0, r1 in [('old', 1., r), ('new', r, 1.)]:
        ak = r0+alpha*(r1-r0)*beta
        mu = ak/math.sqrt(KAPPA*q)
        m = alpha*((r1-r0)*b*bp-r0*beta)/(KAPPA**1.5*math.sqrt(q))
        w = ell + mu*mu
        t = Z-mu
        phi = math.exp(-t*t/2)/math.sqrt(2*math.pi)
        # Joint score/radius closure derived in the report, conjectural at indicators.
        b_drift = m-mu*w/2
        b_shape = -mu/2-mu*mu*t/2
        br = phi*(b_drift+b_shape)
        rows.append(dict(ir=p['ir'], r=r, method=method, task=task, amplitude=alpha,
                         sigma=sigma, q=q, **h, a_k=ak, mu=mu, m=m, ell=ell, w=w,
                         marginal_correct_variance_coefficient=ell+3*mu*mu,
                         proposed_normalized_drift=b_drift,
                         proposed_shape=b_shape, B_recall=br, recall_limit=float(ndtr(mu-Z))))
    return rows

COEFF = [row for p in POP['rows'] for row in coefficients(p)]
CHECK = [row for p in POP['rows'] for row in coefficients(p, 2e-10)]
assert max(abs(a['B_recall']-b['B_recall']) for a,b in zip(COEFF,CHECK)) < 1e-8
BY = {(a['ir'],a['method'],a['task']): a for a in COEFF}
for p in POP['rows']:
    assert abs(BY[p['ir'],p['method'],'old']['recall_limit']-p['old_recall'])<1e-11
    assert abs(BY[p['ir'],p['method'],'new']['recall_limit']-POP['target'])<1e-11
for row in COEFF:
    if row['method']=='GD':
        p=next(p for p in POP['rows'] if p['ir']==row['ir'] and p['method']=='GD')
        a,r=p['amplitude'],p['r']
        u=1-a
        h0=(u+r*a)**2
        h1=(r*u+a)**2
        w0=2+a*(-8+4*r)+a*a*(8+r*r-8*r)
        w1=1+r*r+a*(-6-2*r*r+4*r)+a*a*(8+r*r-8*r)
        expected=(w0 if row['task']=='old' else w1)/(KAPPA*row['q'])
        assert abs(row['w']-expected)<2e-11

PRED=[]
for d in [128,256,512]:
    for row in COEFF:
        PRED.append({**row, 'd':d, 'recall_first_order':row['recall_limit']+row['B_recall']/d,
                     'new_matching_bias':row['B_recall']/d if row['task']=='new' else ''})

OBS=[]
with (ROOT/'experiments/x1/X1_new_matching.csv').open() as f:
    for old in csv.DictReader(f):
        d,ir=int(old['d']),int(old['ir'])
        row=BY[ir,old['method'],'new']
        observed=float(old['mean'])
        prediction=row['B_recall']/d
        OBS.append(dict(d=d,r=row['r'],method=row['method'],observed_bias=observed,
                        proposed_bias=prediction,residual=observed-prediction,
                        original_lo=float(old['lo']),original_hi=float(old['hi']),
                        absolute_error_improved=abs(observed-prediction)<abs(observed)))

CON=[]
with (ROOT/'experiments/x1/X1_primary.csv').open() as f:
    for old in csv.DictReader(f):
        d,ir=int(old['d']),int(old['ir'])
        a,b=old['a'],old['b']
        ca,cb=BY[ir,a,'old'],BY[ir,b,'old']
        delta=ca['recall_limit']-cb['recall_limit']
        bc=ca['B_recall']-cb['B_recall']
        predicted=delta+bc/d
        observed=float(old['mean'])
        CON.append(dict(d=d,r=ca['r'],a=a,b=b,population=delta,B_difference=bc,
                        proposed=predicted,observed=observed,residual=observed-predicted,
                        original_lo=float(old['lo']),original_hi=float(old['hi']),
                        absolute_error_improved=abs(observed-predicted)<abs(observed-delta)))

FORM=[json.loads(line) for line in (ROOT/'experiments/x1/X1_formal_rows.jsonl').read_text().splitlines()]
OLD=[]
for d in [128,256]:
    for row in COEFF:
        if row['task']!='old': continue
        vals=np.array([r['old'] for r in FORM if r['d']==d and r['ir']==row['ir'] and r['method']==row['method']])
        observed=float(vals.mean())
        pred=row['recall_limit']+row['B_recall']/d
        OLD.append(dict(d=d,r=row['r'],method=row['method'],population=row['recall_limit'],
                        proposed=pred,observed=observed,residual=observed-pred,
                        seed_se=float(vals.std(ddof=1)/math.sqrt(len(vals))),n=len(vals)))

write_csv('Y3_coefficients.csv', COEFF)
write_csv('Y3_frozen_predictions.csv', PRED)
write_csv('Y3_X1_new_comparison.csv', OBS)
write_csv('Y3_X1_contrasts.csv', CON)
write_csv('Y3_X1_old_comparison.csv', OLD)
write_json('Y3_inputs.json',HASHES)
assert all(hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h for s,h in HASHES.items())
cg=Path('/proc/self/cgroup').read_text().strip().split('::')[-1]
cp=Path('/sys/fs/cgroup')/cg.lstrip('/')
threads=[]
for path in sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'openblas' in line and '.so' in line}):
    lib=ctypes.CDLL(path)
    for symbol in ['scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads','openblas_get_num_threads64_','openblas_get_num_threads']:
        if hasattr(lib,symbol):
            fn=getattr(lib,symbol);fn.restype=ctypes.c_int
            count=fn();assert count<=2
            threads.append(dict(path=path,num_threads=count));break
    else: raise RuntimeError(path)
summary=dict(status='deterministic_quadrature_and_previously_observed_X1_only',
             recall_coefficient_status='proposed_not_proved',
             new_improved=sum(r['absolute_error_improved'] for r in OBS),new_total=len(OBS),
             contrast_improved=sum(r['absolute_error_improved'] for r in CON),contrast_total=len(CON),
             new_rmse_before=float(np.sqrt(np.mean([r['observed_bias']**2 for r in OBS]))),
             new_rmse_after=float(np.sqrt(np.mean([r['residual']**2 for r in OBS]))),
             contrast_rmse_before=float(np.sqrt(np.mean([(r['observed']-r['population'])**2 for r in CON]))),
             contrast_rmse_after=float(np.sqrt(np.mean([r['residual']**2 for r in CON]))),
             elapsed_seconds=time.monotonic()-START,cpu_seconds=time.process_time(),
             peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             memory_max=(cp/'memory.max').read_text().strip(),
             memory_swap_max=(cp/'memory.swap.max').read_text().strip(),threads=threads)
assert summary['memory_max']=='3221225472' and summary['memory_swap_max']=='0'
write_json('Y3_summary.json',summary)
print(json.dumps(summary,ensure_ascii=False))
print('COEFFICIENTS (r method task m w B):')
for row in COEFF:
    print(row['r'],row['method'],row['task'],*[round(row[k],8) for k in ['m','w','B_recall']])
print('CONTRASTS (d r pair proposed observed):')
for row in CON:
    print(row['d'],row['r'],row['a']+'-'+row['b'],round(row['proposed'],8),round(row['observed'],8))
