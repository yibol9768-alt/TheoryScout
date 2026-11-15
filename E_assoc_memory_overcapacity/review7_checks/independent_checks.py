"""Review 7: independent Gaussian simulations and a three-chi-square reduction.

No imports from project code. Run only inside the requested systemd scope.
All output files are opened exclusively. Statistical units are independent datasets.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import sys
sys.dont_write_bytecode = True
import json
import math
import hashlib
import time
from pathlib import Path
import numpy as np
from scipy.special import gammaln
from scipy.stats import t as student_t, chi2

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CS = math.sqrt(2 / math.pi)
START = time.monotonic()
RNG = np.random.default_rng(761903524)


def resources():
    entry = next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines()
                 if s.startswith('0::'))
    base = Path('/sys/fs/cgroup') / entry.lstrip('/')
    limits = {k: (base / k).read_text().strip()
              for k in ('memory.max', 'memory.swap.max', 'memory.peak')}
    threads = int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines()
                       if s.startswith('Threads:')))
    assert limits['memory.max'] == '2147483648'
    assert limits['memory.swap.max'] == '0'
    assert threads <= 2
    return dict(cgroup=limits, process_threads=threads,
                thread_environment={k: os.environ[k] for k in
                                    ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})


def save(name, obj):
    with (OUT / name).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stats(values, target=None):
    a = np.asarray(values, dtype=float)
    se = float(a.std(ddof=1) / math.sqrt(a.size))
    mean = float(a.mean())
    radius = float(student_t.ppf(.975, a.size - 1)) * se
    out = dict(reps=a.size, mean=mean, se=se, ci95=[mean-radius, mean+radius])
    if target is not None:
        out.update(target=float(target), z=(mean-target)/se if se else None)
    return out


def mn(n):
    if n < 100000:
        return math.sqrt(2/n) * math.exp(gammaln((n+1)/2)-gammaln(n/2))
    # Gamma ratio expansion; error O(n^-4), below floating point precision here.
    return 1 - 1/(4*n) + 1/(32*n*n) + 5/(128*n**3)


def exact_sign(d, n):
    mu = CS*d*mn(n)/math.sqrt(n)
    common = CS**2*(2*d-1)*mn(n)**2/n
    return dict(mu=mu, var=1+6*CS**2*(d-1)/(n+2)-common,
                cov=2*CS**2*(d-1)/(n+2)-common,
                samplevar=1+4*CS**2*(d-1)/(n+2),
                wrong2=1+2*CS**2*(d-1)/(n+2))


before = resources()
paths = [ROOT / name for name in ('W8_FINITE_D.md', 'THEOREM_M.md',
         'W7_FT_RANK.md', 'PROOF_REVIEW6.md', 'W4_RESULTS.md',
         'experiments/w4/W4_raw.jsonl')]
paths += sorted((ROOT / 'w8_checks').glob('*'))
fingerprints = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths if p.is_file()}
save('input_hashes.json', fingerprints)

# A: Exact finite-(d,n) Sign scores, computed by explicit tensor contractions.
# Include n<d, d=1, and the minimum legal subset size, not only asymptotic cases.
small = []
for d, n in ((1, 2), (2, 3), (3, 7), (5, 4)):
    pred = exact_sign(d, n)
    observations = []
    for batch in range(400):
        g = RNG.standard_normal((1000, d, n))
        h = RNG.standard_normal((1000, d, n))
        u = np.sign(np.einsum('rai,rbi->rab', g, h))
        score = np.einsum('rai,rab,rbi->ri', g, u, h) / d
        wrong = np.einsum('ra,rab,rb->r', g[:, :, 1], u, h[:, :, 0])/d
        centered = score-pred['mu']
        observations.append(np.column_stack((score[:, 0], centered[:, 0]**2,
                            centered[:, 0]*centered[:, 1],
                            np.var(score, axis=1, ddof=1),
                            (score[:, 0]-score[:, 1])**2/2, wrong**2)))
    data = np.concatenate(observations)
    labels = ('mu', 'var', 'cov', 'samplevar', 'subset2_samplevar', 'wrong2')
    targets = [pred[k] for k in ('mu', 'var', 'cov', 'samplevar', 'samplevar', 'wrong2')]
    small.append(dict(d=d, n=n, results={k: stats(data[:, j], target)
                 for j, (k, target) in enumerate(zip(labels, targets))}))
save('sign_small.json', small)
print('small exact moments complete', flush=True)

# B: Direct actual Gaussian-data GD -> Sign, without W8 routines.
eta = alpha = .6
q = eta**2+alpha**2+2*eta*alpha*CS
mu = (eta+alpha*CS)/math.sqrt(q)
beta = -alpha*CS*eta**2/(2*(1-eta)**2*math.sqrt(q))
direct = []
with (OUT / 'direct_raw.jsonl').open('x', buffering=1) as raw:
    for d, reps in ((16, 512), (32, 512), (64, 384), (128, 192)):
        n = d*d
        records = []
        for r in range(reps):
            x = RNG.standard_normal((d, n))/math.sqrt(d)
            y = RNG.standard_normal((d, n))/math.sqrt(d)
            xy = y @ x.T
            s = (d/n)*(x @ x.T)
            z = (d/math.sqrt(n))*xy
            residual = z-eta*(z @ s)
            u = eta*z+alpha*np.sign(residual)
            actual = float(np.sum(u*xy)/(n*math.sqrt(q)))
            c = np.eye(d)-eta*s
            # Matrix products C S and C S C; no S^2/S^3 implementation.
            numerator = np.diag(c @ s)
            denominator = np.sqrt(np.diag(c @ s @ c))
            rb = d/math.sqrt(n*q)*(eta*np.trace(s)/d +
                                  alpha*CS*np.mean(numerator/denominator))
            sign_scores = np.einsum('ai,ai->i', y, np.sign(z) @ x)
            row = dict(d=d, replicate=r, actual=actual, rb=float(rb),
                       sign_samplevar=float(np.var(sign_scores, ddof=1)))
            records.append(row)
            raw.write(json.dumps(row)+'\n')
        direct.append(dict(d=d, n=n, actual=stats([r['actual'] for r in records], mu+beta/d),
                      rb=stats([r['rb'] for r in records], mu+beta/d),
                      actual_minus_rb=stats([r['actual']-r['rb'] for r in records], 0),
                      sign_samplevar=stats([r['sign_samplevar'] for r in records],
                                           exact_sign(d,n)['samplevar'])))
        print(f'direct d={d} complete; seconds={time.monotonic()-START:.1f}', flush=True)
save('direct_summary.json', dict(eta=eta, alpha=alpha, mu=mu, beta=beta, rows=direct))

# C: Independent exact law of the Wishart ratio from first-row orthogonalization:
# A=chi_n^2/n, B=chi_(d-1)^2, C=chi_(n-1)^2 independent,
# K=1-eta*(A+B/n), ratio=sqrt(A)*K/sqrt(K^2+eta^2*B*C/n^2).
# Subtract a fixed analytic control variate; no fitted coefficients or seed selection.
reduced = []
for eta, kappa in ((.3, 1.), (.6, .5), (.6, 1.), (.6, 2.), (.8, 1.)):
    alpha = .6
    q = eta**2+alpha**2+2*eta*alpha*CS
    factor = alpha*CS/math.sqrt(kappa*q)
    beta = -factor*eta**2/(2*(1-eta)**2*kappa)
    for d in (32, 64, 128, 256, 512, 1024, 4096):
        n = round(kappa*d*d)
        t = 1-eta
        control_mean = mn(n)*(1-eta**2*(d-1)*(n-1)/(2*t*t*n*n))
        deltas = []
        raw_sums = []
        for batch in range(10):
            a = RNG.chisquare(n, 100000)/n
            b = RNG.chisquare(d-1, 100000)
            c = RNG.chisquare(n-1, 100000)
            k = 1-eta*(a+b/n)
            f = np.sqrt(a)*k/np.hypot(k, eta*np.sqrt(b*c)/n)
            control = np.sqrt(a)*(1-eta**2*b*c/(2*t*t*n*n))
            deltas.append(f-control)
            raw_sums.append(float(f.mean()))
        delta = np.concatenate(deltas)
        scaled = d*factor*(delta+control_mean-1)
        result = stats(scaled, beta)
        result.update(d=d, n=n, eta=eta, alpha=alpha, kappa=kappa,
                      raw_ratio_mean=float(np.mean(raw_sums)),
                      ratio_mean=float(delta.mean()+control_mean),
                      note='z against asymptotic beta includes finite-d remainder; not an exact-null test')
        reduced.append(result)
save('chi_reduction.json', reduced)
print('chi-square coefficient study complete', flush=True)

# D: Recalculate W4 and W8 descriptive statistics, including paired Y-noise checks.
with (ROOT/'experiments/w4/W4_raw.jsonl').open() as f:
    w4rows = [json.loads(line) for line in f]
with (ROOT/'w8_checks/W8_raw.jsonl').open() as f:
    w8rows = [json.loads(line) for line in f]
compat = []
zvalues = []
for label, rows, key in (('W4', w4rows, 'full_variance'), ('W8', w8rows, 'sign1_var')):
    for d in (64,128,256):
        selected = [r for r in rows if r['d']==d and
                    (label!='W4' or (r.get('kind')=='M' and r.get('method')=='Sign1'))]
        result = stats([r[key] for r in selected], exact_sign(d,d*d)['samplevar'])
        result.update(source=label,d=d)
        compat.append(result)
        zvalues.append(result['z'])
paired = []
for d in (64,128,256):
    selected = [r for r in w8rows if r['d']==d]
    paired.append(dict(d=d, difference=stats([r['hybrid_mean']-r['hybrid_rb_mean']
                                            for r in selected],0)))
save('compatibility.json', dict(rows=compat, w8_actual_minus_rb=paired,
     approximate_chisq=float(np.sum(np.square(zvalues))),
     approximate_chisq_p=float(chi2.sf(np.sum(np.square(zvalues)),len(zvalues))),
     qualification='Descriptive Gaussian/Student approximations, no equivalence margin; no proof by nonrejection.'))

unchanged = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
                for p,h in fingerprints.items())
assert unchanged
save('metadata.json', dict(seed=761903524, python=sys.executable,
     source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     input_files_unchanged=unchanged, before=before, after=resources(),
     seconds=time.monotonic()-START))
print('complete', flush=True)
