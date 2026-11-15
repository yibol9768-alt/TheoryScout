"""Independent AA2 review. Imports no project code. Exclusive new outputs.

Run with systemd-run MemoryMax=3G, MemorySwapMax=0 and BLAS threads <=2.
All random matrices and scores below are generated here from their definitions.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.special import hyp2f1, ndtr, ndtri, roots_legendre
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]


def adam_constants(e):
    # Different variable from the audited code: t=exp(z), with Jacobian t.
    def integral(power):
        return math.sqrt(2 / math.pi) * quad(
            lambda z: math.exp(3*z - math.exp(2*z)/2) /
            (math.exp(z)+e)**power, -35, 4, epsabs=2e-13,
            epsrel=2e-13, points=[math.log(e)], limit=200)[0]
    a, q, r = (integral(k) for k in (1, 2, 3))
    return dict(a=a, q=q, c=a/math.sqrt(q), cp=(a*r-q*q)/q**1.5)


def mu_value(gamma):
    return float(8/(3*math.pi)*hyp2f1(-.5, .5, 2.5, 1/gamma))


def polar_mixed_resolvent(gamma, order):
    """Compute k from the unsimplified two-resolvent MP insertion identity.

    Does not implement aa2's simplified rational 2D kernel. MP integral
    evaluated separately in its angular variable; no random-matrix proxy.
    """
    lam = 1/gamma
    z, w = roots_legendre(order)
    x, weights = (z+1)/2, w/2
    p = 1-x*x
    times = p*p*(1-lam+lam*p)/(1-p)
    deriv = ((2*p*(1-lam)+3*lam*p*p)*(1-p)
             + p*p*(1-lam+lam*p))/(1-p)**2
    theta = (z+1)*math.pi/2
    s = 1+lam-2*math.sqrt(lam)*np.cos(theta)
    mpweights = w*np.sin(theta)**2/s
    b = 1/(times[:, None]+p[:, None]*s[None, :])
    ij = [(b*(mpweights*s**j))@b.T for j in (0, 1, 2)]
    pp = p[:, None]*p[None, :]
    kernel = pp*(ij[0]*ij[2]-ij[1]**2)/(1-pp*ij[2])
    jac = 2*x*deriv*np.sqrt(times)
    k = float((weights*jac)@kernel@(weights*jac)/math.pi**2)
    mu = mu_value(gamma)
    return dict(mu=mu, k=k, nu=1+k/gamma-mu*mu/gamma**2)


def invert_c(target):
    return root_scalar(lambda e: adam_constants(e)['c']-target,
                       bracket=[1e-6, 8.], method='toms748', xtol=1e-13).root


def theory():
    rows = []
    for gamma in (16, 32, 64, 128, 256, 1024):
        constants = polar_mixed_resolvent(gamma, 192)
        other = polar_mixed_resolvent(gamma, 96)
        for tau in (0., 1., 2.):
            mu, nu = constants['mu'], constants['nu']
            t = 1+tau*tau
            rows.append(dict(gamma=gamma, tau=tau, **constants,
                root_mean=invert_c(mu),
                root_cue=invert_c(mu/math.sqrt(1-mu*mu/(gamma*t))),
                root_msr=invert_c(mu/math.sqrt((nu+tau*tau-2*mu*mu/gamma)/t)),
                resolvent_order_difference=abs(constants['k']-other['k'])))
    b = 8/(3*math.pi)
    star = invert_c(b)
    params = {}
    const = rows[0]
    for e in (.06, .1, .12, .3):
        a = adam_constants(e)
        ell = 1+a['c']**2/16
        pa = ndtr(a['c']/(.2*math.sqrt(16*ell))-ndtri(.9))
        pp = ndtr(const['mu']/(.2*4)-ndtri(.9))
        params[str(e)] = dict(**a, recall_A=float(pa), recall_P=float(pp),
            recall_difference=float(pa-pp),
            normalized_msr_A=a['c']/math.sqrt(1+2*a['c']**2/16),
            normalized_msr_P=const['mu']/math.sqrt(const['nu']))
    return dict(roots=rows, reference_root=star,
        gap_coefficient=b**3/(2*adam_constants(star)['cp']), points=params,
        arithmetic='Floating diagnostics; strict ordering is reviewed analytically.')


def distance_to_normal(values, mean, std):
    ordered = np.sort(values)
    n = len(ordered)
    expected = ndtr((ordered-mean)/std)
    return float(max(np.max(np.arange(1, n+1)/n-expected),
                     np.max(expected-np.arange(n)/n)))


def evaluate_direction(x, y, u, idx, noise_seed, d, e, batch):
    n = x.shape[1]
    vx = u@x
    diag = np.einsum('ij,ij->j', y, vx)/d
    wrong_sum = y.sum(axis=1)@vx/d-diag
    wrong_means = wrong_sum/(n-1)
    # Exact full-array MSR using d by d contractions; no sampled wrong pairs.
    total_squared = np.sum((y@y.T)*(u@(x@x.T)@u.T))/d**2
    variance = (total_squared-diag@diag)/(n*(n-1))-np.mean(wrong_means**2)
    margin = float(np.mean(diag-wrong_means))
    del vx
    ytu = y.T@u/d
    leverage = np.einsum('ij,ij->i', ytu, ytu)
    signal_constant = mu_value(16) if e is None else adam_constants(e)['c']
    ell = 1 if e is None else 1+signal_constant**2/16
    rng = np.random.default_rng(noise_seed)  # Same draws across methods.
    successes = []
    correct_list = []
    cdf_dists = []
    noise_cdf_dists = []
    wrong_pair_products = []
    ordinary_successes = []
    # Ordinary fixed-noise queries are checked on a small common subset.
    ordinary_remaining = 128
    for start in range(0, len(idx), batch):
        ids = idx[start:start+batch]
        xi = rng.standard_normal((d, len(ids)))
        noise = .2*(ytu@xi)
        base = ytu@x[:, ids]
        scores = base/math.sqrt(d)+noise
        correct = scores[ids, np.arange(len(ids))]
        ranks = np.count_nonzero(scores > correct, axis=0)
        successes.extend((ranks <= math.floor(.1*(n-1))).tolist())
        correct_list.extend(correct.tolist())
        if ordinary_remaining:
            k = min(ordinary_remaining, len(ids))
            ordinary = base[:, :k]+noise[:, :k]
            dc = ordinary[ids[:k], np.arange(k)]
            ordinary_successes.extend((np.count_nonzero(ordinary > dc, axis=0)
                                      <= math.floor(.1*(n-1))).tolist())
            ordinary_remaining -= k
        if start == 0:
            for j in range(min(16, len(ids))):
                wrong = np.concatenate((scores[:ids[j], j], scores[ids[j]+1:, j]))
                wn = np.concatenate((noise[:ids[j], j], noise[ids[j]+1:, j]))
                cdf_dists.append(distance_to_normal(wrong, 0., .2*math.sqrt(ell)))
                noise_cdf_dists.append(distance_to_normal(wn, 0., .2*math.sqrt(ell)))
        # Covariance diagnostic: wrong noise scores share a query.
        wrong_pair_products.extend((noise[0]*noise[1]).tolist())
    successes = np.asarray(successes, dtype=np.uint8)
    record = dict(method='polar' if e is None else f'adam_{e:g}',
        recall=float(successes.mean()), normalized_msr=margin/math.sqrt(variance*d/16),
        margin_coefficient=margin/math.sqrt(d/16), variance=float(variance),
        leverage_mean=float(np.mean(leverage)), leverage_mse=float(np.mean((leverage-ell)**2)),
        leverage_prediction=ell, correct_scaled_mse=float(np.mean((diag/math.sqrt(d)-signal_constant/4)**2)),
        correct_mean=float(np.mean(correct_list)), correct_std=float(np.std(correct_list)),
        correct_ks=distance_to_normal(np.array(correct_list), signal_constant/4, .2*math.sqrt(ell)),
        wrong_cdf_sup=float(np.mean(cdf_dists)), noise_only_cdf_sup=float(np.mean(noise_cdf_dists)),
        ordinary_noisy_recall=float(np.mean(ordinary_successes)),
        sampled_noise_pair_product=float(np.mean(wrong_pair_products)))
    return record, successes


def uncertainty(values):
    values = np.array(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1)/math.sqrt(len(values))) if len(values)>1 else float('nan')
    half = float(student_t.ppf(.975, len(values)-1)*se) if len(values)>1 else float('nan')
    return dict(mean=mean, dataset_se=se, ci95=[mean-half, mean+half])


def simulate(args):
    records, paired = [], []
    for d in args.dims:
        n = 16*d
        for seed in range(args.seeds):
            begin = time.monotonic()
            rng = np.random.default_rng(np.random.SeedSequence([202620, d, seed]))
            x = rng.normal(size=(d,n))
            y = rng.normal(size=(d,n))
            # Explicitly center both arrays before forming the CE gradient.
            xc = x-x.mean(axis=1, keepdims=True)
            yc = y-y.mean(axis=1, keepdims=True)
            gradient = yc@xc.T/math.sqrt(n)
            del xc, yc
            left, _, rt = np.linalg.svd(gradient, full_matrices=False)
            polar = (left@rt)*math.sqrt(d)
            idx = rng.integers(0, n, size=args.queries)
            noise_seed = int(rng.integers(0, 2**62))
            outcomes = {}
            stats = {}
            for e in [None]+args.es:
                u = polar if e is None else gradient/(np.abs(gradient)+e)/math.sqrt(adam_constants(e)['q'])
                record, success = evaluate_direction(x,y,u,idx,noise_seed,d,e,args.batch)
                record.update(d=d, n=n, seed=seed, queries=args.queries)
                records.append(record)
                outcomes[record['method']] = success
                stats[record['method']] = record
            for e in args.es:
                key = f'adam_{e:g}'
                dif = outcomes[key].astype(int)-outcomes['polar'].astype(int)
                paired.append(dict(d=d, seed=seed, method=key, recall_difference=float(dif.mean()),
                    discordant_queries=int(np.count_nonzero(dif)),
                    msr_difference=stats[key]['normalized_msr']-stats['polar']['normalized_msr']))
            print(json.dumps(dict(d=d, seed=seed, seconds=time.monotonic()-begin,
                paired=[r for r in paired if r['d']==d and r['seed']==seed])), flush=True)
    summary = []
    for d in args.dims:
        for e in args.es:
            subset = [r for r in paired if r['d']==d and r['method']==f'adam_{e:g}']
            summary.append(dict(d=d, method=f'adam_{e:g}', datasets=args.seeds,
                recall_difference=uncertainty([r['recall_difference'] for r in subset]),
                msr_difference=uncertainty([r['msr_difference'] for r in subset]),
                positive_recall_datasets=sum(r['recall_difference']>0 for r in subset),
                negative_msr_datasets=sum(r['msr_difference']<0 for r in subset)))
    return dict(records=records, paired=paired, summary=summary)


def fingerprints():
    files = [ROOT/'AA2_EXT_RETRIEVAL.md', ROOT/'X3_POLAR_TOOL.md',
             *sorted((ROOT/'aa2_checks').glob('*')),
             ROOT/'paper/sections/setup.tex', ROOT/'paper/sections/main_results.tex',
             ROOT/'paper/appendix/A.tex', ROOT/'paper/appendix/E.tex', ROOT/'paper/appendix/F.tex']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--mode', choices=['theory', 'simulate'], required=True)
    ap.add_argument('--dims', type=int, nargs='+', default=[256,512,1024])
    ap.add_argument('--seeds', type=int, default=24)
    ap.add_argument('--queries', type=int, default=8192)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--es', type=float, nargs='+', default=[.1,.12])
    args = ap.parse_args()
    output = Path(args.output).resolve()
    assert output.parent == ROOT/'review20_checks'
    if output.exists():
        raise FileExistsError(output)
    begin = time.monotonic()
    hashes = fingerprints()
    result = theory() if args.mode == 'theory' else simulate(args)
    after = fingerprints()
    assert hashes == after, 'Audited inputs changed during run'
    cg = Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
    resource = {}
    for name in ('memory.max','memory.swap.max','memory.peak'):
        p = Path('/sys/fs/cgroup')/cg.lstrip('/')/name
        resource[name] = p.read_text().strip() if p.exists() else None
    result.update(parameters=vars(args), seconds=time.monotonic()-begin,
        interpreter=sys.executable, cgroup=cg, resource=resource,
        threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS',
            'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}, input_sha256_before=hashes,
        input_sha256_after=after, inputs_unchanged=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(output=str(output), seconds=result['seconds'], resource=resource)), flush=True)


if __name__ == '__main__':
    main()
