"""Review 19: independently written numerical audit; no audited code imported.

Full Gaussian training arrays and actual matrix optimizers are the main check.
Independent complete datasets, not correlated queries, are the sampling units.
Only review19_checks/ receives outputs; all output files are exclusive-create.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    assert os.environ.get(name) == '1'
import csv
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
OUT = ROOT / 'review19_checks'
METHODS = ['GD2', 'Sign1', 'Sign2_06', 'Sign2_15', 'Adam2_e1']
PAIRS = [(0, 3), (2, 1), (1, 3), (4, 1)]
KAPPA, RHO = .25, .1
SEED = 202611061919
SOURCES = [ROOT/'AA1_QUANT_RANK.md', *sorted((ROOT/'aa1_checks').rglob('*')),
           ROOT/'paper/sections/setup.tex', ROOT/'paper/sections/main_results.tex']
SOURCES = [p for p in SOURCES if p.is_file()]

def save(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def digest():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SOURCES}

def run_optimizer(target, covariance, method):
    """One implementation accepts both scalar and matrix inputs, per setup.tex."""
    state = np.zeros_like(target, dtype=float)
    first = np.zeros_like(state)
    second = np.zeros_like(state)
    steps = 1 if method == 'Sign1' else 2
    for iteration in range(1, steps+1):
        predicted = state if covariance is None else state @ covariance
        gradient = predicted-target
        if method == 'GD2':
            direction = gradient
            step = .5
        elif method.startswith('Sign'):
            direction = np.sign(gradient)
            step = 1.5 if method == 'Sign2_15' else .6
        else:
            first = .9*first + .1*gradient
            second = .99*second + .01*gradient*gradient
            direction = (first/(1-.9**iteration))/(1+np.sqrt(second/(1-.99**iteration)))
            step = .6
        state = state-step*direction
    return state

def population():
    result = {}
    for method in METHODS:
        # Infinite tails, with exact discontinuity breakpoints for Sign.
        knots = [0., .6, 1.5, math.inf]
        def integrand(x, power):
            h = float(run_optimizer(np.asarray(x), None, method))
            return (x*h if power == 1 else h*h)*math.sqrt(2/math.pi)*math.exp(-x*x/2)
        a = sum(quad(integrand, l, u, args=(1,), epsabs=2e-12)[0]
                for l, u in zip(knots[:-1], knots[1:]))
        q = sum(quad(integrand, l, u, args=(2,), epsabs=2e-12)[0]
                for l, u in zip(knots[:-1], knots[1:]))
        c = a/math.sqrt(q)
        result[method] = dict(a=a, q=q, c=c,
                             recall=float(ndtr(c/math.sqrt(KAPPA)-ndtri(1-RHO))))
    return result

def optimizer_constants(method):
    if method == 'GD2':
        k = .75
        return dict(K=k, va=2*k*k, vq=2*k**4, r=.5, factor=2*(2*.25+k))
    if method.startswith('Sign'):
        T = 1 if method == 'Sign1' else 2
        alpha = 1.5 if method == 'Sign2_15' else .6
        K = T*alpha
        A = 2*T*T/math.sqrt(2*math.pi)
        B = sum((j*alpha+2)**2 for j in range(T))
        factor = 2*T*alpha*math.sqrt(A*(2*B/A)**(1/3)+B*(A/(2*B))**(2/3))
        return dict(K=K, va=K*K, vq=K**4, r=1/6, factor=factor)
    alpha, e, beta2, T = .6, 1., .99, 2
    lam, c0 = 1+2*alpha/e, math.sqrt(1-beta2)
    H = [e*math.expm1(t*math.log(lam))/(2*c0) for t in range(T+1)]
    accum = [0.]
    for t in range(1, T+1):
        lipschitz = (1+lam**(t-1)/c0)/e
        accum.append(accum[-1]+alpha*lipschitz*math.sqrt(sum(
            (2*accum[j]+H[j]+2)**2 for j in range(t))))
    K = H[-1]
    return dict(K=K, va=K*K, vq=K**4, r=.5, factor=accum[-1], H=H, A=accum)

def dimension_constants(method, gap, pop):
    cfg = optimizer_constants(method)
    K, va, vq = (cfg[s] for s in ['K', 'va', 'vq'])
    xi = gap/40
    q = pop[method]['q']
    b = math.sqrt(q/2)
    z = float(ndtri(1-RHO)); Z = z+1
    rstar = min(1/(64*Z), 1/(4*math.sqrt(52*Z)))
    M = 3/math.sqrt(KAPPA); D4 = 8*M**4+480
    KR = 12*(1+Z)+3*math.sqrt(52*Z)
    Kn = 12*(1+Z)+3*math.sqrt(2*Z*(M*M+2))
    ax = 2*math.log(9)+math.log(2/xi)
    D = 4*math.sqrt(ax/KAPPA)+4*ax/KAPPA
    V0 = 2*K*math.sqrt(2*ax)
    F = cfg['factor']*D**(2*cfg['r'])
    s = min(RHO/2, (.5-RHO)/2)
    invphi = math.sqrt(2*math.pi)*math.exp(float(ndtri(1-RHO+s))**2/2)
    Hrho = 5*math.sqrt(2)*invphi+4*math.sqrt(2)*z
    x = math.log(4/xi)
    N0 = max(128, (32*Z)**2, 8*Z*(M*M+2), 8*math.sqrt(Z*D4),
             25*x/s**2, 16*x, x*Hrho**2/min(z/2, 1)**2)
    d0 = max(1, 3/KAPPA, (8*D)**2, (4*V0/(b*rstar))**2,
             2*math.sqrt(vq)/(q*math.sqrt(xi)), math.sqrt(N0/KAPPA))
    pieces = [4*F/(b*rstar), 4*F/(b*math.sqrt(KAPPA)), 2*D/math.sqrt(KAPPA),
              math.sqrt(va)/(b*math.sqrt(KAPPA*xi)),
              math.sqrt(vq)/(b*b*math.sqrt(KAPPA*xi)),
              12*(1+Z)*D, 2*KR/b*(V0+F), Kn/math.sqrt(KAPPA),
              12*math.sqrt(Z*D4)/KAPPA, Hrho*math.sqrt(x/KAPPA),
              2/9, 1/(2*KAPPA**1.5)]
    return dict(C=sum(pieces), d0=d0, rate=cfg['r'], xi=xi)

def finite_budget(d, method, pop, xi=.01):
    n = math.ceil(KAPPA*d*d); m = n-1
    cfg = optimizer_constants(method)
    K, va, vq = (cfg[s] for s in ['K', 'va', 'vq'])
    q = pop[method]['q']
    xc = 2*d*math.log(9)+math.log(2/xi)
    delta = 4*math.sqrt(xc/n)+4*xc/n
    ua, uq = math.sqrt(va)/(d*math.sqrt(xi)), math.sqrt(vq)/(d*math.sqrt(xi))
    vd = 2*K*math.sqrt(2*xc)/d
    # q-uq can be negative; do not silently evaluate b or an illegal formula.
    x = math.log(4/xi); theta = math.ceil(RHO*n)/m
    omega = math.sqrt(x/(2*m))
    rm2 = 1-2*math.sqrt(x/m)
    valid_quantile = 0 < 1-theta-omega < 1-theta+omega < 1
    cond = dict(moment=uq<=q/2, delta=delta<=1/8, quantile=valid_quantile,
                radius=rm2>0)
    data = dict(d=d, method=method, delta=delta, ua=ua, uq=uq, q=q, vd=vd)
    raw = None
    if valid_quantile and rm2>0 and q>uq:
        b = math.sqrt(q-uq)
        z = float(ndtri(1-RHO))
        zm = float(ndtri(1-theta-omega))/math.sqrt(1+2*math.sqrt(x/m)+2*x/m)
        zp = float(ndtri(1-theta+omega))/math.sqrt(rm2)
        Z = max(1., z, zm, zp)
        rstar = min(1/(64*Z), 1/(4*math.sqrt(52*Z)))
        cond['positive_threshold'] = zm>0
        cond['operator'] = vd<=b*rstar/4
        M = 3/math.sqrt(KAPPA); D4 = 8*M**4+480
        cond['n'] = n>=max(128, (32*Z)**2, 8*Z*(M*M+2), 8*math.sqrt(Z*D4))
        ell = cfg['factor']*delta**(2*cfg['r'])
        KR = 12*(1+Z)+3*math.sqrt(52*Z)
        Kn = 12*(1+Z)+3*math.sqrt(2*Z*(M*M+2))
        raw = (5*xi+2/(9*d*d)+max(abs(zm-z), abs(zp-z))
               +4*ell/(b*rstar)+(4*ell/b+2*delta+ua/b+uq/b**2)/math.sqrt(KAPPA)
               +12*(1+Z)*delta+2*KR*(vd+ell)/b+Kn/math.sqrt(n)
               +12*math.sqrt(Z*D4)/n)
        data.update(b=b, rstar=rstar, ell=ell, spectral_allowance=b*rstar/4)
    data.update(conditions=cond, admissible=all(cond.values()),
                formal_raw_expression_not_a_bound_when_inadmissible=raw,
                usable_bound=min(1., raw) if all(cond.values()) and raw is not None else 1.)
    return data

def main():
    started = time.monotonic()
    before = digest()
    cg = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split(':')[-1].lstrip('/')
    assert (cg/'memory.max').read_text().strip() == '3221225472'
    assert (cg/'memory.swap.max').read_text().strip() == '0'
    design = dict(seed=SEED, kappa=KAPPA, rho=RHO, methods=METHODS,
                  full_array_datasets={32:128, 64:128, 128:96, 256:64}, queries_per_dataset=32,
                  sufficient_stat_outer={32:2048, 64:8192, 128:1024, 256:512},
                  arithmetic='float64 diagnostic, not directed rounding', input_sha256=before)
    save('design.json', design)
    pop = population()
    rng = np.random.default_rng(SEED)
    rows = []
    full_summaries = []
    # Directly generate all x_i,y_i. No residual integral or Bartlett sampling here.
    for d, repeats in design['full_array_datasets'].items():
        n = math.ceil(KAPPA*d*d)
        per_dataset = []
        trajectory = []
        for rep in range(repeats):
            keys = rng.normal(0, 1/math.sqrt(d), (d, n))
            values = rng.normal(0, 1/math.sqrt(d), (d, n))
            S = (d/n)*(keys@keys.T)
            Z = (d/math.sqrt(n))*(values@keys.T)
            eigen, vectors = np.linalg.eigh(S)
            invroot = (vectors/np.sqrt(eigen))@vectors.T
            independent_gaussian = Z@invroot
            ids = rng.choice(n, design['queries_per_dataset'], replace=False)
            recalled = []
            errors = []
            for method in METHODS:
                U = run_optimizer(Z, S, method)
                scores = values.T@(U@keys[:, ids])
                correct = scores[ids, np.arange(len(ids))]
                ahead = np.count_nonzero(scores>correct[None, :], axis=0)
                # U may be exactly zero on a rare small-d event; then random priorities.
                tied = np.count_nonzero(scores==correct[None, :], axis=0)-1
                if np.any(tied):
                    priorities = rng.random(scores.shape)
                    ahead += np.count_nonzero((scores==correct[None, :]) &
                        (priorities<priorities[ids, np.arange(len(ids))][None, :]), axis=0)
                recall = float(np.mean(ahead<math.ceil(RHO*n)))
                proxy = run_optimizer(independent_gaussian, None, method)
                errors.append(float(np.linalg.norm(U-proxy)/d))
                recalled.append(recall)
                rows.append(dict(d=d, rep=rep, method=method, recall=recall,
                                 frobenius_error=errors[-1]))
            per_dataset.append(recalled); trajectory.append(errors)
        recalls = np.array(per_dataset)
        mean = recalls.mean(axis=0)
        se = recalls.std(axis=0, ddof=1)/math.sqrt(repeats)
        full_summaries.append(dict(d=d, n=n, repeats=repeats,
            methods={method:dict(recall=float(mean[j]), dataset_standard_error=float(se[j]),
                target=pop[method]['recall'], signed_bias=float(mean[j]-pop[method]['recall']),
                absolute_bias_estimate=float(abs(mean[j]-pop[method]['recall'])),
                mean_trajectory_frobenius=float(np.mean(trajectory, axis=0)[j]))
                for j, method in enumerate(METHODS)},
            pairs=[dict(A=METHODS[a], B=METHODS[b], mean=float((recalls[:,a]-recalls[:,b]).mean()),
                dataset_standard_error=float((recalls[:,a]-recalls[:,b]).std(ddof=1)/math.sqrt(repeats)))
                for a,b in PAIRS]))
        print(json.dumps(dict(stage='full_arrays', d=d, seconds=time.monotonic()-started)), flush=True)
    with (OUT/'full_array_rows.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save('full_array_results.json', full_summaries)

    # Separate representation: complete 2d scatter, a Haar column and sphere threshold.
    # This integrates neither the residual root nor the audited draw algorithm.
    # Conditional on the complete scatter, sample the actual D,V event directly.
    sphere_summaries = []
    sphere_rows = []
    for d, repeats in design['sufficient_stat_outer'].items():
        n = math.ceil(KAPPA*d*d); m = n-1; width = 2*d
        hit_rows = []
        for rep in range(repeats):
            lower = np.tril(rng.standard_normal((width,width)), k=-1)
            lower[np.diag_indices(width)] = np.sqrt(rng.chisquare(n-np.arange(width)))
            F = lower/math.sqrt(n)
            Lx, Ly = F[:d], F[d:]
            S = Lx@Lx.T
            Z = math.sqrt(n)*(Ly@Lx.T)
            g = rng.standard_normal(width)
            unit = g/math.sqrt(float(g@g)+rng.chisquare(n-width))
            px, py = Lx@unit, Ly@unit
            wrong = rng.standard_normal(m)
            kth = m-math.ceil(RHO*n)
            threshold = float(np.partition(wrong,kth)[kth]/math.sqrt(float(wrong@wrong)/m))
            hits = []
            for method in METHODS:
                U = run_optimizer(Z, S, method)
                prediction = U@px
                squared_norm = float(np.sum(U*U))
                if squared_norm == 0:
                    hit = float(rng.random()<math.ceil(RHO*n)/n)
                else:
                    D = n*float(py@prediction)/math.sqrt(squared_norm)
                    # ||Ly^T U Lx unit||^2 / ||U||_F^2 is u^T N u.
                    v_component = Ly.T@prediction
                    V = (n*n*float(v_component@v_component)/squared_norm-D*D)/m
                    assert V >= -1e-11
                    hit = float(D>=threshold*math.sqrt(max(V,0.)))
                hits.append(hit)
                sphere_rows.append(dict(d=d, rep=rep, method=method, hit=hit))
            hit_rows.append(hits)
        hits = np.array(hit_rows)
        sphere_summaries.append(dict(d=d, n=n, repeats=repeats,
            methods={method:dict(recall=float(hits[:,j].mean()),
                standard_error=float(hits[:,j].std(ddof=1)/math.sqrt(repeats)),
                target=pop[method]['recall'], signed_bias=float(hits[:,j].mean()-pop[method]['recall']))
                for j,method in enumerate(METHODS)},
            pairs=[dict(A=METHODS[a], B=METHODS[b], mean=float((hits[:,a]-hits[:,b]).mean()),
                standard_error=float((hits[:,a]-hits[:,b]).std(ddof=1)/math.sqrt(repeats)),
                joint_99_hoeffding_radius=math.sqrt(2*math.log(800)/repeats)) for a,b in PAIRS]))
        print(json.dumps(dict(stage='complete_scatter', d=d, seconds=time.monotonic()-started)), flush=True)
    with (OUT/'complete_scatter_rows.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(sphere_rows[0])); writer.writeheader(); writer.writerows(sphere_rows)
    save('complete_scatter_results.json', sphere_summaries)

    thresholds = []
    for a,b in PAIRS:
        ma, mb = METHODS[a], METHODS[b]
        gap = pop[ma]['recall']-pop[mb]['recall']
        ca, cb = dimension_constants(ma,gap,pop), dimension_constants(mb,gap,pop)
        sufficient = max(ca['d0'], cb['d0'], (2*(ca['C']+cb['C'])/gap)**(1/min(ca['rate'],cb['rate'])))
        thresholds.append(dict(A=ma, B=mb, gap=gap, constants_A=ca, constants_B=cb, sufficient_d=sufficient))
    budgets = [finite_budget(d,m,pop) for d in [32,64,128,256,10**6,10**12] for m in METHODS]
    # Recompute reported statistical intervals from raw rows, without importing code.
    with (ROOT/'aa1_checks/AA1_final/AA1_outer_rows.csv').open() as f:
        existing = list(csv.DictReader(f))
    data_checks = []
    for a,b in PAIRS:
        ma,mb=METHODS[a],METHODS[b]
        lows = np.array([float(r[ma+'_lower'])-float(r[mb+'_upper']) for r in existing])
        highs = np.array([float(r[ma+'_upper'])-float(r[mb+'_lower']) for r in existing])
        rad = math.sqrt(2*math.log(800)/len(existing))
        data_checks.append(dict(A=ma, B=mb, rows=len(existing), mean_lower=float(lows.mean()),
            mean_upper=float(highs.mean()), lower=float(lows.mean()-rad), upper=float(highs.mean()+rad)))
    assert digest() == before
    threads = len(list(Path('/proc/self/task').iterdir()))
    assert threads<=2
    save('audit_results.json', dict(population=pop, optimizer_constants={m:optimizer_constants(m) for m in METHODS},
        thresholds=thresholds, budgets=budgets, existing_csv_recomputed=data_checks,
        source_hashes_unchanged=True, resource=dict(memory_max=(cg/'memory.max').read_text().strip(),
            memory_swap_max=(cg/'memory.swap.max').read_text().strip(),
            memory_peak=(cg/'memory.peak').read_text().strip(), threads=threads,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            seconds=time.monotonic()-started)))
    print('FINISHED', flush=True)

if __name__ == '__main__':
    main()
