"""EXP2 only: Gaussian snapshots, scalar normalizer, and fresh-data Muon."""
import time
import numpy as np

R = 4
DIM = 256
NUS = (.25, .5, 1., 2., 4., 8.)
TIMES = (0, 1, 4, 16, 64, 256, 1024, 2048)


def frames(rng, n):
    out = []
    for _ in range(n // R):
        q, rr = np.linalg.qr(rng.normal(size=(R, R)))
        out.append(q * np.where(np.diag(rr) >= 0, 1., -1.))
    return np.concatenate(out)


def problem(seed, n, alpha):
    v = frames(np.random.default_rng([20261106, 1, seed]), DIM)
    i = np.arange(1, DIM + 1, dtype=float)
    cov = (v[n:].T * i[n:] ** -2) @ v[n:] + np.eye(R) / R
    eig, vec = np.linalg.eigh(cov)
    return dict(n=n, alpha=alpha, lam=i[:n] ** -alpha,
                teacher=i[:n, None] ** ((alpha - 2) / 2) * v[:n],
                cov=cov, root=vec * np.sqrt(eig), tail=float(np.sum(i[n:] ** -2)))


def snapshot(seed, n, alpha, ai, amplitude):
    rng = np.random.default_rng([20261106, 2, seed + 10000, n, int(2 * alpha), ai])
    v = frames(rng, n)
    q = np.repeat(rng.uniform(.8, 1.25, size=n // R), R)
    i = np.arange(1, n + 1, dtype=float)
    return -amplitude * (i ** ((alpha - 2) / 2) * q)[:, None] * v


def polar(g):
    u, s, vt = np.linalg.svd(g, full_matrices=False)
    active = s > np.finfo(float).eps * max(g.shape) * s[0] if s[0] else np.zeros_like(s, bool)
    return (u * active) @ vt


def moments(p, e, b, nu):
    lam = p['lam']
    pp = 1 / np.sqrt(lam + float(b) ** -nu)
    a = (pp * lam)[:, None] * e
    s = e.T @ (lam[:, None] * e) + p['cov']
    rowvar = pp ** 2 * (lam * np.trace(s) + lam ** 2 * np.sum(e * e, axis=1)) / b
    output = (np.sum(pp ** 2 * lam) * s + a.T @ a) / b
    riskvar = float(np.dot(lam, rowvar))
    gamma = float(np.sqrt(R / (np.sum(a * a) + np.sum(rowvar))))
    return dict(P=pp, A=a, gamma=gamma, rowvar=rowvar, output=output, riskvar=riskvar,
                h1=float(pp[0] * lam[0]))


def norm2(x, lam):
    return float(np.sum(lam[:, None] * x * x))


def statistics(values, e, lam, a, pred, fixed):
    """Linear-projection residuals, unbiased squared-mean noise corrections."""
    count = len(values)
    mu = values.mean(axis=0)
    centered = values - mu
    rowcov = np.einsum('sni,snj->nij', centered, centered) / (count - 1)
    noise = float(np.einsum('n,nii->', lam, rowcov))
    output = rowcov.sum(axis=0)
    energy = norm2(a, lam)
    coeff = np.einsum('n,nj,snj->s', lam, a, values) / energy
    scalar_noise = float(energy * np.var(coeff, ddof=1))
    shape_raw = norm2(mu, lam) - energy * float(coeff.mean()) ** 2
    erow = np.sum(e * e, axis=1)
    rowcoef = np.einsum('sni,ni->sn', values, e) / erow
    orth_mean = mu - rowcoef.mean(axis=0)[:, None] * e
    orth_noise = noise - float(np.sum(lam * erow * np.var(rowcoef, axis=0, ddof=1)))
    inverse = np.linalg.solve(a.T @ (lam[:, None] * a), a.T * lam)
    rightcoef = np.einsum('in,snj->sij', inverse, values)
    right = np.einsum('ni,sij->snj', a, rightcoef) - coeff[:, None, None] * a
    rightmean = right.mean(axis=0)
    rightnoise = float(np.sum(lam[:, None] * np.var(right, axis=0, ddof=1)))
    return dict(N=count, prediction_energy=norm2(pred, lam),
                prediction_error=norm2(mu - pred, lam) - noise / count,
                fixed_energy=norm2(fixed, lam),
                fixed_error=norm2(mu - fixed, lam) - noise / count,
                signal=norm2(mu, lam) - noise / count,
                mean_noise=noise / count,
                gain_numerator=float(np.sum(lam[:, None] * pred * mu)),
                shape_error=shape_raw - (noise - scalar_noise) / count,
                orth_error=norm2(orth_mean, lam) - orth_noise / count,
                coupling_error=norm2(rightmean, lam) - rightnoise / count,
                noise_risk=noise, noise_output=output.tolist()), mu, rowcov


def measure(ctx, stage, seed, n, alpha, ai, amplitude, b, count, nu, c, nufix, cfix):
    p = problem(seed, n, alpha)
    e = snapshot(seed, n, alpha, ai, amplitude)
    m = moments(p, e, b, nu)
    mf = moments(p, e, b, nufix)
    pred = c * m['gamma'] * m['A']
    fixed = cfix * mf['A']
    rng = np.random.default_rng([20261106, stage, seed, n, int(alpha * 2), ai, b])
    q = np.empty((count, n, R))
    pg = np.empty_like(q)
    sample_time = gradient_time = svd_time = 0.
    started = time.perf_counter()
    for j in range(count):
        if j % 128 == 0:
            ctx.guard()
        tic = time.perf_counter()
        x = rng.normal(size=(b, n)) * np.sqrt(p['lam'])
        eps = rng.normal(size=(b, R)) @ p['root'].T
        sample_time += time.perf_counter() - tic
        tic = time.perf_counter()
        g = x.T @ (x @ e + eps) / b
        gradient_time += time.perf_counter() - tic
        tic = time.perf_counter()
        q[j] = polar(g)
        svd_time += time.perf_counter() - tic
        pg[j] = m['gamma'] * m['P'][:, None] * g
    pieces = dict(full=slice(None), first=slice(0, count // 2), second=slice(count // 2, None))
    metrics = {}
    arrays = dict(E=e, lam=p['lam'], Sigma0=p['cov'],
                  q_blockmeans=q.reshape(16, count // 16, n, R).mean(axis=1),
                  pg_blockmeans=pg.reshape(16, count // 16, n, R).mean(axis=1))
    for part, sl in pieces.items():
        s, mu, rc = statistics(q[sl], e, p['lam'], m['A'], pred, fixed)
        ctrl, cm, cr = statistics(pg[sl], e, p['lam'], m['A'], m['gamma'] * m['A'], m['gamma'] * m['A'])
        ctrl['predicted_noise_risk'] = m['gamma'] ** 2 * m['riskvar']
        s['control'] = ctrl
        metrics[part] = s
        arrays['mean_' + part] = mu
        arrays['rowcov_' + part] = rc
        arrays['pg_mean_' + part] = cm
        arrays['pg_rowcov_' + part] = cr
    rates = np.einsum('sni,ni->sn', q, e) / np.sum(e * e, axis=1)
    arrays['rates'] = rates.mean(axis=0)
    arrays['rate_se'] = rates.std(axis=0, ddof=1) / np.sqrt(count)
    if n == 32 and seed == 2001 and b == 16:
        arrays['full_covariance'] = np.cov(q.reshape(count, -1), rowvar=False)
    name = f's{seed}_n{n}_a{alpha}_amp{ai}_b{b}'
    np.savez_compressed(ctx.out / (name + '.npz'), **arrays)
    return dict(seed=seed, snapshot_seed=seed + 10000, n=n, alpha=alpha, ai=ai, amplitude=amplitude,
                b=b, batches=count, file=name + '.npz', metrics=metrics,
                gamma0=m['gamma'], gain=c*m['gamma'],
                predicted_noise_risk=(c*m['gamma'])**2*m['riskvar'],
                predicted_noise_output=((c*m['gamma'])**2*m['output']).tolist(),
                sample_seconds=sample_time, gradient_seconds=gradient_time, svd_seconds=svd_time,
                elapsed_seconds=time.perf_counter()-started, samples=count*b)


def fit(ctx, records, normalized):
    candidates = []
    for nu in NUS:
        numerator = denominator = 0.
        targets = []
        for row in records:
            with np.load(ctx.out / row['file']) as z:
                e, mu = z['E'], z['mean_full']
            p = problem(row['seed'], row['n'], row['alpha'])
            m = moments(p, e, row['b'], nu)
            a = m['A'] * (m['gamma'] if normalized else 1.)
            numerator += float(np.sum(p['lam'][:, None] * a * mu))
            denominator += norm2(a, p['lam'])
            targets.append((p['lam'], a, mu))
        c = numerator / denominator
        error = sum(norm2(mu - c*a, lam) for lam, a, mu in targets)
        candidates.append(dict(nu=nu, c=c, calibration_squared_error=error))
    chosen = min(candidates, key=lambda x: (x['calibration_squared_error'], x['nu']))
    if chosen['c'] <= 0:
        raise ValueError('nonpositive locked gain')
    return dict(**chosen, candidates=candidates, endpoint=chosen['nu'] in (NUS[0], NUS[-1]))


def family(e, lam):
    rowenergy = np.sum(e*e, axis=1)
    z = lam * rowenergy * np.arange(1, len(e)+1, dtype=float)**2
    directions = e / np.sqrt(np.maximum(rowenergy, 1e-300))[:, None]
    eigs = [np.linalg.eigvalsh(v.T@v) for v in directions.reshape(-1, R, R)]
    lo, hi = float(np.min(eigs)), float(np.max(eigs))
    ratio = float(np.max(z)/max(np.min(z), 1e-300))
    return dict(energy_ratio=ratio, frame_min=lo, frame_max=hi,
                in_relaxed_family=bool(ratio <= (1.25/.8)**2 and lo >= .5 and hi <= 1.5))


def train(ctx, seed, n, alpha, b, eta, steps, nu, c, save_snapshots=False):
    p = problem(seed, n, alpha)
    e = -p['teacher'].copy()
    rng = np.random.default_rng([20261106, 30, seed, n, int(2*alpha), b])
    rows = []
    sample_time = gradient_time = svd_time = gamma_time = 0.
    cumulative = 0.
    started = time.perf_counter()
    for t in range(steps + 1):
        if t % 128 == 0:
            ctx.guard()
        tic = time.perf_counter()
        m = moments(p, e, b, nu)
        gamma = c*m['gamma']
        gamma_time += time.perf_counter()-tic
        if t in TIMES or t == steps:
            row = dict(T=t, D=b*t, nD=n*b*t, gamma=gamma, cumulative_gamma=cumulative,
                       proxy_stability=eta*gamma*m['h1'],
                       risk=norm2(e, p['lam'])+p['tail'], tail=p['tail'], family=family(e, p['lam']),
                       sample_seconds=sample_time, gradient_seconds=gradient_time,
                       svd_seconds=svd_time, gamma_seconds=gamma_time,
                       wall_seconds=time.perf_counter()-started)
            # Invert cumulative-gain clock for a descriptive learned-scale diagnostic.
            rates = eta*cumulative*p['lam']/np.sqrt(p['lam']+float(b)**-nu)
            learned = np.flatnonzero(rates >= 1.)
            row['clock_h'] = int(learned[-1]+1) if len(learned) else 0
            rows.append(row)
            if save_snapshots:
                np.savez_compressed(ctx.out/f'train_s{seed}_n{n}_a{alpha}_b{b}_t{t}.npz',
                                    E=e, lam=p['lam'])
        if t == steps:
            break
        tic = time.perf_counter()
        x = rng.normal(size=(b, n))*np.sqrt(p['lam'])
        eps = rng.normal(size=(b, R))@p['root'].T
        sample_time += time.perf_counter()-tic
        tic = time.perf_counter()
        g = x.T@(x@e+eps)/b
        gradient_time += time.perf_counter()-tic
        tic = time.perf_counter()
        delta = polar(g)
        svd_time += time.perf_counter()-tic
        e -= eta*delta
        cumulative += gamma
        if not np.all(np.isfinite(e)):
            raise FloatingPointError('nonfinite online residual')
    return dict(seed=seed,n=n,alpha=alpha,b=b,eta=eta,steps=steps,rows=rows,
                sample_seconds=sample_time,gradient_seconds=gradient_time,svd_seconds=svd_time,
                gamma_seconds=gamma_time,elapsed_seconds=time.perf_counter()-started,samples=b*steps)
