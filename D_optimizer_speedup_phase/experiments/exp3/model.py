"""EXP3: zero-fit population-output-Gram closure. CPU, local modules only."""
import time
import numpy as np

R = 4
DIM = 256
TIMES = (0, 32, 128, 512)


def frames(rng, n):
    blocks = []
    for _ in range(n // R):
        q, z = np.linalg.qr(rng.normal(size=(R, R)))
        blocks.append(q * np.where(np.diag(z) >= 0, 1., -1.))
    return np.concatenate(blocks)


def problem(seed, n, alpha):
    v = frames(np.random.default_rng([20261106, 300, seed]), DIM)
    i = np.arange(1, DIM + 1, dtype=float)
    cov = (v[n:].T * i[n:] ** -2) @ v[n:] + np.eye(R) / R
    eig, vec = np.linalg.eigh(cov)
    return dict(n=n, alpha=alpha, lam=i[:n] ** -alpha,
                teacher=i[:n, None] ** ((alpha - 2) / 2) * v[:n],
                cov=cov, root=vec * np.sqrt(eig), tail=float(np.sum(i[n:] ** -2)))


def snapshot(seed, n, alpha, ai, amplitude):
    rng = np.random.default_rng([20261106, 301, seed, n, int(2 * alpha), ai])
    v = frames(rng, n)
    q = np.repeat(rng.uniform(.8, 1.25, n // R), R)
    return -amplitude * (np.arange(1, n + 1) ** ((alpha - 2) / 2) * q)[:, None] * v


def polar(g):
    u, s, vt = np.linalg.svd(g, full_matrices=False)
    active = s > np.finfo(float).eps * max(g.shape) * s[0] if s[0] else np.zeros_like(s, bool)
    return (u * active) @ vt


def norm2(z, lam):
    return float(np.sum(lam[:, None] * z * z))


def bridge(p, e, b):
    lam = p['lam']
    a = lam[:, None] * e
    s = e.T @ (lam[:, None] * e) + p['cov']
    aa = a.T @ a
    m = (1 + 1 / b) * aa + np.sum(lam) * s / b
    eig, vec = np.linalg.eigh(m)
    assert np.min(eig) > 0
    right = (vec * (1 / np.sqrt(eig))) @ vec.T
    pred = a @ right
    out = right @ (np.sum(lam) * s + aa) @ right / b
    risk = float(np.trace(right @ (np.sum(lam ** 2) * s + e.T @ (lam[:, None] ** 3 * e)) @ right) / b)
    # The scalar ablation is the risk projection of the frozen prediction, not a fit to Q.
    scalar = a * (np.sum(lam[:, None] * a * pred) / norm2(a, lam))
    return dict(A=a, S=s, M=m, B=right, pred=pred, scalar=scalar,
                output=out, riskvar=risk, totalrisk=norm2(pred, lam) + risk,
                mmin=float(eig[0]), bmin=float(1 / np.sqrt(eig[-1])),
                bmax=float(1 / np.sqrt(eig[0])))


def statistics(values, lam, pred, scalar):
    count = len(values)
    mu = values.mean(axis=0)
    rowcov = np.einsum('sni,snj->nij', values - mu, values - mu) / (count - 1)
    noise = float(np.einsum('n,nii->', lam, rowcov))
    return dict(N=count, prediction_energy=norm2(pred, lam),
                prediction_error=norm2(mu - pred, lam) - noise / count,
                scalar_energy=norm2(scalar, lam),
                scalar_error=norm2(mu - scalar, lam) - noise / count,
                signal=norm2(mu, lam) - noise / count, mean_noise=noise / count,
                gain_numerator=float(np.sum(lam[:, None] * pred * mu)),
                noise_risk=noise, noise_output=rowcov.sum(axis=0).tolist(),
                total_risk=float(np.mean(np.sum(lam[None, :, None] * values ** 2, axis=(1, 2))))), mu, rowcov


def measure(ctx, stage, seed, p, e, b, count, tag):
    started = time.perf_counter()
    tic = time.perf_counter()
    m = bridge(p, e, b)
    bridge_time = time.perf_counter() - tic
    n, alpha, lam = p['n'], p['alpha'], p['lam']
    # tag is encoded without Python's salted hash.
    rng = np.random.default_rng([20261106, stage, seed, n, int(2 * alpha), b, *tag])
    q = np.empty((count, n, R)); u = np.empty_like(q)
    grams = np.empty((count, R, R))
    deviations = np.empty(count); errors = np.empty(count)
    sample_time = gradient_time = svd_time = 0.
    for j in range(count):
        if j % 128 == 0:
            ctx.guard()
        tic = time.perf_counter()
        x = rng.normal(size=(b, n)) * np.sqrt(lam)
        eps = rng.normal(size=(b, R)) @ p['root'].T
        sample_time += time.perf_counter() - tic
        tic = time.perf_counter()
        g = x.T @ (x @ e + eps) / b
        gradient_time += time.perf_counter() - tic
        tic = time.perf_counter()
        q[j] = polar(g)
        svd_time += time.perf_counter() - tic
        u[j] = g @ m['B']
        grams[j] = g.T @ g
        deviations[j] = np.linalg.norm(grams[j] - m['M'], ord=2) / m['mmin']
        errors[j] = np.sum((q[j] - u[j]) ** 2)
    arrays = dict(E=e, lam=lam, Sigma0=p['cov'], M=m['M'], B=m['B'], prediction=m['pred'],
                  scalar=m['scalar'], gram_mean=grams.mean(axis=0),
                  q_blockmeans=q.reshape(16, count // 16, n, R).mean(axis=1),
                  u_blockmeans=u.reshape(16, count // 16, n, R).mean(axis=1),
                  gram_blockmeans=grams.reshape(16, count // 16, R, R).mean(axis=1))
    metrics = {}
    for part, sl in dict(full=slice(None), first=slice(0, count // 2), second=slice(count // 2, None)).items():
        st, mu, rc = statistics(q[sl], lam, m['pred'], m['scalar'])
        ctrl, cm, cr = statistics(u[sl], lam, m['pred'], m['scalar'])
        ctrl['predicted_noise_risk'] = m['riskvar']
        st['control'] = ctrl
        metrics[part] = st
        arrays['mean_' + part] = mu; arrays['rowcov_' + part] = rc
        arrays['u_mean_' + part] = cm; arrays['u_rowcov_' + part] = cr
    if n == 32 and seed == 6101 and b == 16:
        arrays['full_covariance'] = np.cov(q.reshape(count, -1), rowvar=False)
    file = f'{stage}_s{seed}_n{n}_a{alpha}_b{b}_tag' + '_'.join(map(str, tag)) + '.npz'
    np.savez_compressed(ctx.out / file, **arrays)
    gm_noise = float(np.sum(np.var(grams, axis=0, ddof=1)) / count)
    gm_error = float(np.sum((grams.mean(axis=0) - m['M']) ** 2) - gm_noise)
    pbad = float(np.mean(deviations > .5))
    k4 = float(np.mean(np.sum(u * u, axis=(1, 2)) ** 2))
    lip = np.sqrt(1.5 * np.trace(m['M']) / m['mmin']) * .5 / (2 * .5 ** 1.5)
    return dict(seed=seed, n=n, alpha=alpha, b=b, tag=list(tag), batches=count, file=file,
                metrics=metrics, predicted_noise_risk=m['riskvar'],
                predicted_noise_output=m['output'].tolist(), predicted_totalrisk=m['totalrisk'],
                gram_energy=float(np.sum(m['M'] ** 2)), gram_error=gm_error, gram_mean_noise=gm_noise,
                gram_bad_fraction=pbad, gram_relative_mean=float(np.mean(deviations)),
                gram_relative_max=float(np.max(deviations)), k4=k4,
                empirical_delta2=float(lip ** 2 + 2 * R * pbad + 2 * np.sqrt(k4 * pbad)),
                actual_coupled_delta2=float(np.mean(errors)),
                sample_seconds=sample_time, gradient_seconds=gradient_time,
                svd_seconds=svd_time, bridge_seconds=bridge_time,
                elapsed_seconds=time.perf_counter() - started, samples=count * b)


def train(ctx, seed, n, alpha, b, steps=512, eta=.01):
    p = problem(seed, n, alpha)
    e0 = -p['teacher'].copy(); e = e0.copy()
    rng = np.random.default_rng([20261106, 304, seed, n, int(2 * alpha), b])
    started = time.perf_counter()
    sample_time = gradient_time = svd_time = bridge_time = 0.
    records = []; snapshots = []
    radius_sum = 0.
    for t in range(steps + 1):
        if t % 64 == 0:
            ctx.guard()
        tic = time.perf_counter()
        m = bridge(p, e, b)
        bridge_time += time.perf_counter() - tic
        distance = float(np.linalg.norm(e - e0))
        assert distance <= np.sqrt(R) * radius_sum + 1e-10
        if t in TIMES or t == steps:
            snapshots.append((t, e.copy()))
            records.append(dict(T=t, D=b*t, nD=n*b*t, V=norm2(e, p['lam']),
                risk=norm2(e, p['lam'])+p['tail'], tail=p['tail'], distance=distance,
                reachable_radius=float(np.sqrt(R)*radius_sum),
                bmin=m['bmin'], bmax=m['bmax'], eta_bmax_lambda1=eta*m['bmax']*p['lam'][0],
                predicted_alignment=float(np.sum(p['lam'][:, None]*e*m['pred'])),
                cumulative_eta=radius_sum))
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
        e -= eta*polar(g)
        svd_time += time.perf_counter()-tic
        radius_sum += eta
    # Measurements use a different namespace; they do not feed back into training.
    training_elapsed = time.perf_counter()-started
    tests = []
    for t, es in snapshots:
        test = measure(ctx, 305, seed, p, es, b, 4096, (t,))
        test['T'] = t
        test['V'] = norm2(es, p['lam'])
        test['predicted_alignment'] = float(np.sum(p['lam'][:, None]*es*bridge(p, es, b)['pred']))
        with np.load(ctx.out/test['file']) as z:
            test['measured_alignment'] = float(np.sum(p['lam'][:, None]*es*z['mean_full']))
        # Exact fresh-batch risk increment reconstructed from first and second moments.
        test['measured_increment'] = -2*eta*test['measured_alignment'] + eta**2*test['metrics']['full']['total_risk']
        test['predicted_increment'] = -2*eta*test['predicted_alignment'] + eta**2*test['predicted_totalrisk']
        tests.append(test)
    return dict(seed=seed,n=n,alpha=alpha,b=b,eta=eta,steps=steps,rows=records,tests=tests,
                sample_seconds=sample_time,gradient_seconds=gradient_time,svd_seconds=svd_time,
                bridge_seconds=bridge_time,elapsed_seconds=training_elapsed,samples=b*steps,
                diagnostic_elapsed_seconds=sum(s['elapsed_seconds'] for s in tests),
                diagnostic_samples=sum(s['samples'] for s in tests))
