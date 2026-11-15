"""Frozen round-2 CPU experiment; shared exact objectives from week 1."""
import os
for key, value in [('OMP_NUM_THREADS', '3'), ('OPENBLAS_NUM_THREADS', '1'),
                   ('MKL_NUM_THREADS', '1'), ('NUMEXPR_NUM_THREADS', '1')]:
    os.environ[key] = value
import resource
resource.setrlimit(resource.RLIMIT_AS, (3500000000, 3500000000))
import copy
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from week1_common import objective, normalize, memory

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SETTINGS = [(20, 2, d) for d in (5, 6, 8)] + [(16, 3, d) for d in (6, 8, 10)]
LOSSES = ['nce', 'pe', 'sigmoid']
QUANTILES = [0, .05, .1, .25, .5, .75, .9, .95, 1]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def init(n, k, d, seed):
    pos = np.array(list(combinations(range(n), k)))
    A = np.zeros((len(pos), n), bool)
    A[np.arange(len(pos))[:, None], pos] = True
    neg = np.array([np.flatnonzero(~a) for a in A])
    rng = np.random.Generator(np.random.PCG64(seed))
    p = dict(U=normalize(rng.normal(size=(len(A), d))),
             V=normalize(rng.normal(size=(n, d))), a=np.array([np.log(16.)]))
    return A, pos, neg, p


def obj(p, A, pos, neg, loss):
    return objective(p, A, pos, neg, loss, 'train', 'zero')


class Adam:
    def __init__(self, p, rates):
        self.rates = rates
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.t = 0

    def update(self, p, g, factor=1.):
        self.t += 1
        for key in p:
            self.m[key] *= .9
            self.m[key] += .1*g[key]
            self.v[key] *= .999
            self.v[key] += .001*g[key]*g[key]
            direction = (self.m[key]/(1-.9**self.t))/(
                np.sqrt(self.v[key]/(1-.999**self.t))+1e-8)
            p[key] -= factor*self.rates[key]*direction
            if key in ('U', 'V'):
                p[key][:] = normalize(p[key])


def displacement(p0, p):
    return {key: dict(rms=float(np.sqrt(np.mean((p[key]-p0[key])**2))),
                      frobenius=float(np.linalg.norm(p[key]-p0[key]))) for key in p}


def calibrate(p, g):
    rates = {}
    for key in ('U', 'V'):
        direction = g[key]/(np.abs(g[key])+1e-8)
        def rms(rate):
            return float(np.sqrt(np.mean((normalize(p[key]-rate*direction)-p[key])**2)))
        lo, hi = 0., .02
        for _ in range(60):
            if rms(hi) >= .001:
                break
            hi *= 2
        else:
            raise ValueError('cannot bracket calibration root')
        for _ in range(60):
            mid = (lo+hi)/2
            if rms(mid) < .001:
                lo = mid
            else:
                hi = mid
        rates[key] = (lo+hi)/2
    rates['a'] = .1*rates['U']
    q = copy.deepcopy(p)
    Adam(q, rates).update(q, g)
    moves = displacement(p, q)
    assert all(abs(moves[key]['rms']/.001-1) <= 1e-6 for key in ('U', 'V'))
    return rates, moves


def gaps(p, A):
    S = p['U']@p['V'].T
    low = np.where(A, S, np.inf).min(1)
    high = np.where(~A, S, -np.inf).max(1)
    return low-high, np.where(A, S, -S).min(1), S


def metrics(p, A, arrays=False):
    gamma, zeta, S = gaps(p, A)
    k = int(A[0].sum())
    cutoff = np.partition(S, S.shape[1]-k, axis=1)[:, -k]
    above, tied = S > cutoff[:, None], S == cutoff[:, None]
    recall = ((above*A).sum(1)+(k-above.sum(1))*(tied*A).sum(1)/tied.sum(1))/k
    result = dict(success_fraction=float((gamma>0).mean()), recall_at_k=float(recall.mean()),
                  min_gap=float(gamma.min()), mean_gap=float(gamma.mean()),
                  gamma_quantiles=np.quantile(gamma, QUANTILES).tolist(),
                  global_zero_sign_margin=float(zeta.min()),
                  sign_row_success_fraction=float((zeta>0).mean()),
                  sign_element_success_fraction=float((np.where(A, S, -S)>0).mean()),
                  zeta_quantiles=np.quantile(zeta, QUANTILES).tolist(),
                  beta=float(np.exp(p['a'][0])),
                  scaled_global_sign_margin=float(np.exp(p['a'][0])*zeta.min()),
                  positive_score_std=float(S[A].reshape(len(A), k).std(1).mean()))
    if arrays:
        result.update(gamma=gamma.tolist(), zeta=zeta.tolist())
    return result


def d_gradient(p, A, pos):
    Z = np.exp(p['a'][0])*(p['U']@p['V'].T)
    lp = logsumexp(np.where(A, Z, -np.inf), axis=1, keepdims=True)
    Q = (np.exp(np.where(A, Z-lp, -np.inf))-A/pos.shape[1])/len(A)
    g = dict(U=np.exp(p['a'][0])*(Q@p['V']),
             V=np.exp(p['a'][0])*(Q.T@p['U']), a=np.array([(Q*Z).sum()]))
    D = float(np.mean(lp[:, 0]-Z[A].reshape(len(A), -1).mean(1)-np.log(pos.shape[1])))
    return D, g


def counterfactual(p, opt, A, pos, neg, factor):
    before = metrics(p, A)
    gamma0, _, S = gaps(p, A)
    D, gd = d_gradient(p, A, pos)
    val, full = obj(p, A, pos, neg, 'nce')
    noD = {key: full[key]-gd[key] for key in full}
    branches, scores = {}, {}
    for tag, grad in [('with_D', full), ('without_D', noD)]:
        q, state = copy.deepcopy(p), copy.deepcopy(opt)
        state.update(q, grad, factor)
        met = metrics(q, A)
        gamma, _, sq = gaps(q, A)
        branches[tag] = dict(metrics=met, displacement=displacement(p, q),
                             gamma_delta=(gamma-gamma0).tolist())
        scores[tag] = sq
    # Preserve the original active worst pair, allowing averaging of exact ties.
    active = np.flatnonzero(np.abs(gamma0-gamma0.min()) <= 1e-12)
    effects = []
    for row in active:
        low = S[row, A[row]].min()
        high = S[row, ~A[row]].max()
        ips = A[row] & (np.abs(S[row]-low) <= 1e-12)
        jns = (~A[row]) & (np.abs(S[row]-high) <= 1e-12)
        diff = scores['without_D'][row]-scores['with_D'][row]
        effects.append(float(diff[ips].mean()-diff[jns].mean()))
    w, wo = (branches[x]['metrics'] for x in ('with_D', 'without_D'))
    return dict(D=D, R=val-np.log(pos.shape[1])-D, branches=branches,
                original_active_pair_effect=float(np.mean(effects)),
                q10_effect=wo['gamma_quantiles'][2]-w['gamma_quantiles'][2],
                min_gap_effect=wo['min_gap']-w['min_gap'],
                success_fraction_effect=wo['success_fraction']-w['success_fraction'],
                before=before)


def checkpoints(T):
    return sorted(set([x for x in (0, 1000, 5000, 10000, 25000, 50000, 75000, T) if x<=T]
                      + [round(T*f) for f in (.1, .25, .5, .75)]))


def run(config, T, deadline):
    start = time.time()
    A, pos, neg, p = init(*(config[x] for x in ('n', 'k', 'd', 'seed')))
    fingerprint = hashlib.sha256(p['U'].tobytes()+p['V'].tobytes()).hexdigest()
    _, g = obj(p, A, pos, neg, config['loss'])
    rates, moves = calibrate(p, g)
    opt = Adam(p, rates)
    hist, cfs, status, latest_move = [], [], 'completed', None
    initial = metrics(p, A, True)
    check = set(checkpoints(T))
    t = 0
    for t in range(T+1):
        a = p['a'][0]
        if (not all(np.isfinite(v).all() for v in p.values()) or
                not np.log(1e-8)<=a<=np.log(1e4)):
            status = 'numerical_or_temperature_failure'
            break
        if t%1000 == 0 and time.time()>deadline:
            status = 'budget_truncated'
            break
        if t in check:
            hist.append(dict(step=t, loss=obj(p, A, pos, neg, config['loss'])[0],
                             last_actual_displacement=latest_move, **metrics(p, A)))
            if t<T:
                factor = .5*(1+np.cos(np.pi*t/T))
                cfs.append(dict(step=t, cosine_factor=factor,
                                **counterfactual(p, opt, A, pos, neg, factor)))
        if t==T:
            break
        _, g = obj(p, A, pos, neg, config['loss'])
        capture = t+1 in check
        prev = copy.deepcopy(p) if capture else None
        opt.update(p, g, .5*(1+np.cos(np.pi*t/T)))
        if capture:
            latest_move = displacement(prev, p)
    final = metrics(p, A, True) if all(np.isfinite(v).all() for v in p.values()) else None
    return dict(config=config, initial_fingerprint=fingerprint, steps_requested=T,
                steps_completed=t, status=status, calibration=dict(rates=rates, actual=moves),
                initial=initial, final=final, history=hist, counterfactuals=cfs,
                elapsed_seconds=time.time()-start, memory=memory())
