"""Float64 analytic objectives and optimizers for the preregistered week-1 study."""
from itertools import combinations
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, logsumexp

V0 = np.array([[0, 0], [1, -1], [2, 0], [1, -9 / 8],
               [-1, .5], [-1, .5], [-1, .5]], dtype=np.float64)
STAR = np.array([np.log(2.), 0.])
TARGET = np.array([1 / 3] * 3 + [0.] * 4)
Y0 = np.array([1.] * 3 + [-1.] * 4)
OUT = Path(__file__).resolve().parent / 'week1'


def save(name, value):
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n')
    temp.replace(p)


def memory():
    lines = Path('/proc/self/status').read_text().splitlines()
    return {s.split(':')[0]: int(s.split()[1]) * 1024 for s in lines
            if s.startswith(('VmPeak:', 'VmHWM:', 'VmSize:', 'VmRSS:'))}


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def tangent(g, x):
    return g - (g * x).sum(axis=1, keepdims=True) * x


def data(k, d, seed):
    pos = np.array(list(combinations(range(16), k)), dtype=int)
    A = np.zeros((len(pos), 16), dtype=bool)
    A[np.arange(len(pos))[:, None], pos] = True
    neg = np.array([np.flatnonzero(~row) for row in A])
    rng = np.random.Generator(np.random.PCG64(seed))
    U = normalize(rng.normal(size=(len(pos), d)))
    V = normalize(rng.normal(size=(16, d)))
    return A, pos, neg, U, V


def parameters(k, d, seed, loss, bias, beta):
    A, pos, neg, U, V = data(k, d, seed)
    p = {'U': U, 'V': V}
    if loss == 'sigmoid' and bias != 'zero':
        p['b'] = np.full((len(A),) if bias == 'row' else (1,),
                         np.log(k / (16 - k)))
    if beta == 'train':
        p['a'] = np.array([np.log(16.)])
    return A, pos, neg, p


def objective(p, A, pos, neg, loss, beta, bias='global'):
    """Return mean loss and exact Euclidean gradients of all trainable groups."""
    U, V = p['U'], p['V']
    m, n = A.shape
    k = pos.shape[1]
    inv_temp = float(np.exp(p['a'][0])) if 'a' in p else float(beta)
    S = U @ V.T
    Z = inv_temp * S
    gradients = {}
    if loss == 'nce':
        lse = logsumexp(Z, axis=1, keepdims=True)
        val = float(np.mean(lse[:, 0] - Z[A].reshape(m, k).mean(axis=1)))
        Q = (np.exp(Z - lse) - A / k) / m
    elif loss == 'sigmoid':
        logits = Z + (p['b'][:, None] if bias == 'row' else p.get('b', 0.))
        Y = 2. * A - 1.
        val = float(np.logaddexp(0., -Y * logits).mean())
        Q = -Y * expit(-Y * logits) / (m * n)
        if 'b' in p:
            gradients['b'] = Q.sum(axis=1) if bias == 'row' else np.array([Q.sum()])
    elif loss == 'pe':
        # Only M x (n-k) temporaries per positive slot; includes a zero logit.
        zn = np.take_along_axis(Z, neg, axis=1)
        Q = np.zeros_like(Z)
        val = 0.
        rows = np.arange(m)
        for slot in range(k):
            delta = zn - Z[rows, pos[:, slot]][:, None]
            lse = np.logaddexp(0., logsumexp(delta, axis=1))
            probs = np.exp(delta - lse[:, None]) / (m * k)
            val += float(lse.mean()) / k
            Q[rows[:, None], neg] += probs
            Q[rows, pos[:, slot]] -= probs.sum(axis=1)
    else:
        raise ValueError(loss)
    gradients['U'] = inv_temp * (Q @ V)
    gradients['V'] = inv_temp * (Q.T @ U)
    if 'a' in p:
        gradients['a'] = np.array([(Q * Z).sum()])
    return val, gradients


def metrics(U, V, A):
    S = U @ V.T
    low = np.where(A, S, np.inf).min(axis=1)
    high = np.where(~A, S, -np.inf).max(axis=1)
    gaps = low - high
    k = int(A[0].sum())
    cutoff = np.partition(S, S.shape[1] - k, axis=1)[:, -k]
    above = S > cutoff[:, None]
    tied = S == cutoff[:, None]
    recall = ((above * A).sum(axis=1) +
              (k - above.sum(axis=1)) * (tied * A).sum(axis=1) /
              tied.sum(axis=1)) / k
    return {'success_fraction': float(np.mean(gaps > 0)),
            'recall_at_k': float(recall.mean()), 'min_gap': float(gaps.min()),
            'mean_gap': float(gaps.mean()),
            'positive_score_std': float(np.std(S[A].reshape(len(A), k), axis=1).mean())}


def diagnostics(p, A, pos, beta):
    """Spherical gradients of D, R and the global worst strict ranking margin."""
    U, V = p['U'], p['V']
    m = len(A)
    k = pos.shape[1]
    inv_temp = float(np.exp(p['a'][0])) if 'a' in p else float(beta)
    S = U @ V.T
    Z = inv_temp * S
    lp = logsumexp(np.where(A, Z, -np.inf), axis=1, keepdims=True)
    la = logsumexp(Z, axis=1, keepdims=True)
    pp = np.exp(np.where(A, Z - lp, -np.inf))
    qD = (pp - A / k) / m
    qR = (np.exp(Z - la) - pp) / m
    lo = np.where(A, S, np.inf).min(axis=1)
    hi = np.where(~A, S, -np.inf).max(axis=1)
    gap = lo - hi
    active = np.flatnonzero(np.abs(gap - gap.min()) <= 1e-12)
    H = np.zeros_like(S)
    for row in active:
        ip = np.flatnonzero(A[row] & (np.abs(S[row] - lo[row]) <= 1e-12))
        jn = np.flatnonzero(~A[row] & (np.abs(S[row] - hi[row]) <= 1e-12))
        H[row, ip] = 1. / (len(active) * len(ip))
        H[row, jn] = -1. / (len(active) * len(jn))
    hu, hv = tangent(H @ V, U), tangent(H.T @ U, V)
    result = {'D': float(np.mean(lp[:, 0] - Z[A].reshape(m, k).mean(axis=1) - np.log(k))),
              'R': float(np.mean(la - lp)),
              'h_norm': float(np.sqrt(np.sum(hu * hu) + np.sum(hv * hv))),
              'active_worst_rows': int(len(active))}
    for tag, Q in [('D', qD), ('R', qR)]:
        gu = tangent(inv_temp * (Q @ V), U)
        gv = tangent(inv_temp * (Q.T @ U), V)
        result['minus_g' + tag + '_dot_h'] = float(-np.sum(gu * hu) - np.sum(gv * hv))
        result['g' + tag + '_norm'] = float(np.sqrt(np.sum(gu * gu) + np.sum(gv * gv)))
    return result


class Optimizer:
    def __init__(self, p, method, lr, ratio):
        self.method, self.lr, self.ratio = method, lr, ratio
        self.first = {key: np.zeros_like(val) for key, val in p.items()}
        self.second = {key: np.zeros_like(val) for key, val in p.items()}
        self.step = 0

    def update(self, p, gradients):
        self.step += 1
        for key, value in p.items():
            rate = self.lr * (self.ratio if key == 'V' else .1 if key == 'a' else 1.)
            g = gradients[key]
            if self.method == 'adam':
                self.first[key] *= .9
                self.first[key] += .1 * g
                self.second[key] *= .999
                self.second[key] += .001 * g * g
                update = (self.first[key] / (1 - .9 ** self.step) /
                          (np.sqrt(self.second[key] / (1 - .999 ** self.step)) + 1e-8))
            else:
                update = tangent(g, value) if key in ('U', 'V') else g
            value -= rate * update
            if key in ('U', 'V'):
                value[:] = normalize(value)


def calibrate(p, A, pos, neg, loss, beta, bias, method, ratio):
    def trial(rate):
        trial_p = {key: val.copy() for key, val in p.items()}
        _, grads = objective(trial_p, A, pos, neg, loss, beta, bias)
        Optimizer(trial_p, method, rate, ratio).update(trial_p, grads)
        return float(np.sqrt(np.mean((trial_p['U'] - p['U']) ** 2)))
    initial = trial(.02)
    raw = .001 / initial
    scale = float(np.clip(raw, .1, 10.))
    actual = trial(.02 * scale)
    return {'base_lr': .02, 'trial_rms': initial, 'unclipped_scale': raw,
            'scale': scale, 'lr': .02 * scale, 'actual_rms': actual,
            'clipped': bool(scale != raw),
            'matched_20_percent': bool(abs(actual / .001 - 1) <= .2)}


def train(config, steps=1500, method='adam', half_lr=False, record=True):
    start = time.perf_counter()
    k, d, seed = (config[x] for x in ('k', 'd', 'seed'))
    loss, bias, beta = (config[x] for x in ('loss', 'bias', 'beta'))
    A, pos, neg, p = parameters(k, d, seed, loss, bias, beta)
    calibration = calibrate(p, A, pos, neg, loss, beta, bias, method, config['ratio'])
    rate = calibration['lr'] * (.5 if half_lr else 1.)
    opt = Optimizer(p, method, rate, config['ratio'])
    hist = []
    status = 'completed'
    initial, final = None, None
    for step in range(steps + 1):
        inv_temp = float(np.exp(p['a'][0])) if 'a' in p else float(beta)
        if not np.isfinite(inv_temp) or inv_temp > 1e4 or not all(np.isfinite(v).all() for v in p.values()):
            status = 'temperature_runaway' if 'a' in p else 'numerical_overflow'
            break
        if step % 20 == 0 or step == steps:
            loss_val, _ = objective(p, A, pos, neg, loss, beta, bias)
            row = {'step': step, 'loss': loss_val, 'beta': inv_temp,
                   **metrics(p['U'], p['V'], A), **diagnostics(p, A, pos, beta)}
            if 'b' in p:
                row.update(bias_min=float(p['b'].min()), bias_max=float(p['b'].max()),
                           bias_mean=float(p['b'].mean()))
            if record:
                hist.append(row)
            if initial is None:
                initial = row
            final = row
        if step == steps:
            break
        _, grads = objective(p, A, pos, neg, loss, beta, bias)
        opt.update(p, grads)
    return {'config': config, 'method': method, 'half_lr': half_lr,
            'steps_requested': steps, 'steps_completed': step,
            'calibration': calibration, 'actual_lr': rate,
            'status': status, 'initial': initial, 'final': final,
            'history': hist, 'elapsed_seconds': time.perf_counter() - start}


def core_nce(w, v):
    s = v @ w
    peak = s.max()
    p = np.exp(s - peak)
    total = p.sum()
    p /= total
    residual = p - TARGET
    return (float(peak + np.log(total) - s[:3].mean()), residual @ v,
            residual[:, None] * w[None, :], p)


def core_metrics(w, v):
    s = v @ w
    gap = float(s[:3].min() - s[3:].max())
    denominator = float(np.linalg.norm(w) * np.linalg.norm(v, axis=1).max())
    # Expected recall under uniform tie-breaking; strict success uses gap only.
    cutoff = np.partition(s, 4)[4]
    above, tied = s > cutoff, s == cutoff
    recall = (above[:3].sum() + (3 - above.sum()) * tied[:3].sum() / tied.sum()) / 3
    return {'gap': gap, 'recall_at_3': float(recall), 'success': bool(gap > 0),
            'normalized_gap': gap / denominator if denominator else None,
            'degenerate_norm': denominator == 0,
            'w_norm': float(np.linalg.norm(w)), 'max_v_norm': float(np.linalg.norm(v, axis=1).max())}
