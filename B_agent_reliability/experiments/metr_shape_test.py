"""Test: under constant within-task hazard + between-task heterogeneity, S(t)=E[exp(-lam t)]
is completely monotone; log-logistic S(t)=1/(1+(t/h)^k) is CM iff k<=1.
Fit METR-style weighted logistic in log(t) per model; report shape k (natural-log units) with bootstrap CI."""
import json, sys, numpy as np
from scipy.optimize import minimize
path = sys.argv[1]
rs = [json.loads(l) for l in open(path)]
rs = [r for r in rs if r['alias'] != 'human' and r.get('human_minutes') and r['human_minutes'] > 0]
rng = np.random.default_rng(0)

def fit(lt, y, w):
    # P(success) = sigmoid(a - k * ln t)  -> S(t) = 1/(1+(t/h)^k), h=exp(a/k)
    def nll(p):
        a, k = p; z = a - k * lt
        return -np.sum(w * (y * -np.logaddexp(0, -z) + (1 - y) * -np.logaddexp(0, z)))
    res = minimize(nll, [1.0, 0.5], method='L-BFGS-B')
    a, k = res.x
    return a, k

print(f"{'model':40s} {'n':>5s} {'h50(min)':>9s} {'k':>6s} {'95% CI':>15s}")
models = sorted(set(r['alias'] for r in rs))
out = []
for m in models:
    R = [r for r in rs if r['alias'] == m]
    lt = np.log(np.array([r['human_minutes'] for r in R]))
    y = np.array([float(r['score_binarized']) for r in R])
    w = np.array([r.get('invsqrt_task_weight') or 1.0 for r in R]); w = w / w.mean()
    a, k = fit(lt, y, w)
    tasks = np.array([r['task_id'] for r in R]); ut = np.unique(tasks)
    idx = {t: np.where(tasks == t)[0] for t in ut}
    ks = []
    for b in range(200):  # task-level bootstrap
        sel = np.concatenate([idx[t] for t in rng.choice(ut, len(ut))])
        ks.append(fit(lt[sel], y[sel], w[sel])[1])
    lo, hi = np.percentile(ks, [2.5, 97.5])
    h = np.exp(a / k)
    out.append((m, len(R), h, k, lo, hi))
    print(f"{m:40s} {len(R):5d} {h:9.1f} {k:6.2f} [{lo:5.2f},{hi:5.2f}]", flush=True)
ks = np.array([o[3] for o in out])
print(f"\nmodels with k>1 (CI excludes 1): {sum(o[4]>1 for o in out)}/{len(out)}; k<1 (CI excludes 1): {sum(o[5]<1 for o in out)}; median k={np.median(ks):.2f}")
