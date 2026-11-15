"""Independent checks; no imports of project implementation and no network."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    assert os.environ[key] == "2"
import json
import ctypes
import math
import resource
import sys
import time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent
START = time.monotonic()
RNG = np.random.default_rng(84627613)

def threadpool_info():
    found=[]
    libs=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if 'openblas' in line and '/' in line})
    for path in libs:
        lib=ctypes.CDLL(path)
        for symbol in ('openblas_get_num_threads','openblas_get_num_threads64_',
                       'scipy_openblas_get_num_threads','scipy_openblas_get_num_threads64_'):
            try:
                func=getattr(lib,symbol)
            except AttributeError:
                continue
            func.restype=ctypes.c_int
            found.append(dict(library=path,symbol=symbol,num_threads=func()))
            break
    return found

def scalar_update(r, method, steps, step, eps=0.1, beta1=0.9, beta2=0.99):
    u = np.zeros_like(r)
    m = np.zeros_like(r)
    v = np.zeros_like(r)
    for t in range(1, steps+1):
        grad = u-r
        if method == "GD":
            u -= step*grad
        elif method == "Sign":
            u -= step*np.sign(grad)
        else:
            m = beta1*m+(1-beta1)*grad
            v = beta2*v+(1-beta2)*grad*grad
            u -= step*(m/(1-beta1**t))/(np.sqrt(v/(1-beta2**t))+eps)
    return u

def train(u0, s, z, method, steps, step, eps, beta1, beta2):
    u = u0.copy()
    m = np.zeros_like(u)
    v = np.zeros_like(u)
    for t in range(1, steps+1):
        grad = u@s-z
        if method == "GD":
            u -= step*grad
        elif method == "Sign":
            u -= step*np.sign(grad)
        else:
            m = beta1*m+(1-beta1)*grad
            v = beta2*v+(1-beta2)*grad*grad
            u -= step*(m/(1-beta1**t))/(np.sqrt(v/(1-beta2**t))+eps)
    return u

def loss(u, s, opt):
    e = u-opt
    return float(np.sum((e@s)*e)/u.size)

def measures(u, x, y):
    d, n = x.shape
    ux = u@x
    diag = np.sum(y*ux, axis=0)
    wm = (y.sum(axis=1)@ux-diag)/(n-1)
    total = float(np.sum((ux@ux.T)*(y@y.T)))
    var = (total-diag@diag)/(n*(n-1))-np.mean(wm*wm)
    margin = float(np.mean(diag-wm))
    z = d/math.sqrt(n)*(y@x.T)
    sx, sy = d/n*(x@x.T), d/n*(y@y.T)
    trace = np.trace(u@sx@u.T@sy)/d**2
    return dict(margin=margin, variance=float(var), msr=margin/math.sqrt(var),
                correct_variance=float(np.var(diag)),
                bridge_margin_error=margin-float(np.sum(u*z))/(d*math.sqrt(n)),
                total_trace_error=float(total/n**2-trace),
                wrongmean_square=float(np.mean(wm*wm)))

def exact_first_step(rho, h, eps):
    sig2 = 2*(1-rho)
    density = lambda t: math.sqrt(2/math.pi)*math.exp(-t*t/2)
    if eps == "GD":
        c, qh = 1., 1.
    elif eps == "Sign":
        c, qh = math.sqrt(2/math.pi), 1.
    else:
        aa = quad(lambda t: t*t/(t+eps)*density(t), 0, np.inf, epsabs=2e-13)[0]
        qh = quad(lambda t: (t/(t+eps))**2*density(t), 0, np.inf, epsabs=2e-13)[0]
        c = aa/math.sqrt(qh)
    p = 1-h
    if c*c < p:
        return None
    t = p/(c+math.sqrt(c*c-p))
    b, v = sig2*c*t, sig2*t*t
    q, a0, a1 = 1-b+v, 1-b/2, rho+b/2
    return dict(eps=eps, rho=rho, h=h, c=c, b=b, v=v, q=q,
                a0=a0, a1=a1, old_msr=a0/math.sqrt(q),
                new_msr=a1/math.sqrt(q), loss_ratio=(sig2-2*b+v)/sig2)

# Replay the stated fixed scan / 24-bisection rule on its exact scalar GD limit.
def quantized_gd(steps):
    fn = lambda x: (1-x)**(2*steps)-.75
    grid = np.linspace(0, .6/steps, 33)
    cache = {0.: fn(0.)}
    prev = 0.
    for val in grid[1:]:
        val = float(val)
        cache[val] = fn(val)
        if cache[prev] >= 0 and cache[val] <= 0:
            lo, hi = prev, val
            break
        prev = val
    initial = [lo, hi]
    for _ in range(24):
        mid = (lo+hi)/2
        cache[mid] = fn(mid)
        if cache[mid] > 0:
            lo = mid
        else:
            hi = mid
    best = min(cache, key=lambda a: abs(cache[a]))
    exact = 1-.75**(1/(2*steps))
    return dict(T=steps, exact_root=exact, selected=best,
                parameter_error=best-exact, limiting_loss_error=cache[best],
                initial_bracket=initial, final_bracket=[lo, hi])

result = dict(first_step=[exact_first_step(0., .75, e) for e in ("GD", 1., .1, "Sign")],
              negative_mean_counterexample=[exact_first_step(-.99, .0625, e) for e in ("GD", 10.)],
              fixed_accuracy=[quantized_gd(t) for t in (1,4)])
assert all(abs(x['loss_ratio']-.75) < 1e-12 for x in result['first_step'])
assert all(result['first_step'][i]['old_msr'] > result['first_step'][i+1]['old_msr'] for i in range(3))
neg = result['negative_mean_counterexample']
assert neg[1]['old_msr'] > neg[0]['old_msr']

# Actual paired tasks: separate r_x/r_y, negative correlation, and beta2=0.
configs = [("GD",4,.08,.1,.9,.99), ("Sign",4,.08,.1,.9,.99),
           ("Adam",4,.08,.1,.9,.99), ("Adam",4,.08,.1,.9,0.),
           ("Adam",4,.08,1.,0.,0.)]
cases = [(0.,0.), (1.,0.), (1.,.5), (math.sqrt(.5),math.sqrt(.5)), (1.,-.8)]
rows = []
for d in (16,32,64,128):
    n = d*d
    for rx, ry in cases:
        for seed in range(3):
            x0, y0, xi, yi = [RNG.normal(size=(d,n))/math.sqrt(d) for _ in range(4)]
            x1 = rx*x0+math.sqrt(1-rx*rx)*xi
            y1 = ry*y0+math.sqrt(1-ry*ry)*yi
            s = d/n*(x1@x1.T)
            z0, z1 = [d/math.sqrt(n)*(y@x.T) for x,y in ((x0,y0),(x1,y1))]
            opt = np.linalg.solve(s,z1.T).T
            sig = math.sqrt(2*(1-rx*ry))
            for method, steps, step, eps, b1, b2 in configs:
                actual_step = step if method == "GD" else step*sig
                actual_eps = eps*sig
                u = train(z0,s,z1,method,steps,actual_step,actual_eps,b1,b2)
                proxy = z0+scalar_update(z1-z0,method,steps,actual_step,actual_eps,b1,b2)
                cross = float(np.mean(z0*(u-z0)))
                update2 = float(np.mean((u-z0)**2))
                q = float(np.mean(u*u))
                old, new = measures(u,x0,y0), measures(u,x1,y1)
                expanded = (np.sum((u@s)*u)-2*np.sum(u*z1)+np.sum(opt*z1))/d**2
                assert abs(loss(u,s,opt)-expanded) < 1e-10
                assert abs(q-(np.mean(z0*z0)+2*cross+update2)) < 1e-12
                assert max(abs(old['total_trace_error']),abs(new['total_trace_error'])) < 1e-10
                if d == 16 and seed == 0:
                    scores = y0.T@u@x0
                    dd = np.diag(scores)
                    wm = (scores.sum(axis=0)-dd)/(n-1)
                    dense_v = np.mean((np.sum(scores*scores,axis=0)-dd*dd)/(n-1)-wm*wm)
                    assert abs(dense_v-old['variance']) < 1e-10
                rows.append(dict(d=d,rx=rx,ry=ry,seed=seed,method=method,beta1=b1,beta2=b2,
                    proxy_mse=float(np.mean((u-proxy)**2)),q=q,cross=cross,
                    empirical_loss=loss(u,s,opt),old=old,new=new))
result['real_tasks'] = rows

# A norm-controlled, data-dependent matrix invalidates any attempt to infer
# correct-score variance from the universal wrong-score bridge alone.
adversarial = []
for d in (16,32,64,128):
    n=d*d
    x,y=[RNG.normal(size=(d,n))/math.sqrt(d) for _ in range(2)]
    u=d*np.outer(y[:,0]/np.linalg.norm(y[:,0]), x[:,0]/np.linalg.norm(x[:,0]))
    adversarial.append(dict(d=d, q=float(np.mean(u*u)), **measures(u,x,y)))
result['adversarial_data_dependent_U'] = adversarial
result['resources'] = dict(seconds=time.monotonic()-START, executable=sys.executable,
    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    threadpools=threadpool_info(), pid=os.getpid())
assert all(x['num_threads'] <= 2 for x in result['resources']['threadpools'])
with (ROOT/'check_results.json').open('x') as f:
    json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items() if k not in ('real_tasks',)},indent=2))
print('real_task_records',len(rows))
for method in configs:
    name,_,_,_,b1,b2=method
    print(name,b1,b2,[(d,float(np.mean([r['proxy_mse'] for r in rows if r['d']==d and r['method']==name and r['beta1']==b1 and r['beta2']==b2]))) for d in (16,32,64,128)])
