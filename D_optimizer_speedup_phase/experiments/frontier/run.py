"""CPU fresh-sample frontier experiment; only stdlib, numpy and scipy imports."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/frontier'
MASTER = 2026110617

def dump(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def coefficients(lam, sig, algo):
    if algo == 'sign':
        return np.ones(len(lam)), 2*np.sqrt(lam)/(np.pi*sig)
    def f(s):
        lt = lam/(1+2*s*s*lam)
        z = np.exp(-.5*np.log1p(2*s*s*lam).sum())
        return np.r_[2*s*z*lt, 2*np.sqrt(2)/np.pi/sig*z*lt]
    v, err = quad_vec(f, 0, np.inf, epsabs=1e-11, epsrel=1e-11)
    return v[:len(lam)], v[len(lam):]

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('mode', choices=['smoke', 'formal'])
    pa.add_argument('--max-seconds', type=float, required=True)
    args = pa.parse_args()
    start = time.monotonic()
    out = BASE/args.mode
    out.mkdir(exist_ok=False)
    (out/'arrays').mkdir()
    subprocess.run(['sha256sum', '-c', str(BASE/'FROZEN.sha256')], cwd=ROOT,
                   stdout=(out/'freeze_check.txt').open('w'), check=True)
    scope = os.environ['FRONTIER_SCOPE']
    props = subprocess.check_output(['systemctl', '--user', 'show', scope, '-p', 'MemoryMax',
                                    '-p', 'MemorySwapMax', '-p', 'ControlGroup'], text=True)
    (out/'scope_properties.txt').write_text(props)
    assert 'MemoryMax=2147483648' in props and 'MemorySwapMax=0' in props
    threads = {k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}
    assert all(v == '1' for v in threads.values())
    manifest = dict(mode=args.mode, status='running', start_utc=datetime.now(timezone.utc).isoformat(),
                    executable=sys.executable, threads=threads, device='CPU', master_seed=MASTER,
                    scope=scope, properties=props, source_sha256=sha(Path(__file__)),
                    prereg_sha256=sha(ROOT/'PREREG_FRONTIER.md'), project_imports=[], network_calls=0)
    dump(out/'manifest.json', manifest)
    records = []
    nd = samples = 0
    alphas = [1.5, 2., 3.] if args.mode == 'formal' else [1.5, 3.]
    sizes = [4, 6, 9, 13, 18, 26] if args.mode == 'formal' else [4, 6]
    reps = 96 if args.mode == 'formal' else 16
    factors = [.3, .6] if args.mode == 'formal' else [.6]
    p0 = dict(one_step=0., sum_k=0., nonfinite=0)
    def deadline():
        if time.monotonic()-start > args.max_seconds:
            raise TimeoutError('Frozen computation deadline')
    try:
        for ai, alpha in enumerate(alphas):
            for ni, n in enumerate(sizes):
                lam = np.arange(1, n+1, dtype=float)**(-alpha)
                theta = np.arange(1, n+1, dtype=float)**((alpha-2)/2)
                tail = float(polygamma(1, n+1))
                sig = math.sqrt(1+tail)
                for algo in ['sign', 'polar', 'sgd']:
                    k, slope = coefficients(lam, sig, algo) if algo != 'sgd' else (lam, lam)
                    if algo == 'polar':
                        p0['sum_k'] = max(p0['sum_k'], abs(float(k.sum())-1))
                    p = alpha/2 if algo == 'sign' else alpha
                    hsum = float(np.sqrt(lam).sum()) if algo == 'sign' else 1.
                    for fi, factor in enumerate(factors):
                        batches = [1]
                        if args.mode == 'formal' and alpha == 3 and n in [6,13,26] and fi == 0 and algo in ['sign','sgd']:
                            batches += [4]
                        for b in batches:
                            eta1 = factor/n/hsum
                            eta = eta1*(math.sqrt(b) if algo == 'sign' else b)
                            # b=1 polar, otherwise equal-paid-compute batch intervention.
                            unit = n**p/eta1/b
                            times = sorted(set([int(math.ceil(unit)), int(math.ceil(2*unit))]))
                            if args.mode == 'smoke':
                                times = [min(t, 256) for t in times]
                                times = sorted(set(times))
                            seed = [MASTER, 1 if args.mode == 'formal' else 0, ai, ni, ['sign','polar','sgd'].index(algo), fi, b]
                            gen = np.random.default_rng(np.random.SeedSequence(seed))
                            e = np.broadcast_to(-theta, (reps,n)).copy()
                            # Reuse an independent block only as a sampling optimization; each row is consumed once.
                            block = 256 if b == 1 else 64
                            checkpoint = 0
                            for begin in range(0, times[-1], block):
                                deadline()
                                length = min(block, times[-1]-begin)
                                xblock = gen.standard_normal((length,reps,b,n))*np.sqrt(lam)
                                epsblock = gen.standard_normal((length,reps,b))*sig
                                for j in range(length):
                                    x = xblock[j]
                                    residual = np.einsum('rbn,rn->rb',x,e)-epsblock[j]
                                    if b == 1 and algo in ['sign','polar']:
                                        direction = np.sign(residual[:,0])[:,None]
                                        if algo == 'sign':
                                            u = np.sign(x[:,0,:])*direction
                                        else:
                                            xx = x[:,0,:]
                                            u = xx/np.linalg.norm(xx,axis=1)[:,None]*direction
                                    else:
                                        g = np.einsum('rbn,rb->rn',x,residual)/b
                                        u = np.sign(g) if algo == 'sign' else g
                                    if begin == 0 and j == 0:
                                        before = (e*e)@lam
                                        after = ((e-eta*u)**2)@lam
                                        ident = before-2*eta*((e*u)@lam)+eta*eta*((u*u)@lam)
                                        p0['one_step'] = max(p0['one_step'],float(np.max(np.abs(after-ident))))
                                    e -= eta*u
                                    t = begin+j+1
                                    if t == times[checkpoint]:
                                        v = (e*e)@lam
                                        total = tail+v
                                        name = f'a{ai}_n{n}_{algo}_f{fi}_b{b}_t{t}.npz'
                                        np.savez_compressed(out/'arrays'/name,e=e,lam=lam,tail=tail,risk=total)
                                        s = sig*sig+v
                                        weighted = (e*e*s[:,None]).mean(axis=0)/(e*e).mean(axis=0)
                                        rec = dict(alpha=alpha,beta=1.,n=n,algo=algo,b=b,factor=factor,eta=eta,T=t,
                                                   tau=checkpoint+1,C=n*b*t,reps=reps,seed=seed,file=name,
                                                   risk=float(total.mean()),se=float(total.std(ddof=1)/math.sqrt(reps)),
                                                   loc_max=float(weighted.max()),max_sample_risk=float(total.max()))
                                        if b == 1 and algo != 'sgd':
                                            def ref(c):
                                                rate=2*c*eta*slope
                                                q=np.exp(t*np.log1p(-rate))
                                                return tail+float(np.sum(lam*(theta**2*q+eta*eta*k*(1-q)/rate)))
                                            rec['lower']=ref(1.)
                                            rec['upper']=ref(sig/4.)
                                        records.append(rec)
                                        with (out/'events.jsonl').open('a') as f:
                                            f.write(json.dumps(rec,allow_nan=False)+'\n')
                                        checkpoint += 1
                                        if checkpoint == len(times):
                                            break
                            samples += reps*b*times[-1]
                            nd += n*reps*b*times[-1]
                            if not np.isfinite(e).all():
                                p0['nonfinite'] += 1
                                raise ArithmeticError('nonfinite trajectory')
                            print(f'{alpha=} {n=} {algo=} {factor=} {b=} T={times[-1]} elapsed={time.monotonic()-start:.1f}',flush=True)
        dump(out/'data.json',dict(records=records,p0=p0))
        manifest['status']='complete'
    finally:
        manifest.update(end_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
                        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                        training_samples=samples,training_nD=nd,records=len(records),p0=p0)
        dump(out/'manifest.json',manifest)

if __name__ == '__main__':
    main()
