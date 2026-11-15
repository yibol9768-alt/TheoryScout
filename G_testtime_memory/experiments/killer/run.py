"""CPU-only, seed-level experiment; no imports from sibling projects."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
from design import (DIMS, LOADS, ALPHAS, BETAS, CAL_SEEDS, TEST_SEEDS,
                    SMOKE_SEED, METHODS, spectrum, query_ids, prediction)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/killer'

def check_freeze():
    for line in (ROOT/'PREREG_KILLER.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name

def resources():
    from threadpoolctl import threadpool_info
    cg = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
    meta = {'cgroup':str(cg), 'memory_max':(cg/'memory.max').read_text().strip(),
            'swap_max':(cg/'memory.swap.max').read_text().strip(),
            'threadpools':threadpool_info(), 'numpy':np.__version__,
            'python':os.sys.version, 'pid':os.getpid()}
    assert meta['memory_max'] == str(2*1024**3), meta
    assert meta['swap_max'] == '0', meta
    assert all(x['num_threads'] <= 2 for x in meta['threadpools'])
    return meta

def states(K, V):
    d = K.shape[1]
    ss = np.zeros((len(BETAS),d,d))
    bb = np.array(BETAS)[:,None,None]
    for k,v in zip(K,V):
        residual = v[None,:] - np.einsum('bij,j->bi',ss,k,optimize=False)
        ss += bb * residual[:,:,None] * k[None,None,:]
    return np.concatenate(((V.T@K)[None,:,:],ss))

def metrics(S,K,V,ids):
    score = (K[ids]@S.T)@V.T
    good = score[np.arange(len(ids)), ids]
    rank = (score > good[:,None]).sum(axis=1)
    assert np.all((0 <= rank) & (rank < len(K)))
    ok = rank == 0
    return {'top1':float(ok.mean()), 'top10':float((rank < int(np.ceil(.1*len(K)))).mean()),
            'mean_rank':float((rank/(len(K)-1)).mean()),
            'old':float(ok[ids<len(K)//4].mean()),
            'new':float(ok[ids>=3*len(K)//4].mean())}

def data(d,k,a,seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed,d,k,int(2*a)]))
    K = rng.standard_normal((d*k,d))*np.sqrt(spectrum(d,a)/d)
    V = rng.standard_normal((d*k,d))/np.sqrt(d)
    return K,V

def smoke():
    for a in (0.,1.5):
        K,V=data(16,2,a,SMOKE_SEED)
        ss=states(K,V)
        for b,S in zip(BETAS,ss[1:]):
            C=np.zeros_like(K)
            for t,k in enumerate(K):
                C -= b*(C@k)[:,None]*k[None,:]
                C[t] += b*k
            np.testing.assert_allclose(S,V.T@C,rtol=1e-10,atol=1e-11)
        small=1e-8
        S=np.zeros((16,16))
        for k,v in zip(K,V):
            S+=small*np.outer(v-S@k,k)
        assert np.linalg.norm(S/small-ss[0])/np.linalg.norm(ss[0]) < 1e-5
        for name,S in zip(METHODS,ss):
            assert np.isfinite(S).all()
            m=metrics(S,K,V,query_ids(len(K)))
            assert 0<=m['top1']<=1
            assert 0<=prediction(16,2,a,name)<=1
    x=np.array([0.,1.])
    assert np.isclose(x.std(ddof=1)/np.sqrt(2),.5)
    print('SMOKE PASS: coefficient identity, small-beta limit, ranks, predictor, seed SE',flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['smoke','formal'],required=True)
    args=parser.parse_args()
    check_freeze()
    start=time.time()
    meta=resources()
    meta.update({'mode':args.mode,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                 'frozen_manifest':(ROOT/'PREREG_KILLER.sha256').read_text()})
    meta_path=OUT/f'{args.mode}_meta.json'
    assert not meta_path.exists(), f'Refuse overwrite: {meta_path}'
    meta_path.write_text(json.dumps(meta,indent=2))
    if args.mode=='smoke':
        smoke()
        complete=True
    else:
        assert (OUT/'smoke_meta.json').exists()
        assert json.loads((OUT/'smoke_meta.json').read_text()).get('complete') is True
        records=OUT/'raw.jsonl'
        assert not records.exists(), 'Refuse reuse of formal observations'
        complete=True
        elapsed_prepare=json.loads((OUT/'prepare_timing.json').read_text())['elapsed_seconds']
        elapsed_smoke=json.loads((OUT/'smoke_meta.json').read_text())['elapsed_seconds']
        with records.open('x',buffering=1) as f:
            tasks=[('calibration',64,CAL_SEEDS)]+[('validation',d,TEST_SEEDS) for d in DIMS]
            for split,d,seeds in tasks:
                for a in ALPHAS:
                    for load in LOADS:
                        for seed in seeds:
                            if time.time()-start+elapsed_prepare+elapsed_smoke>3300:
                                complete=False
                                break
                            K,V=data(d,load,a,seed)
                            ss=states(K,V)
                            assert np.isfinite(ss).all(), (d,load,a,seed)
                            ids=query_ids(len(K))
                            lam=spectrum(d,a)
                            diag={'effective_dim':float(d*d/(lam@lam)),
                                  'key_norm_mean':float((K*K).sum(1).mean()),
                                  'key_norm_var':float((K*K).sum(1).var()),
                                  'max_key_norm':float((K*K).sum(1).max())}
                            for name,S in zip(METHODS,ss):
                                row={'split':split,'d':d,'load':load,'alpha':a,'seed':seed,
                                     'method':name,'queries':len(ids),**diag,**metrics(S,K,V,ids)}
                                f.write(json.dumps(row)+'\n')
                        print(f'{split} d={d} a={a} load={load} elapsed={time.time()-start:.1f}s',flush=True)
                        if not complete: break
                    if not complete: break
                if not complete: break
    meta.update({'complete':complete,'elapsed_seconds':time.time()-start,
                 'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'process_cpu_seconds':time.process_time()})
    meta_path.write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:meta[k] for k in ('complete','elapsed_seconds','max_rss_kib')}),flush=True)
    assert complete, 'Partial grid: killer decision inconclusive'

if __name__=='__main__': main()
