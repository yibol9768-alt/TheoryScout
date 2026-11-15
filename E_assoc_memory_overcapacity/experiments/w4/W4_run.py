import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys, json, time, signal, hashlib
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'repro'))
from W4_core import *
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'experiments/w4'

def freeze():
    for line in (ROOT/'W4_FREEZE.sha256').read_text().splitlines():
        sha,path=line.split('  ',1)
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path

def main():
    freeze(); start=time.monotonic(); before=resource_info()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('3300 second formal limit')))
    signal.alarm(3300)
    completed=0; stop=None
    with (OUT/'W4_raw.jsonl').open('x',buffering=1) as out, (OUT/'W4_events.jsonl').open('x',buffering=1) as events:
        def emit(row): out.write(json.dumps(row,allow_nan=False)+'\n')
        def event(kind,d,seed):
            nonlocal completed
            completed+=1; row=dict(kind=kind,d=d,seed=seed,seconds=time.monotonic()-start)
            events.write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
        try:
            for d in [64,128,256]:
                for seed in range(16):
                    x,y,s,z=mdata(d,seed); n=d*d
                    ids=rng(1,d,seed,4).choice(n,1024,replace=False)
                    for name,u,q,c,hp in cases(z,s):
                        diag=diag_scores(u,x,y,q)
                        r=retrieve(u,x,y,ids,q,c,diag)
                        emit(dict(kind='M',d=d,seed=seed,method=name,q=q,c=c,
                             full_variance=float(diag.var(ddof=1)),full_mean=float(diag.mean()),
                             reduction_mse=float(np.mean((u-hp)**2)/q),**r))
                        if d==64:
                            np.savez_compressed(OUT/f'W4_full_d{d}_s{seed}_{name}.npz',scores=diag)
                    event('M',d,seed)
                    del x,y,s,z,u,hp,diag
            for d in [64,128,256,320]:
                for seed in range(16):
                    for r in ft(d,seed): emit(r)
                    event('FT',d,seed)
            for seed in range(5):
                d=64; n=d*d; rr=np.random.default_rng(np.random.SeedSequence([20261106,d,n,seed,1]))
                x=rr.normal(size=(d,n))/math.sqrt(d); y=rr.normal(size=(d,n))/math.sqrt(d)
                s=d/n*(x@x.T); z=d/math.sqrt(n)*(y@x.T); ids=rr.choice(n,128,replace=False)
                sr=rng(1,d,seed,5)
                subsets=np.array([sr.choice(n,128,replace=False) for j in range(32)])
                for name,u,q,c,hp in cases(z,s):
                    diag=diag_scores(u,x,y,q)
                    vv=np.var(diag[subsets],axis=1,ddof=1)
                    np.savez_compressed(OUT/f'W4_legacy_s{seed}_{name}.npz',scores=diag,ids=ids,subsets=subsets)
                    emit(dict(kind='legacy',d=d,seed=seed,method=name,
                        original_variance=float(diag[ids].var(ddof=1)),full_variance=float(diag.var(ddof=1)),
                        resample_variances=vv.tolist(),ids=ids.tolist()))
                event('legacy',d,seed)
        except TimeoutError as ex: stop=str(ex)
    signal.alarm(0); freeze()
    meta=dict(seconds=time.monotonic()-start,completed=completed,expected=117,stop=stop,
              before=before,after=resource_info())
    with (OUT/'W4_metadata.json').open('x') as f: json.dump(meta,f,indent=2)
    print(json.dumps(meta),flush=True)

if __name__=='__main__': main()
