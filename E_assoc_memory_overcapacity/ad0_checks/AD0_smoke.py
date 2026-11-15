"""AD0: full-array training smoke; exclusive outputs, no existing writes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'ad0_checks'

def emit(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def manifest():
    paths = list(ROOT.glob('*.md'))
    paths += [p for p in (ROOT/'paper').rglob('*')
              if p.suffix in ('.tex','.bib','.md','.py')
              and 'build' not in p.parts and '.cache' not in p.parts]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file() and not p.name.startswith('AD0_')}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['baseline','pilot','main','audit'])
    mode = parser.parse_args().mode
    resource.setrlimit(resource.RLIMIT_CPU,(900,920))
    wall, cpu = time.monotonic(), time.process_time()
    if mode == 'baseline':
        m = manifest()
        emit('AD0_baseline.json', m)
        print('baseline',len(m),flush=True)
        return
    if mode == 'audit':
        old = json.loads((OUT/'AD0_baseline.json').read_text())
        new = manifest()
        result = {'protected_count':len(old),
                  'modified':[p for p in old if p in new and old[p]!=new[p]],
                  'deleted':sorted(old.keys()-new.keys()),
                  'other_new':sorted(new.keys()-old.keys())}
        emit('AD0_integrity.json',result)
        print(json.dumps(result),flush=True)
        return
    import numpy as np
    from scipy.special import ndtri
    seed = 20261107001 if mode=='pilot' else 20261107002
    rng = np.random.default_rng(seed)
    steps = [1,2,3,4,6,8,12,16,24,32,64,128]
    etas = [.5,1.0] if mode=='pilot' else [.5,1.0,1.5]
    dims = [64,128] if mode=='pilot' else [64,128,256]
    reps = 8 if mode=='pilot' else 32
    queries = 64
    rows=[]
    # The n defined by d/sqrt(n)=b_(n-1) keeps top-1 in a nondegenerate window.
    def critical_n(d):
        candidates=np.arange(max(2*d+1,10),d*d+1)
        return int(candidates[np.argmin(np.abs(d/np.sqrt(candidates)-ndtri(1-1/(candidates-1))))])
    cells=[(d,'critical',critical_n(d)) for d in dims]
    if mode=='main':
        cells += [(d,'quadratic',int(d*d/4)) for d in dims]
    design={'seed':seed,'mode':mode,'steps':steps,'etas':etas,
            'cells':cells,'reps':reps,'queries':queries,
            'sign_alpha':.6,'sign_steps':[1,2,8,9,32,33,64,65],
            'method':'explicit full X,Y; paired datasets; direct all-candidate ranks',
            'evidence':'float64 descriptive smoke, not confidence certificate'}
    emit('AD0_'+mode+'_design.json',design)
    with (OUT/('AD0_'+mode+'_rows.jsonl')).open('x',encoding='utf-8') as stream:
        for d,regime,n in cells:
            for rep in range(reps):
                X=rng.standard_normal((d,n))/np.sqrt(d)
                Y=rng.standard_normal((d,n))/np.sqrt(d)
                S=(d/n)*(X@X.T)
                Z=(d/np.sqrt(n))*(Y@X.T)
                eig,V=np.linalg.eigh(S)
                ZV=Z@V
                ids=rng.choice(n,size=min(n,queries),replace=False)
                VX=V.T@X[:,ids]
                normz=np.linalg.norm(Z)
                def record(U,alg,T,eta=None,unstable=False):
                    pred=U@X[:,ids]
                    scores=Y.T@pred
                    correct=scores[ids,np.arange(len(ids))]
                    ranks=1+np.sum(scores>correct[None,:],axis=0)
                    normu=np.linalg.norm(U)
                    mu=float(np.sum(U*Z)/(np.sqrt(n)*normu))
                    center=mu-float(ndtri(1-1/(n-1)))
                    row={'d':d,'n':n,'regime':regime,'rep':rep,'alg':alg,
                         'T':T,'eta':eta,'unstable_step':bool(unstable),
                         'top1':float(np.mean(ranks<=1)),
                         'top5':float(np.mean(ranks<=5)),
                         'top10pct':float(np.mean(ranks<=int(np.ceil(.1*n)))),
                         'mean_rank_fraction':float(np.mean(ranks/n)),
                         'mu':mu,'center':center,
                         'cosine_Z':float(np.sum(U*Z)/(normu*normz))}
                    rows.append(row)
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                for eta in etas:
                    stable=bool(eta*eig[-1]<2)
                    # Matrix-polynomial endpoint, not S=I or an iid proxy.
                    for T in steps:
                        p=(1-(1-eta*eig)**T)/eig
                        U=(ZV*p)@V.T
                        record(U,'GD',T,eta,not stable)
                    # Check recurrence and spectrum on an actual dataset.
                    if rep==0:
                        U0=np.zeros_like(Z)
                        for t in range(12):
                            U0-=eta*(U0@S-Z)
                        p=(1-(1-eta*eig)**12)/eig
                        rel=float(np.linalg.norm(U0-(ZV*p)@V.T)/np.linalg.norm(U0))
                        if rel>1e-10:
                            raise RuntimeError(('spectral recurrence mismatch',rel))
                Uinf=(ZV/eig)@V.T
                record(Uinf,'GD','infinity',None)
                if mode=='main':
                    # True Sign matrix trajectory plus iid scalar coupling on the same data.
                    G=(ZV/np.sqrt(eig))@V.T
                    U=np.zeros_like(Z); H=np.zeros_like(G)
                    wanted=set(design['sign_steps'])
                    for T in range(1,66):
                        U+=.6*np.sign(Z-U@S)
                        H+=.6*np.sign(G-H)
                        if T in wanted:
                            record(U,'Sign',T,None)
                            rows[-1]['coupling_mse']=float(np.sum((U-H)**2)/np.sum(H**2))
                            # Auxiliary values are a separate record so JSONL never gets rewritten.
                            stream.write(json.dumps({'aux':'Sign_coupling','d':d,'n':n,
                                'regime':regime,'rep':rep,'T':T,
                                'relative_mse':rows[-1]['coupling_mse']})+'\n')
                stream.flush()
            print(mode,d,regime,n,'completed',flush=True)
    import collections
    grouped=collections.defaultdict(list)
    for row in rows:
        grouped[(row['d'],row['n'],row['regime'],row['alg'],row['eta'],str(row['T']))].append(row)
    summaries=[]
    for key,rs in grouped.items():
        item=dict(zip(('d','n','regime','alg','eta','T'),key))
        for metric in ('top1','top5','top10pct','mean_rank_fraction','mu','cosine_Z'):
            values=np.array([r[metric] for r in rs])
            item[metric]=float(values.mean())
            item[metric+'_se']=float(values.std(ddof=1)/np.sqrt(len(values)))
        item['unstable_reps']=sum(r['unstable_step'] for r in rs)
        summaries.append(item)
    paired=[]
    for d,regime,n in cells:
        base={r['rep']:r for r in rows if r['d']==d and r['regime']==regime and r['alg']=='GD' and r['T']=='infinity'}
        for eta in etas:
            for T in steps:
                rs=[r for r in rows if r['d']==d and r['regime']==regime and r['alg']=='GD' and r['eta']==eta and r['T']==T]
                ds=np.array([r['top1']-base[r['rep']]['top1'] for r in rs])
                paired.append({'d':d,'regime':regime,'eta':eta,'T':T,
                               'top1_minus_infinity':float(ds.mean()),
                               'paired_se':float(ds.std(ddof=1)/np.sqrt(len(ds))),
                               'unstable_reps':sum(r['unstable_step'] for r in rs)})
    cgroup=Path('/proc/self/cgroup').read_text().strip()
    cgpath=cgroup.split(':')[-1]
    cgd=Path('/sys/fs/cgroup')/cgpath.lstrip('/')
    cg={name:(cgd/name).read_text().strip() for name in ('memory.max','memory.swap.max','memory.peak') if (cgd/name).exists()}
    result={'design':design,'summary':summaries,'paired':paired,
            'resources':{'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-wall,
                         'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         'cgroup':cgroup,'limits':cg,
                         'Threads':[x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Threads:')],
                         'thread_env':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}}
    emit('AD0_'+mode+'_results.json',result)
    print(json.dumps(result['resources']),flush=True)

if __name__=='__main__':
    main()
