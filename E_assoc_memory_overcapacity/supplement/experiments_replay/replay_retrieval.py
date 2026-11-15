"""Actual Gaussian training-data replay; all training candidates are ranked.

Seeds, streams, shapes and draw order match the published experiments.
Small replay uses the first two formal seeds. It is a software replay, not a
new powered confirmation. The full mode reproduces the formal designs.
"""
import argparse,csv,json,math,time
from pathlib import Path
import numpy as np
from population_model import METHODS,CORRELATIONS,population,predictions
HERE=Path(__file__).resolve().parent

def load_measurements(design):
    if design=='matched':
        rows=[json.loads(s) for s in (HERE/'matched_retrieval_measurements.jsonl').read_text().splitlines()]
        return {(r['d'],r['ir'],r['seed'],r['method'],task):r[task] for r in rows for task in ['old','new']}
    with (HERE/'finite_dimension_measurements.csv').open() as f:rows=list(csv.DictReader(f))
    return {(int(r['d']),int(r['ir']),int(r['seed']),r['method'],r['task']):float(r['recall']) for r in rows}

def rng(design,d,ir,seed,stream):
    prefix=[20261106,913752,22] if design=='matched' else [20261106,87410329,83]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(prefix+[d,ir,seed,stream])))

def dataset(design,d,ir,seed,queries):
    n=d*d//4;r=CORRELATIONS[ir]
    x=rng(design,d,ir,seed,0).standard_normal((d,n));x/=math.sqrt(d)
    y0=rng(design,d,ir,seed,1).standard_normal((d,n));y0/=math.sqrt(d)
    y1=rng(design,d,ir,seed,2).standard_normal((d,n));y1/=math.sqrt(d)
    y1*=math.sqrt(1-r*r);y1+=r*y0
    ids=rng(design,d,ir,seed,3).choice(n,size=queries,replace=False,shuffle=True)
    S=d/n*(x@x.T);z0=d/math.sqrt(n)*(y0@x.T);z1=d/math.sqrt(n)*(y1@x.T)
    return x,y0,y1,ids,z0,z1-z0@S

def direction(h,method,r):
    if method=='GD':return h
    if method=='Sign':return np.sign(h)
    e={'Adam2':2.,'Adam1':1.,'Adam.1':.1}[method]
    return h/(np.abs(h)+e*math.sqrt(2*(1-r)))

def ranks(u,x,y,ids,q):
    n=x.shape[1];rank=np.empty(len(ids),dtype=np.int64);D=np.empty(len(ids));V=np.empty(len(ids))
    for start in range(0,len(ids),256):
        ix=ids[start:start+256];uv=u@x[:,ix];truth=np.einsum('ij,ij->j',y[:,ix],uv)
        greater=np.zeros(len(ix),dtype=np.int64);squares=np.zeros(len(ix))
        for lo in range(0,n,2048):
            hi=min(n,lo+2048);scores=y[:,lo:hi].T@uv;mask=(ix>=lo)&(ix<hi)
            scores[ix[mask]-lo,np.flatnonzero(mask)]=0.
            squares+=np.einsum('ij,ij->j',scores,scores)
            gt=(scores>truth)|((scores==truth)&(np.arange(lo,hi)[:,None]<ix))
            gt[ix[mask]-lo,np.flatnonzero(mask)]=False
            greater+=np.count_nonzero(gt,axis=0)
        rank[start:start+len(ix)]=greater+1;D[start:start+len(ix)]=truth/math.sqrt(q);V[start:start+len(ix)]=squares/((n-1)*q)
    return rank,rank<=math.ceil(.1*n),D,V

def main():
    p=argparse.ArgumentParser();p.add_argument('--design',choices=['matched','confirmation'],default='matched')
    p.add_argument('--full',action='store_true');p.add_argument('--plan',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=time.monotonic()
    if a.plan:
        result=dict(matched=dict(dimensions=[128,256],seeds=192,datasets=1152,queries=2048,estimated_hours='1–3'),
                    confirmation=dict(dimensions=[128,256,512],seeds=32,datasets=288,queries=512,target_queries=8192,estimated_hours='2–5'),
                    small=dict(matched_datasets=12,confirmation_datasets=18,seeds=2),memory='3 GiB, swap 0',threads=1)
        with a.output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
        print(json.dumps(result,ensure_ascii=False));return
    a.output.mkdir(exist_ok=False);params=population();pred=predictions(params)
    frozen=json.loads((HERE/'population_parameters.json').read_text())
    err=max(abs(p[k]-next(t for t in frozen if int(t['ir'])==p['ir'] and t['method']==p['method'])[k]) for p in params for k in ['amplitude','q'])
    assert err<1e-10
    with (a.output/'population_predictions.json').open('x') as f:json.dump(dict(parameters=params,finite_dimension_predictions=pred),f,indent=2)
    lookup={(p['ir'],p['method']):p for p in params};observed=load_measurements(a.design)
    dims=[128,256] if a.design=='matched' else [128,256,512]
    seeds=(192 if a.design=='matched' else 32) if a.full else 2
    report=[];max_recall_error=0.
    # Each stream is indexed explicitly; job scheduling does not change samples.
    for seed in range(seeds):
        for d in dims:
            for ir in range(3):
                queries=2048 if a.design=='matched' else 8192 if (d,ir)==(512,2) else 512
                x,y0,y1,ids,z0,h=dataset(a.design,d,ir,seed,queries);arrays=dict(ids=ids)
                for method in METHODS:
                    par=lookup[ir,method];u=z0+par['amplitude']*direction(h,method,CORRELATIONS[ir])
                    for task,y in [('old',y0),('new',y1)]:
                        rank,ok,D,V=ranks(u,x,y,ids,par['q'])
                        for key,value in [('ranks',rank),('success',ok),('D',D),('V',V)]:arrays[method+'_'+task+'_'+key]=value
                        recall=float(ok.mean());expected=observed[d,ir,seed,method,task]
                        max_recall_error=max(max_recall_error,abs(recall-expected));assert recall==expected,(a.design,d,ir,seed,method,task,recall,expected)
                        report.append(dict(d=d,ir=ir,seed=seed,method=method,task=task,recall=recall,D=float(D.mean()),V=float(V.mean()),
                            var_D=float(D.var(ddof=1)),var_V=float(V.var(ddof=1)),cov_DV=float(np.cov(D,V,ddof=1)[0,1])))
                with (a.output/f'd{d}_r{ir}_s{seed}.npz').open('xb') as f:np.savez_compressed(f,**arrays)
                print(json.dumps(dict(design=a.design,d=d,ir=ir,seed=seed,seconds=time.monotonic()-start)),flush=True)
                del arrays,x,y0,y1,z0,h
    with (a.output/'measurements.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
    result=dict(status='PASS',design=a.design,full=a.full,datasets=len(dims)*3*seeds,
                seed_level_recalls_equal_to_original=True,max_recall_error=max_recall_error,
                regenerated_population_max_error=err,seconds=time.monotonic()-start,
                small_replay_is_not_statistical_confirmation=not a.full)
    with (a.output/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
