"""Future formal worker; imported safely by synthetic tests, main requires pipe credential."""
import resource
import signal
import sys
import time
import math
import json
import numpy as np
from common_v2 import OUT, METHODS, ARMS, config, verify, entropy, plan, jobname, dump, sha, commit, usage
from model import alpha_grid, direction, ranks_many, select, analytic_alpha, pava
from protocol_v2 import credential, Events

STOP=None
class StopRun(Exception): pass
def checkpoint():
    if STOP: raise StopRun(STOP)
def stop(sig, frame):
    global STOP
    STOP='cpu_soft' if sig==signal.SIGXCPU else 'wall_soft' if sig==signal.SIGALRM else 'signal_'+str(sig)

def dataset(phase,d,method,seed,Q):
    def rng(k): return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy(phase,d,method,seed,k))))
    n=d*d//4
    x=rng(0).standard_normal((d,n),dtype=np.float64);x/=math.sqrt(d);checkpoint()
    y0=rng(1).standard_normal((d,n),dtype=np.float64);y0/=math.sqrt(d);checkpoint()
    # r=0, so independent Y1, with no unnecessary mix.
    y1=rng(2).standard_normal((d,n),dtype=np.float64);y1/=math.sqrt(d);checkpoint()
    ids=rng(3).choice(n,Q,replace=False,shuffle=True)
    S=(d/n)*(x@x.T);checkpoint()
    z0=(d/math.sqrt(n))*(y0@x.T);checkpoint()
    z1=(d/math.sqrt(n))*(y1@x.T);checkpoint()
    H=z1-z0@S
    return x,y0,y1,ids,z0,H

def calibration(d,m,s,cfg,manifest):
    x,y0,y1,ids,z0,H=dataset(0,d,m,s,cfg['q_cal'])
    grid=alpha_grid(METHODS[m],cfg['grid_count'])
    rank=ranks_many(z0,direction(H,METHODS[m]),grid,x,y1,ids,checkpoint)
    return dict(ids=ids,ranks=rank,metadata=np.array(json.dumps(dict(job=['cal',d,m,s],manifest=manifest))))

def lock_alphas(data,cfg,manifest):
    result=dict(manifest=manifest,source_hashes={},cells=[])
    for d in cfg['dims']:
        for m,method in enumerate(METHODS):
            curves=[]
            for s in range(cfg['n_cal']):
                p=data/jobname(['cal',d,m,s]);result['source_hashes'][p.name]=sha(p)
                with np.load(p,allow_pickle=False) as f: curves.append((f['ranks']<=math.ceil(.1*(d*d//4))).mean(axis=1))
            curves=np.array(curves);raw=curves.mean(axis=0);grid=alpha_grid(method,cfg['grid_count'])
            for target in cfg['targets']:
                choice=select(grid,raw,target,cfg['nonmonotonic_limit'])
                result['cells'].append(dict(d=d,method=method,target=target,empirical=choice,
                    analytic=analytic_alpha(method,d,target,True),grid=grid.tolist(),raw=raw.tolist(),
                    fitted=pava(raw).tolist(),se=(curves.std(axis=0,ddof=1)/math.sqrt(cfg['n_cal'])).tolist()))
    return result

def evaluation(d,s,cfg,manifest,locked,lock_hash):
    x,y0,y1,ids,z0,H=dataset(1,d,-1,s,cfg['q_eval'])
    shape=(len(ARMS),len(cfg['targets']),len(METHODS))
    alphas=np.zeros(shape);valid=np.zeros(shape,dtype=bool)
    ranks=np.zeros(shape+(2,len(ids)),dtype=np.int64)
    for m,method in enumerate(METHODS):
        positions=[];aa=[]
        for a,arm in enumerate(ARMS):
            for t,target in enumerate(cfg['targets']):
                cell=next(c for c in locked['cells'] if (c['d'],c['method'],c['target'])==(d,method,target))
                alpha=cell['empirical']['alpha'] if arm=='empirical' else cell['analytic']
                if alpha is not None:
                    positions.append((a,t));aa.append(alpha);alphas[a,t,m]=alpha;valid[a,t,m]=True
        h=direction(H,method)
        for task,y in enumerate([y0,y1]):
            rr=ranks_many(z0,h,aa,x,y,ids,checkpoint)
            for i,(a,t) in enumerate(positions): ranks[a,t,m,task]=rr[i]
    return dict(ids=ids,alphas=alphas,valid=valid,ranks=ranks,
        metadata=np.array(json.dumps(dict(job=['eval',d,s],manifest=manifest,lock_sha256=lock_hash))))

def main():
    if sys.argv[1:]!=['--credential-worker']: raise ValueError('supervisor only')
    cfg=config();resource.setrlimit(resource.RLIMIT_CPU,(cfg['runner_cpu_soft'],cfg['runner_cpu_hard']))
    mh=verify();cred=credential('runner',mh)
    data=OUT/'formal';start=time.monotonic()
    dump(data/'start.json',dict(manifest=mh,credential=cred,streams=sha(OUT/'streams.json'),config=cfg,usage=usage()))
    for sig in [signal.SIGXCPU,signal.SIGALRM,signal.SIGTERM,signal.SIGINT]: signal.signal(sig,stop)
    signal.alarm(cfg['runner_wall_soft'])
    events=Events(data/'events.jsonl',mh,cred);completed=[];active=None;error=None;reason='planned_complete'
    locked=None
    try:
        for job in plan(cfg):
            checkpoint();active=job;events.append('started',job=job)
            p=data/jobname(job)
            if job[0]=='cal': commit(p,calibration(*job[1:],cfg,mh))
            elif job==['lock']:
                locked=lock_alphas(data,cfg,mh);dump(p,locked)
            else: commit(p,evaluation(*job[1:],cfg,mh,locked,sha(data/'lock.json')))
            completed.append(job);events.append('complete',job=job,sha256=sha(p));active=None
    except StopRun as e: reason=str(e)
    except BaseException as e: reason='exception';error=repr(e)
    events.append('stopped',reason=reason,active=active,error=error);events.close();signal.alarm(0)
    dump(data/'runner_status.json',dict(reason=reason,active=active,error=error,completed=completed,
        event_tail=events.prev,usage=usage(),wall=time.monotonic()-start))
    return 0 if error is None else 1

if __name__=='__main__': sys.exit(main())
