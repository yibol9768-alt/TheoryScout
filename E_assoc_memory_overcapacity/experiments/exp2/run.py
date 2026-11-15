"""Frozen EXP2: exact full-candidate ranks with deterministic streamed data."""
import argparse,hashlib,json,os,resource,signal,sys,time
import numpy as np
from design import *

def frozen():
    digest=hashlib.sha256((ROOT/'PREREG_EXP2.md').read_bytes()).hexdigest()
    assert digest==(ROOT/'PREREG_EXP2.sha256').read_text().split()[0]
    for path,h in json.loads((OUT/'frozen_inputs.json').read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    return digest

class Budget:
    def __init__(self,deadline):self.start=time.monotonic();self.deadline=deadline;self.max_op=0.
    def check(self):
        if time.time()+max(3.,2*self.max_op)>self.deadline:raise TimeoutError('frozen total wall-clock reserve')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>=1700:raise TimeoutError('RSS reserve 1700 MiB')
    def dot(self,a,b):
        self.check();t=time.monotonic();c=a@b;self.max_op=max(self.max_op,time.monotonic()-t);return c

def rngs(d,k,seed,tag):
    entropy=[20261106,2002,d,round(k*d*d),seed,tag]
    return entropy,[np.random.default_rng(np.random.SeedSequence(entropy+[j])) for j in range(3)]

def stream(d,k,seed,tag,budget,m=M,dense=False):
    n=round(k*d*d);entropy,(rx,ry,rq)=rngs(d,k,seed,tag)
    ids=rq.choice(n,size=min(m,n),replace=False);xq=np.empty((d,len(ids)));yq=np.empty_like(xq)
    s=np.zeros((d,d));z=np.zeros_like(s);xx=[];yy=[]
    for start in range(0,n,8192):
        width=min(8192,n-start)
        x=rx.standard_normal((d,width))/math.sqrt(d);y=ry.standard_normal((d,width))/math.sqrt(d)
        s+=budget.dot(x,x.T)*(d/n);z+=budget.dot(y,x.T)*(d/math.sqrt(n))
        select=(ids>=start)&(ids<start+width);xq[:,select]=x[:,ids[select]-start];yq[:,select]=y[:,ids[select]-start]
        if dense:xx.append(x);yy.append(y)
    return s,z,ids,xq,yq,entropy,(np.concatenate(xx,axis=1),np.concatenate(yy,axis=1)) if dense else None

def cases(z,s,high,mom,budget,p2alpha=None):
    yield ('Sign1',.6*np.sign(z),.36,math.sqrt(2/math.pi),None,{})
    for a in ((p2alpha,) if p2alpha is not None else ((.6,1.5) if high else (.2,.6,1.2,1.5))):
        h1=a*np.sign(z);proxy=h1+a*np.sign(z-h1);actual=h1+a*np.sign(z-budget.dot(h1,s))
        q=sign_q(a);c=sign_c(a)
        if p2alpha is None:yield (f'Sign2_a{a:g}_proxy',proxy,q,c,None,dict(alpha=a,T=2))
        yield (f'Sign2_a{a:g}_actual',actual,q,c,proxy,dict(alpha=a,T=2))
    if p2alpha is not None:return
    for p in mom:
        if high and not (p['b1']==.9 and p['e']==.1 and p['T']==16):continue
        budget.check();proxy=adam(z,None,p);actual=adam(z,s,p)
        base=f"Adam_b{p['b1']:g}_e{p['e']:g}_T{p['T']}"
        params={key:p[key] for key in ('alpha','b1','b2','e','T')}
        yield (base+'_proxy',proxy,p['q'],p['c'],None,params)
        yield (base+'_actual',actual,p['q'],p['c'],proxy,params)

def retrieve(vectors,truth,d,k,seed,tag,ids,budget):
    n=round(k*d*d);nmethods=truth.shape[0];m=len(ids)
    _,(_,ry,_)=rngs(d,k,seed,tag)
    flat=truth.ravel();greater=np.zeros(len(flat),np.int64)
    sums=np.zeros(len(flat));squares=np.zeros(len(flat));allids=np.tile(ids,nmethods)
    # vectors is d x (nmethods*m), normalised independently by q_T.
    for start in range(0,n,8192):
        width=min(8192,n-start);y=ry.standard_normal((d,width))/math.sqrt(d)
        for lo in range(0,width,1024):
            hi=min(width,lo+1024);scores=budget.dot(vectors.T,y[:,lo:hi]);begin=start+lo;end=start+hi
            mask=(allids>=begin)&(allids<end);rr=np.flatnonzero(mask);cc=allids[mask]-begin
            scores[rr,cc]=-np.inf;greater+=np.sum(scores>flat[:,None],axis=1)
            scores[rr,cc]=0.;sums+=scores.sum(axis=1);scores*=scores;squares+=scores.sum(axis=1)
            del scores
    return (greater.reshape(nmethods,m)+1,sums.reshape(nmethods,m)/(n-1),squares.reshape(nmethods,m)/(n-1))

def configuration(d,k,seed,tag,high,mom,budget,m=M,check_dense=False,p2only=False):
    t=time.monotonic();s,z,ids,xq,yq,entropy,dense=stream(d,k,seed,tag,budget,m,check_dense)
    items=[];vv=[];tt=[]
    for name,u,q,c,proxy,params in cases(z,s,high,mom,budget,(.6 if d==128 else 1.5) if p2only else None):
        v=budget.dot(u,xq)/math.sqrt(q);truth=np.sum(v*yq,axis=0)
        item=dict(d=d,n=round(k*d*d),kappa=k,seed=seed,seed_entropy=entropy,name=name,q=q,c=c,
                  query_indices=ids.tolist(),true_scores=truth.tolist(),parameters=params,
                  raw_lr=params.get('alpha',.6)/math.sqrt(round(k*d*d)),
                  raw_epsilon=params.get('e',0)/math.sqrt(round(k*d*d)),
                  predictions={metric:pred(c,k,metric) for metric in METRICS})
        if proxy is not None:
            error=u-proxy;diag=np.sum(yq*budget.dot(error,xq),axis=0)/math.sqrt(q)
            item.update(relative_weight_mse=float(np.mean(error*error)/q),weight_mse=float(np.mean(error*error)),
                        normalized_diagonal_errors=diag.tolist(),normalized_diagonal_mse=float(np.mean(diag*diag)),
                        diagonal_mse=float(np.mean(diag*diag)*q))
        vv.append(v);tt.append(truth);items.append(item)
    vectors=np.concatenate(vv,axis=1);truths=np.array(tt)
    ranks,wrongmeans,wrongseconds=retrieve(vectors,truths,d,k,seed,tag,ids,budget)
    for j,r in enumerate(items):
        r['ranks']=ranks[j].tolist();r['wrong_mean']=float(wrongmeans[j].mean());r['wrong_second']=float(wrongseconds[j].mean())
        for size in (128,len(ids)):
            size=min(size,len(ids));rr=ranks[j,:size];prefix='full' if size==len(ids) else 'prefix128'
            metrics={'mean_rank':float(np.mean((rr-1)/(r['n']-1)))}
            metrics.update({f'top{q:g}':float(np.mean(rr<=math.ceil(q*r['n']))) for q in (.01,.1,.5)})
            r[prefix]=metrics
        if check_dense:
            xx,yy=dense;raw=vectors[:,j*len(ids):(j+1)*len(ids)].T@yy
            for row,idx in enumerate(ids):raw[row,idx]=-np.inf
            brute=1+np.sum(raw>truths[j,:,None],axis=1)
            assert np.array_equal(ranks[j],brute),(r['name'],'dense-rank mismatch')
    return items,dict(d=d,kappa=k,seed=seed,methods=len(items),seconds=time.monotonic()-t,
        elapsed=time.monotonic()-budget.start,rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)

def validate(budget,mom):
    items,_=configuration(8,1.,999,90,False,mom,budget,m=17,check_dense=True)
    _,(rx,ry,_)=rngs(8,1.,999,90);x=rx.standard_normal((8,64))/math.sqrt(8);y=ry.standard_normal((8,64))/math.sqrt(8)
    s=8/64*(x@x.T);z=8/8*(y@x.T);a=.6;u=a*np.sign(z)
    assert np.array_equal(u+a*np.sign(z-u),2*a*np.sign(z)*(np.abs(z)>a))
    dense=u@x;score=dense.T@y
    for scale in (.25,3.):assert np.allclose((scale*u@x).T@y,scale*score)
    for p in mom:
        scaled=adam(z,s,p);n=64;W=np.zeros_like(z);mm=np.zeros_like(z);v=np.zeros_like(z)
        for t in range(1,p['T']+1):
            g=W@s-z/math.sqrt(n);mm=p['b1']*mm+(1-p['b1'])*g;v=p['b2']*v+(1-p['b2'])*g*g
            W-=.2/math.sqrt(n)*(mm/(1-p['b1']**t))/(np.sqrt(v/(1-p['b2']**t))+p['e']/math.sqrt(n))
        assert np.allclose(scaled,W*math.sqrt(n),atol=3e-13,rtol=2e-12)
    by={r['name']:r for r in items}
    for name,r in by.items():
        if '_T1_actual' in name:assert r['ranks']==by[name.replace('actual','proxy')]['ranks'] and r['relative_weight_mse']==0.
    return dict(stream_replay_dense_full_ranks=True,correct_candidate_removed=True,
                sign2_threshold=True,positive_scaling=True,adam_original_units=True,first_step_identity=True)

def resource_meta():
    cg=Path('/proc/self/cgroup').read_text().strip();path=cg.split('::')[-1]
    root=Path('/sys/fs/cgroup')/path.lstrip('/')
    return dict(cgroup=cg,**{name:(root/name).read_text().strip() for name in ('memory.max','memory.swap.max','memory.events')})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('smoke','formal'));args=parser.parse_args()
    digest=frozen();mom=json.loads((ROOT/'experiments/exp1/moments.json').read_text())
    assert all(os.environ.get(k)=='2' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
    resource.setrlimit(resource.RLIMIT_AS,(3_500_000_000,3_500_000_000))
    startfile=OUT/'budget_start.json'
    if args.mode=='smoke':
        assert not startfile.exists();start=time.time();startfile.write_text(json.dumps(dict(start_unix=start,deadline_unix=start+3600))+'\n')
        deadline=start+120
    else:
        assert json.loads((OUT/'smoke_metadata.json').read_text())['success']
        start=json.loads(startfile.read_text())['start_unix'];deadline=start+3120
    budget=Budget(deadline)
    def alarm(*_):raise TimeoutError('hard total deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,math.floor(deadline-time.time())))
    rows=OUT/(args.mode+'_rows.jsonl');events=OUT/(args.mode+'_events.jsonl')
    assert not rows.exists() and not events.exists(),'Preserve existing runs'
    tasks=[(32,1.,0,False,False),(128,1.,1,False,False)] if args.mode=='smoke' else []
    if args.mode=='formal':
        for seed in range(16):
            tasks.extend((d,k,seed,d==512,False) for d,k in ((128,1.),(256,1.),(512,1.),(128,2.),(256,2.),(512,2.),(128,.5),(256,.5)))
        tasks.extend((512,k,seed,True,False) for seed in range(16,24) for k in (1.,2.))
        tasks.extend((d,k,seed,False,True) for seed in range(16,P2_SEEDS) for d,k in ((128,1.),(128,2.),(256,1.),(256,2.)))
    completed=0;error=None;checks={};resources=resource_meta()
    try:
        checks=validate(budget,mom)
        with rows.open('x',buffering=1) as output,events.open('x',buffering=1) as log:
            for d,k,seed,high,p2only in tasks:
                try:
                    budget.check();items,event=configuration(d,k,seed,90 if args.mode=='smoke' else 10,high,mom,budget,p2only=p2only)
                    for item in items:output.write(json.dumps(item,separators=(',',':'))+'\n')
                    completed+=1;event['event']='completed';log.write(json.dumps(event)+'\n')
                    print(json.dumps(event),flush=True)
                except TimeoutError as ex:
                    error=str(ex);log.write(json.dumps(dict(event='budget_stop',d=d,kappa=k,seed=seed,reason=error))+'\n');break
    except Exception as ex:
        error=repr(ex);raise
    finally:
        signal.alarm(0)
        metadata=dict(mode=args.mode,success=completed==len(tasks) and error is None,completed=completed,planned=len(tasks),
            error=error,checks=checks,prereg_sha256=digest,elapsed_seconds=time.monotonic()-budget.start,
            total_elapsed_seconds=time.time()-start,max_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            max_single_matmul_seconds=budget.max_op,resource_start=resources,resource_end=resource_meta(),
            numpy=np.__version__,python=sys.executable,device='CPU',threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
        (OUT/(args.mode+'_metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata),flush=True)
    frozen()
if __name__=='__main__':main()
