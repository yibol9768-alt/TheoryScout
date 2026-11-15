import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[key]='1'
import argparse, itertools, json, time, resource, signal, hashlib, warnings
from pathlib import Path
warnings.filterwarnings('ignore',message='Unrecognized options detected')
import numpy as np
from core import *
ROOT=Path(__file__).resolve().parent

def configs(layer):
    channels=[(lam,ws,wr) for ws,wr in [(1.,1.),(.5,1.3)] for lam in (7.,31.,112.,255.)] if layer=='E1' else list(itertools.product((.3,1.),(.3,1.,2.)))
    out=[]
    for seed in (0,1,2):
        for channel in channels: out.append((layer,seed,32,0,4/15,0.,0.,channel))
    for g,eps,dist,reader,seed,channel in itertools.product((32,16,64),(4/15,.1,.4),(0,.8,1.4),((0.,0.),(.1,0.),(0.,.1)),(0,1,2),channels):
        c=(layer,seed,g,dist,eps,*reader,channel)
        if c not in out: out.append(c)
    return out

def smoke_checks():
    # Verify fast all-count scan against direct masks, including adversarial ties.
    rr=np.random.default_rng(918); g=5; n=17; z=np.zeros(g,int)
    for tied in (False,True):
        scores=rr.normal(size=(n,5*g)); scores[:]=0 if tied else scores
        ii=rr.integers(g,size=n); jj=rr.integers(K,size=n); cs=rr.random(n)<.7; cr=np.ones(n,bool); guess=rr.random(n)<.5; perm=rr.permutation(g)
        fast,pairs=scan_counts_fast(scores,ii,jj,cs,cr,guess,perm)
        slow,_=scan_counts(scores,ii,jj,cs,cr,guess,perm)
        assert np.array_equal(fast,slow)
        for val,(nr,ns) in zip(fast,pairs):
            z[:]=0; z[perm[:nr]]=2; z[perm[nr:nr+ns]]=1
            hit=retrieve(scores,z,ii,jj,'E2',(1.,1.))
            good=np.where(hit,np.where(z[ii]==1,cs,cr),guess)
            assert val==good.sum(),(tied,nr,ns,val,good.sum())
    return {'count_scan_direct_and_ties':'PASS','batch_max':128}

def main():
    pa=argparse.ArgumentParser(); pa.add_argument('--smoke',action='store_true'); args=pa.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(1900000000,1900000000))
    start=time.monotonic(); mode=os.environ.get('EXP1_SMOKE_DIR','smoke') if args.smoke else 'formal'; outdir=ROOT/mode; outdir.mkdir(exist_ok=True)
    # Refuse silently overwriting a prior run.
    if (outdir/'results.jsonl').exists(): raise RuntimeError('Output already exists: '+str(outdir))
    prereg=ROOT.parent.parent/'PREREG_EXP1.md'; digest=hashlib.sha256(prereg.read_bytes()).hexdigest()
    assert digest==(ROOT.parent.parent/'PREREG_EXP1.sha256').read_text().split()[0]
    meta={'prereg_sha256':digest,'start_unix':time.time(),'mode':mode,'numpy':np.__version__,'python':os.sys.executable,'threads':1,'limits':{'MemoryMax':'2G','MemorySwapMax':0,'RLIMIT_AS':1900000000},'smoke_checks':smoke_checks()}
    (outdir/'metadata.json').write_text(json.dumps(meta,indent=2))
    cache={}; status=[]; lastlog=start
    def stop(sig,frame): raise TimeoutError('external termination')
    signal.signal(signal.SIGTERM,stop)
    try:
        with (outdir/'results.jsonl').open('w',buffering=1) as handle:
            for layer in ('E1','E2'):
                begin=time.monotonic(); deadline=begin+(600 if layer=='E1' else 1680)
                todo=configs(layer)
                if args.smoke: todo=[('E1',0,32,0,4/15,0.,0.,(31.,1.,1.))] if layer=='E1' else [('E2',0,32,0,4/15,0.,0.,(1.,1.))]
                skipped=False
                for number,c in enumerate(todo):
                    cid=config_id(c)
                    if skipped or time.monotonic()>deadline or time.monotonic()-start>3300 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>1700000000:
                        status.append({'id':cid,'state':'NOT_RUN_RESOURCE_DEADLINE'}); skipped=True; continue
                    _,seed,g,dist,eps,ns,nr,ch=c; key=(layer,seed,g,ch)
                    if key not in cache: cache[key]=calibrate(layer,seed,g,ch,256 if args.smoke else 4096)
                    n=256 if args.smoke else 8192
                    # Timed pilot reuses frozen parameters, never selects settings by outcome.
                    permcount=10
                    pilot_start=time.monotonic()
                    pilot=run_condition(c,cache[key],outdir,n=256,permutations=3)
                    pilot_seconds=time.monotonic()-pilot_start
                    if layer=='E2' and dist==0 and pilot_seconds*32*10/3>max(1,deadline-time.monotonic()): permcount=3
                    res=pilot if args.smoke else run_condition(c,cache[key],outdir,n=n,permutations=permcount)
                    res['pilot_seconds_256']=pilot_seconds
                    handle.write(json.dumps(res,ensure_ascii=False)+'\n'); status.append({'id':cid,'state':'COMPLETED'})
                    now=time.monotonic()
                    if now-lastlog>=50 or number<3:
                        print(json.dumps({'layer':layer,'condition':cid,'completed':number+1,'elapsed_seconds':now-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'condition_seconds':res['elapsed_seconds']}),flush=True); lastlog=now
    except (TimeoutError,MemoryError) as e:
        meta['interruption']=str(e)
    finally:
        meta.update(elapsed_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,finished_unix=time.time())
        (outdir/'metadata.json').write_text(json.dumps(meta,indent=2)); (outdir/'status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(meta),flush=True)
if __name__=='__main__': main()
