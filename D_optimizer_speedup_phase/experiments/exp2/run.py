#!/usr/bin/env python3
"""Frozen EXP2 runner. No network, external project imports, or GPU."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE))
import numpy as np
import model


class Context:
    def __init__(self,out,seconds):
        self.out=out
        out.mkdir(parents=True,exist_ok=False)
        self.start=time.monotonic()
        self.deadline=self.start+seconds

    def save(self,name,value):
        path=self.out/(name+'.json')
        tmp=path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        tmp.replace(path)

    def event(self,event,**kw):
        row=dict(event=event,elapsed=time.monotonic()-self.start,**kw)
        with (self.out/'events.jsonl').open('a') as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)

    def guard(self):
        if time.monotonic()>self.deadline:
            raise TimeoutError('preregistered global deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>1_700_000_000:
            raise MemoryError('preregistered soft RSS stop')


def verify_frozen():
    entries={}
    for line in (HERE/'FROZEN.sha256').read_text().splitlines():
        digest, filename=line.split(maxsplit=1)
        path=(ROOT/filename).resolve()
        assert path.is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,filename
        entries[filename]=digest
    digest=hashlib.sha256((ROOT/'PREREG_EXP2.md').read_bytes()).hexdigest()
    assert digest==(ROOT/'PREREG_EXP2.sha256').read_text().split()[0]
    return digest,entries


def smoke(ctx):
    p=model.problem(9001,16,1.5)
    e=model.snapshot(9001,16,1.5,0,1.)
    rng=np.random.default_rng([20261106,90])
    x=rng.normal(size=(4,16))*np.sqrt(p['lam'])
    eps=rng.normal(size=(4,4))@p['root'].T
    g=x.T@(x@e+eps)/4
    u,_=np.linalg.qr(rng.normal(size=(16,16)))
    err=float(np.max(np.abs(u@model.polar(g)-model.polar(u@g))))
    assert err<1e-11
    low=np.zeros((16,4));low[0,0]=2.
    assert np.max(np.abs(model.polar(low)-low/2))<1e-14
    row=model.measure(ctx,90,9001,16,1.5,0,1.,4,64,1.,1.,1.,1.)
    train=model.train(ctx,9001,16,1.5,4,.01,64,1.,1.)
    # Gaussian covariance identities, finite rather than asymptotic checks.
    mt=model.moments(p,e,4,1.)
    assert abs(np.sum(mt['rowvar'])-np.trace(mt['output']))<1e-12
    ctx.save('smoke',dict(rotation_max_error=err,rank_deficient_pass=True,
                         analytic_trace_identity_pass=True,local=row,training=train))
    ctx.event('smoke_pass',rotation_max_error=err)


def formal(ctx):
    calibration=[]
    for seed in (1001,1002):
        for alpha in (1.5,3.):
            for b in (4,16,64,256):
                row=model.measure(ctx,10,seed,32,alpha,0,1.,b,2048,1.,1.,1.,1.)
                calibration.append(row)
                ctx.save('calibration',calibration)
    locked=dict(state=model.fit(ctx,calibration,True),fixed=model.fit(ctx,calibration,False))
    ctx.save('locked_model',locked)
    ctx.event('locked_before_holdout',locked=locked)
    nu,c=locked['state']['nu'],locked['state']['c']
    nuf,cf=locked['fixed']['nu'],locked['fixed']['c']
    records=[]
    for seed in (2001,2002,2003,2004):
        for n in (16,32,64,128):
            for alpha in (1.5,3.):
                for ai,amp in enumerate((1.,.25,.0625)):
                    for b in (4,16,64,256):
                        records.append(model.measure(ctx,20,seed,n,alpha,ai,amp,b,2048,nu,c,nuf,cf))
                        ctx.save('holdout',records)
                    if len(records)%48==0:
                        ctx.event('holdout_progress',completed=len(records),expected=384)
    assert len(records)==384
    calibration_train=[]
    locked_lr={}
    for alpha in (1.5,3.):
        for b in (4,16,64,256):
            candidates=[]
            for eta in (.003,.01,.03,.1,.3):
                row=model.train(ctx,3001,32,alpha,b,eta,256,nu,c)
                calibration_train.append(row)
                candidates.append(row)
            selected=min(candidates,key=lambda r:(r['rows'][-1]['risk'],r['eta']))
            locked_lr[(alpha,b)]=selected['eta']
            ctx.save('training_calibration',dict(records=calibration_train,
                locked=[dict(alpha=a,b=bb,eta=ee,endpoint=ee in (.003,.3)) for (a,bb),ee in locked_lr.items()]))
    training=[]
    for seed in (4001,4002,4003):
        for n in (16,32,64,128):
            for alpha in (1.5,3.):
                for b in (4,16,64,256):
                    training.append(model.train(ctx,seed,n,alpha,b,locked_lr[(alpha,b)],2048,nu,c,True))
                    ctx.save('training',training)
            ctx.event('training_progress',completed=len(training),expected=96)
    assert len(training)==96 and len(calibration_train)==40


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--out',required=True)
    ap.add_argument('--scope',required=True)
    args=ap.parse_args()
    out=(ROOT/args.out).resolve()
    assert out.is_relative_to(ROOT)
    digest,entries=verify_frozen()
    ctx=Context(out,100 if args.smoke else 3200)
    props=subprocess.check_output(['systemctl','--user','show',args.scope,
        '-p','MemoryMax','-p','MemorySwapMax','-p','ControlGroup'],text=True)
    (out/'scope_properties.txt').write_text(props)
    assert 'MemoryMax=2147483648' in props and 'MemorySwapMax=0' in props
    meta=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              python=sys.executable,numpy=np.__version__,prereg_sha256=digest,frozen=entries,
              mode='smoke' if args.smoke else 'formal',scope=args.scope,scope_properties=props,
              threads={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
              status='running')
    ctx.save('manifest',meta)
    try:
        smoke(ctx) if args.smoke else formal(ctx)
        meta['status']='complete'
    except Exception as exc:
        meta['status']='incomplete'
        meta['exception']=repr(exc)
        (out/'exception.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        meta['elapsed_seconds']=time.monotonic()-ctx.start
        meta['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        meta['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        ctx.save('manifest',meta)
        ctx.event('finished',status=meta['status'],elapsed_seconds=meta['elapsed_seconds'],
                  peak_rss_bytes=meta['peak_rss_bytes'])
    if meta['status']!='complete':
        raise SystemExit(1)


if __name__=='__main__':
    main()
