#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
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
            raise TimeoutError('EXP3 preregistered deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>1_700_000_000:
            raise MemoryError('EXP3 preregistered RSS stop')


def verify_frozen():
    entries={}
    for line in (HERE/'FROZEN.sha256').read_text().splitlines():
        digest,filename=line.split(maxsplit=1)
        path=(ROOT/filename).resolve()
        assert path.is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,filename
        entries[filename]=digest
    digest=hashlib.sha256((ROOT/'PREREG_EXP3.md').read_bytes()).hexdigest()
    assert digest==(ROOT/'PREREG_EXP3.sha256').read_text().split()[0]
    return digest,entries


def smoke(ctx):
    import ast
    for source in HERE.glob('*.py'):
        ast.parse(source.read_text(),filename=str(source))
    p=model.problem(6901,16,1.5)
    e=model.snapshot(6901,16,1.5,0,1.)
    rng=np.random.default_rng([20261106,399])
    g=rng.normal(size=(16,4))
    u,_=np.linalg.qr(rng.normal(size=(16,16)))
    v,_=np.linalg.qr(rng.normal(size=(4,4)))
    rotation=float(np.max(np.abs(u@model.polar(g)@v-model.polar(u@g@v))))
    assert rotation<1e-11
    low=np.zeros((16,4));low[0,0]=2.
    assert np.max(np.abs(model.polar(low)-low/2))<1e-14
    m=model.bridge(p,e,16)
    prot=dict(p,cov=v.T@p['cov']@v)
    rotated=model.bridge(prot,e@v,16)
    berr=float(np.max(np.abs(rotated['pred']-m['pred']@v)))
    assert berr<1e-11
    # Independent row covariance contraction, not a second implementation of M.
    rowrisk=sum(p['lam'][i]*np.trace(m['B']@(p['lam'][i]*m['S']+
                np.outer(m['A'][i],m['A'][i]))@m['B'])/16 for i in range(16))
    assert abs(rowrisk-m['riskvar'])<1e-12
    limit=model.bridge(p,e,10**12)['pred']
    limit_error=float(np.linalg.norm(limit-model.polar(p['lam'][:,None]*e)))
    assert limit_error<1e-8
    local=model.measure(ctx,390,6901,p,e,4,128,(0,))
    # Online smoke keeps the formal snapshot count but only 32 update steps.
    train=model.train(ctx,6902,16,1.5,4,steps=32)
    ctx.save('smoke',dict(rotation_max_error=rotation,bridge_rotation_error=berr,
        rank_deficient_pass=True,analytic_row_contraction_error=abs(rowrisk-m['riskvar']),
        large_batch_limit_error=limit_error,local=local,training=train))
    ctx.event('smoke_pass',rotation_max_error=rotation,bridge_rotation_error=berr)


def formal(ctx):
    ctx.save('locked_model',dict(formula='AB, B=(E G^T G)^(-1/2)',fitted_parameters=0,
        eta=.01,calibration_samples=0,gram_epsilon=.5))
    ctx.event('locked_before_holdout',fitted_parameters=0)
    records=[]
    for seed in (6101,6102,6103,6104):
        for n in (16,32,64,128):
            for alpha in (1.5,3.):
                p=model.problem(seed,n,alpha)
                for ai,amp in enumerate((1.,.25,.0625)):
                    e=model.snapshot(seed,n,alpha,ai,amp)
                    for b in (4,16,64,256):
                        row=model.measure(ctx,303,seed,p,e,b,4096,(ai,))
                        row.update(ai=ai,amplitude=amp)
                        records.append(row)
                        ctx.save('holdout',records)
                    if len(records)%24==0:
                        ctx.event('holdout_progress',completed=len(records),expected=384)
    assert len(records)==384
    training=[]
    for seed in (6201,6202):
        for n in (16,64):
            for alpha in (1.5,3.):
                for b in (4,64):
                    training.append(model.train(ctx,seed,n,alpha,b))
                    ctx.save('training',training)
                    ctx.event('training_progress',completed=len(training),expected=16)
    assert len(training)==16 and sum(len(r['tests']) for r in training)==64


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--out',required=True)
    ap.add_argument('--scope',required=True)
    args=ap.parse_args()
    out=(ROOT/args.out).resolve()
    assert out.is_relative_to(ROOT)
    digest,entries=verify_frozen()
    ctx=Context(out,100 if args.smoke else 2900)
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
        meta['status']='incomplete';meta['exception']=repr(exc)
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
