"""Held-out dataset-level inference; PENDING is never a supervisor acceptance."""
import csv
import io
import json
import math
import resource
import sys
import numpy as np
from scipy.stats import t as student
from common_v2 import OUT, METHODS, PAIRS, ARMS, dump, read, sha, config, verify, plan, jobname, scheduled, expected_n
from model import predict
from run_v2 import lock_alphas
from protocol_v2 import credential

def interval(values,family,alpha=.025):
    a=np.asarray(values,dtype=float);n=len(a)
    if n==0: return dict(n=0,mean=None,se=None,critical=None,lower=None,upper=None,p=None,p_adjusted=None)
    mean=float(a.mean())
    if n==1: return dict(n=1,mean=mean,se=None,critical=None,lower=None,upper=None,p=None,p_adjusted=None)
    se=float(a.std(ddof=1)/math.sqrt(n));critical=float(student.ppf(1-alpha/(2*family),n-1))
    p=float(2*student.sf(abs(mean/se),n-1)) if se else (1. if mean==0 else 0.)
    return dict(n=n,mean=mean,se=se,critical=critical,lower=mean-critical*se,upper=mean+critical*se,
                p=p,p_adjusted=min(1.,family*p))

def equivalent(ci,band):
    return ci['lower'] is not None and ci['lower']>-band and ci['upper']<band

def csv_text(rows):
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['empty'],lineterminator='\n')
    w.writeheader();w.writerows(rows);return f.getvalue()

def validate_array_file(p,job,cfg,manifest,lock):
    d=job[1];Q=cfg['q_cal'] if job[0]=='cal' else cfg['q_eval'];n=d*d//4
    with np.load(p,allow_pickle=False) as f:
        keys={'ids','ranks','metadata'} if job[0]=='cal' else {'ids','ranks','metadata','alphas','valid','scheduled'}
        if set(f.files)!=keys: raise ValueError('array key set')
        arrays={k:f[k] for k in f.files}
    ids=arrays['ids'];rr=arrays['ranks'];meta=json.loads(str(arrays['metadata']))
    if ids.dtype!=np.int64 or ids.shape!=(Q,) or len(set(ids.tolist()))!=Q or np.any((ids<0)|(ids>=n)):
        raise ValueError('query ids')
    if meta.get('job')!=job or meta.get('manifest')!=manifest: raise ValueError('metadata binding')
    if rr.dtype!=np.int64: raise ValueError('rank dtype')
    if job[0]=='cal':
        if rr.shape!=(cfg['grid_count'],Q) or np.any((rr<1)|(rr>n)): raise ValueError('calibration ranks')
    else:
        shape=(len(ARMS),len(cfg['targets']),len(METHODS))
        if rr.shape!=shape+(2,Q) or arrays['alphas'].shape!=shape or arrays['alphas'].dtype!=np.float64:
            raise ValueError('evaluation shapes/dtypes')
        if arrays['valid'].shape!=shape or arrays['valid'].dtype!=bool: raise ValueError('valid mask')
        if meta.get('lock_sha256')!=sha(p.parent/'lock.json'): raise ValueError('lock binding')
        mask=np.array([scheduled(cfg,job[2],t) for t in cfg['targets']],dtype=bool)
        if arrays['scheduled'].dtype!=bool or not np.array_equal(arrays['scheduled'],mask): raise ValueError('schedule mask')
        for a,arm in enumerate(ARMS):
            for ti,target in enumerate(cfg['targets']):
                for mi,method in enumerate(METHODS):
                    cell=next(c for c in lock['cells'] if (c['d'],c['target'],c['method'])==(d,target,method))
                    v=cell['empirical']['alpha'] if arm=='empirical' else cell['analytic']
                    idx=(a,ti,mi);ok=v is not None and bool(mask[ti]);v=v if ok else None
                    if bool(arrays['valid'][idx])!=ok or arrays['alphas'][idx]!=(v if ok else 0.):
                        raise ValueError('locked alpha mismatch')
                    if ok and np.any((rr[idx]<1)|(rr[idx]>n)): raise ValueError('evaluation rank range')
                    if not ok and np.any(rr[idx]!=0): raise ValueError('missing alpha ranks must be zero sentinel')
    return arrays

def make_report(data,manifest,cfg):
    data_errors=[];available=[];raw_hashes={};datasets={};lock=None
    if (data/'lock.json').exists():
        raw_hashes['lock.json']=sha(data/'lock.json')
        try:
            lock=read(data/'lock.json')
            if lock!=lock_alphas(data,cfg,manifest): raise ValueError('lock does not recompute from all calibration files')
        except Exception as e: data_errors.append('lock: '+repr(e));lock=None
    for job in plan(cfg):
        p=data/jobname(job);exists=p.is_file();state='not_started'
        if exists and job!=['lock']:
            raw_hashes[p.name]=sha(p)
            try:
                arr=validate_array_file(p,job,cfg,manifest,lock)
                state='complete'
                if job[0]=='eval':
                    d,s=job[1:]; datasets[d,s]=dict(
                        recall=(arr['ranks']<=math.ceil(.1*(d*d//4))).mean(axis=-1),
                        valid=arr['valid'],alphas=arr['alphas'],scheduled=arr['scheduled'])
            except Exception as e: state='invalid';data_errors.append(p.name+': '+repr(e))
        elif job==['lock'] and exists: state='complete' if lock is not None else 'invalid'
        elif p.with_suffix('.partial').exists(): state='partial'
        available.append(dict(job=json.dumps(job),state=state,present=exists))
    completed=[]
    for (d,seed),ds in sorted(datasets.items()):
        for a,arm in enumerate(ARMS):
            for ti,target in enumerate(cfg['targets']):
                for mi,method in enumerate(METHODS):
                    ok=bool(ds['valid'][a,ti,mi])
                    completed.append(dict(d=d,seed=seed,arm=arm,target=target,method=method,
                        scheduled=bool(ds['scheduled'][ti]),valid=ok,
                        alpha=float(ds['alphas'][a,ti,mi]) if ok else None,
                        old=float(ds['recall'][a,ti,mi,0]) if ok else None,
                        new=float(ds['recall'][a,ti,mi,1]) if ok else None))
    matching=[];curves=[];old_pairs=[];scaled=[];gates={} 
    for d in cfg['dims']:
        for a,arm in enumerate(ARMS):
            for ti,target in enumerate(cfg['targets']):
                primary=arm=='empirical' and d==128 and target==cfg['primary_target']
                N=expected_n(cfg,target)
                mf=cfg['match_family'] if primary else 21
                of=cfg['old_family'] if primary else 9
                role='primary' if primary else 'descriptive'
                abs_gates={}
                for mi,method in enumerate(METHODS):
                    vals=[];alpha=None
                    for s in range(cfg['n_eval']):
                        ds=datasets.get((d,s))
                        if ds is not None and ds['valid'][a,ti,mi]:
                            vals.append(ds['recall'][a,ti,mi]);alpha=float(ds['alphas'][a,ti,mi])
                    vals=np.array(vals).reshape((-1,2))
                    ci=interval(vals[:,1]-target,mf)
                    eligible=len(vals)==N
                    match=eligible and equivalent(ci,cfg['match_band'])
                    abs_gates[method]=match
                    matching.append(dict(arm=arm,d=d,target=target,role=role,expected_n=N,type='absolute',label=method,
                        eligible=eligible,matched=match,**ci))
                    for task,name in enumerate(['old','new']):
                        cc=interval(vals[:,task],24,.05)
                        pr=predict(method,alpha,d,name) if alpha is not None else None
                        curves.append(dict(arm=arm,d=d,target=target,role=role,expected_n=N,method=method,task=name,alpha=alpha,**cc))
                        scaled.append(dict(arm=arm,d=d,target=target,role=role,expected_n=N,method=method,task=name,alpha=alpha,
                            n=cc['n'],obs=cc['mean'],R_infty=pr['limit'] if pr else None,B_R=pr['B'] if pr else None,
                            scaled_residual=d*(cc['mean']-pr['limit']) if pr and cc['mean'] is not None else None,
                            scaled_lower=d*(cc['lower']-pr['limit']) if pr and cc['lower'] is not None else None,
                            scaled_upper=d*(cc['upper']-pr['limit']) if pr and cc['upper'] is not None else None,
                            coefficient_confirmation=False))
                for ma,mb in PAIRS:
                    ia,ib=METHODS.index(ma),METHODS.index(mb);old=[];new=[];aa=ab=None
                    for s in range(cfg['n_eval']):
                        ds=datasets.get((d,s))
                        if ds is not None and ds['valid'][a,ti,ia] and ds['valid'][a,ti,ib]:
                            old.append(ds['recall'][a,ti,ia,0]-ds['recall'][a,ti,ib,0])
                            new.append(ds['recall'][a,ti,ia,1]-ds['recall'][a,ti,ib,1])
                            aa=float(ds['alphas'][a,ti,ia]);ab=float(ds['alphas'][a,ti,ib])
                    matchci=interval(new,mf);eligible=len(old)==N
                    gate=eligible and abs_gates[ma] and abs_gates[mb] and equivalent(matchci,cfg['match_band'])
                    gates[arm,d,target,ma,mb]=gate
                    matching.append(dict(arm=arm,d=d,target=target,role=role,expected_n=N,type='pairwise',label=ma+'-'+mb,
                        eligible=eligible,matched=eligible and equivalent(matchci,cfg['match_band']),**matchci))
                    ci=interval(old,of);pr1=predict(ma,aa,d,'old') if aa is not None else None
                    pr2=predict(mb,ab,d,'old') if ab is not None else None
                    finite=pr1['finite']-pr2['finite'] if pr1 and pr2 else None
                    lim=pr1['limit']-pr2['limit'] if pr1 and pr2 else None
                    shift=finite-lim if finite is not None else None
                    resid=dict(ci)
                    if finite is not None and ci['lower'] is not None:
                        resid['lower']-=finite;resid['upper']-=finite
                    agree=finite is not None and eligible and equivalent(resid,cfg['agreement_band'])
                    direction=gate and finite is not None and finite>0 and ci['lower'] is not None and ci['lower']>0
                    correction='inconclusive'
                    if ci['lower'] is not None and lim is not None and shift!=0:
                        if (shift>0 and ci['lower']>lim) or (shift<0 and ci['upper']<lim): correction='same_direction_resolved'
                        elif (shift>0 and ci['upper']<lim) or (shift<0 and ci['lower']>lim): correction='opposite_direction_resolved'
                    old_pairs.append(dict(arm=arm,d=d,target=target,role=role,expected_n=N,pair=ma+'-'+mb,matched=gate,
                        eligible=eligible,Delta_infty=lim,Delta_finite=finite,finite_shift=shift,
                        residual_lower=resid['lower'] if finite is not None else None,
                        residual_upper=resid['upper'] if finite is not None else None,
                        direction_supported=direction,finite_agreement=gate and agree,
                        correction_direction=correction,**ci))
    emp=[r for r in old_pairs if r['role']=='primary']
    matches=[r for r in matching if r['role']=='primary']
    plan_complete=not data_errors and all(r['state']=='complete' for r in available)
    decisions=dict(primary_plan_complete=plan_complete,primary_pair_count=len(emp),
        primary_match_count=len(matches),
        primary_all7_matching=plan_complete and len(matches)==7 and all(r['matched'] for r in matches),
        primary_all3_directions=plan_complete and len(emp)==3 and all(r['direction_supported'] for r in emp),
        primary_all3_finite_agreement=plan_complete and len(emp)==3 and all(r['finite_agreement'] for r in emp),
        coefficient_confirmation=False,descriptive_confirmation=False)
    return dict(manifest=manifest,data_errors=data_errors,raw_hashes=raw_hashes,
        all_completed=completed,availability=available,curves=curves,matching=matching,old_pairs=old_pairs,
        scaled_residuals=scaled,decisions=decisions)

def main():
    if sys.argv[1:]!=['--credential-worker']: raise ValueError('supervisor only')
    cfg=config();resource.setrlimit(resource.RLIMIT_CPU,(cfg['analysis_cpu_hard'],cfg['analysis_cpu_hard']))
    mh=verify();cred=credential('analysis',mh);data=OUT/'formal';dest=data/'analysis';dest.mkdir()
    report=make_report(data,mh,cfg)
    dump(dest/'report.json',report)
    files={'report.json':sha(dest/'report.json')}
    for name,key in [('curves.csv','curves'),('matching.csv','matching'),('old_pairs.csv','old_pairs'),
                     ('scaled_residuals.csv','scaled_residuals'),('availability.csv','availability'),('all_completed.csv','all_completed')]:
        with (dest/name).open('x',newline='') as f:
            f.write(csv_text(report[key]));f.flush()
            import os
            os.fsync(f.fileno())
        files[name]=sha(dest/name)
    dump(dest/'PENDING.json',dict(manifest=mh,credential=cred,files=files,usage=__import__('common_v2').usage(),confirmation_valid=False))

if __name__=='__main__': main()
