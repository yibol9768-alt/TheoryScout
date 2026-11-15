"""Deterministic fabricated ranks, matrices and fault injection only. No scientific RNG."""
import copy
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from scipy.stats import t
from common_v2 import OUT, METHODS, ARMS, dump, read, commit, sha, plan, jobname, usage, config, environment
from model import MOMENTS, predict, pava, select, ranks_many, analytic_alpha, alpha_grid
from analyze_v2 import interval, equivalent, make_report, csv_text
from run_v2 import lock_alphas
from protocol_v2 import Events, inspect, publish, verify_accepted, credential, digest

RESULTS=[]
def check(name,fn):
    try: fn();RESULTS.append(dict(name=name,passed=True))
    except Exception as e: RESULTS.append(dict(name=name,passed=False,error=repr(e)))
def require(cond):
    if not cond: raise AssertionError('synthetic condition')
def rejects(fn):
    try: fn()
    except Exception: return
    raise AssertionError('should reject')

BASE=OUT/'synthetic_resume'

def fixture(name):
    data=BASE/name;data.mkdir()
    cfg=copy.deepcopy(config());cfg.update(dims=[128],n_cal=3,n_eval=3,q_cal=40,q_eval=40)
    mh='a'*64;creds={'runner':'b'*64,'analysis':'c'*64}
    dump(data/'launch.json',dict(manifest=mh,pid=os.getpid(),credentials=creds,config=cfg))
    dump(data/'start.json',dict(manifest=mh,credential=creds['runner'],streams=sha(OUT/'streams.json'),config=cfg,usage=usage()))
    ev=Events(data/'events.jsonl',mh,creds['runner']);complete=[];locked=None
    for job in plan(cfg):
        ev.append('started',job=job);p=data/jobname(job)
        if job[0]=='cal':
            _,d,m,s=job;rr=np.full((cfg['grid_count'],40),d*d//4,dtype=np.int64)
            for i in range(len(rr)): rr[i,:int(round(i/(len(rr)-1)*.7*40))]=1
            commit(p,dict(ids=np.arange(40,dtype=np.int64),ranks=rr,
                metadata=np.array(json.dumps(dict(job=job,manifest=mh)))))
        elif job==['lock']:
            locked=lock_alphas(data,cfg,mh);dump(p,locked)
        else:
            _,d,s=job;shape=(2,3,4)
            rr=np.full(shape+(2,40),d*d//4,dtype=np.int64);aa=np.zeros(shape);valid=np.zeros(shape,dtype=bool)
            for a,arm in enumerate(ARMS):
                for ti,target in enumerate(cfg['targets']):
                    for mi,m in enumerate(METHODS):
                        c=next(c for c in locked['cells'] if c['method']==m and c['target']==target)
                        v=c['empirical']['alpha'] if a==0 else c['analytic']
                        if v is None: rr[a,ti,mi]=0;continue
                        aa[a,ti,mi]=v;valid[a,ti,mi]=True
                        rr[a,ti,mi,0,:32-4*mi]=1
                        rr[a,ti,mi,1,:int(round(target*40))]=1
            commit(p,dict(ids=np.arange(40,dtype=np.int64),ranks=rr,alphas=aa,valid=valid,
                metadata=np.array(json.dumps(dict(job=job,manifest=mh,lock_sha256=sha(data/'lock.json'))))))
        complete.append(job);ev.append('complete',job=job,sha256=sha(p))
    ev.append('stopped',reason='planned_complete',active=None,error=None);ev.close()
    dump(data/'runner_status.json',dict(reason='planned_complete',active=None,error=None,
        completed=complete,event_tail=ev.prev,usage=usage(),wall=1.))
    report=make_report(data,mh,cfg);dest=data/'analysis';dest.mkdir()
    dump(dest/'report.json',report);files={'report.json':sha(dest/'report.json')}
    for name,key in [('curves.csv','curves'),('matching.csv','matching'),('old_pairs.csv','old_pairs'),
                     ('scaled_residuals.csv','scaled_residuals'),('availability.csv','availability'),('all_completed.csv','all_completed')]:
        with (dest/name).open('x',newline='') as f: f.write(csv_text(report[key]))
        files[name]=sha(dest/name)
    dump(dest/'PENDING.json',dict(manifest=mh,credential=creds['analysis'],files=files,usage=usage(),confirmation_valid=False))
    return data,cfg,mh

GOOD=[dict(stage='runner',returncode=0,killed=False),dict(stage='analysis',returncode=0,killed=False)]

def protocol_good():
    data,cfg,mh=fixture('good')
    require(inspect(data,mh,cfg)['errors']==[])
    require(publish(data,mh,cfg,GOOD,2.,1.,1.)['confirmation_valid'])
    observer_receipt(data,mh)
    require(verify_accepted(data,mh,cfg)['confirmation_valid'])
    report=read(data/'analysis/report.json')
    require(report['decisions']['all18_directions'])
    require(not report['decisions']['coefficient_confirmation'])

def fault(name,mutate,stages=GOOD,wall=2.,children=1.,self_cpu=1.):
    data,cfg,mh=fixture(name);mutate(data)
    status=publish(data,mh,cfg,stages,wall,children,self_cpu)
    require(not status['confirmation_valid']);require(not (data/'ACCEPTED.json').exists())

def overwrite_json(path,func):
    obj=read(path);func(obj);path.write_text(json.dumps(obj)) # synthetic fixtures only

def numeric():
    c=interval([.1,.2,.3],18)
    require(abs(c['critical']-t.ppf(1-.025/36,2))<1e-12)
    require(abs(c['se']-.1/math.sqrt(3))<1e-12)
    require(interval([],42)['mean'] is None and interval([.2],42)['lower'] is None)
    require(interval([0,0],18)['p']==1 and interval([1,1],18)['p']==0)
    require(equivalent(dict(lower=-.009,upper=.009),.01))
    require(not equivalent(dict(lower=-.01,upper=.009),.01))
    require(not equivalent(dict(lower=-.009,upper=.01),.01))

def selection():
    require(np.allclose(pava([0,.4,.2,.8]),[0,.3,.3,.8]))
    require(select([0,1,2],[.1,.4,.6],.5)['alpha']==1.5)
    require(select([0,1],[.1,.2],.5)['reason']=='unbracketed')
    require(select([0,1,2],[0,.5,.1],.3)['reason']=='nonmonotonic')
    require(select([0,1,2],[.1,.3,.3],.3)['alpha']==1.)

def ranking():
    x=np.sin(np.arange(24).reshape(4,6)+.1);y=np.cos(np.arange(24).reshape(4,6)+.3)
    b=np.sin(np.arange(16).reshape(4,4)+.4);v=np.cos(np.arange(16).reshape(4,4)+.8)
    ids=np.array([4,0,3],dtype=np.int64);aa=[0,.2,.7]
    rr=ranks_many(b,v,aa,x,y,ids,qc=2,cc=2)
    for i,a in enumerate(aa):
        sc=y.T@((b+a*v)@x[:,ids])
        for j,k in enumerate(ids): require(rr[i,j]==1+sum(sc[c,j]>sc[k,j] for c in range(6) if c!=k))
    tied=ranks_many(np.zeros((4,4)),v*0,aa,x,y,ids,qc=2,cc=2)
    require(np.array_equal(tied,np.tile(ids+1,(3,1))))

def formula():
    # Independent GD moment identities and physical-epsilon derivative difference.
    require(MOMENTS['GD']==(1.,2.,0.,1.))
    require(abs(MOMENTS['Sign'][0]-1/math.sqrt(math.pi))<1e-12)
    for m in METHODS:
        for target in [.35,.425,.5]:
            for d in [128,256]:
                a=analytic_alpha(m,d,target)
                require(a is not None and abs(predict(m,a,d,'new')['finite']-target)<1e-10)
    # Frozen legacy prediction values are a regression oracle, not imported executable code.
    import csv
    with (OUT.parent/'y3_checks/Y3_frozen_predictions.csv').open() as f:
        for row in csv.DictReader(f):
            if float(row['r'])==0 and row['method'] in METHODS:
                p=predict(row['method'],float(row['amplitude']),int(row['d']),row['task'])
                require(abs(p['B']-float(row['B_recall']))<2e-8)

def kernel_limits():
    env=dict(os.environ)
    # No experiment imports, matrices or queries in these process-kill checks.
    code='import resource;resource.setrlimit(resource.RLIMIT_CPU,(1,2));exec("while True: pass")'
    p=subprocess.run([sys.executable,'-B','-c',code],env=env,timeout=6)
    require(p.returncode==-signal.SIGXCPU)
    code='import resource,signal;resource.setrlimit(resource.RLIMIT_CPU,(1,2));signal.signal(signal.SIGXCPU,signal.SIG_IGN);exec("while True: pass")'
    p=subprocess.run([sys.executable,'-B','-c',code],env=env,timeout=6)
    require(p.returncode==-signal.SIGKILL)
    p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(10)'],start_new_session=True)
    try: p.wait(timeout=.05)
    except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL)
    require(p.wait()==-signal.SIGKILL)

def rejection_after_commit():
    data,cfg,mh=fixture('mutate_accepted');require(publish(data,mh,cfg,GOOD,2.,1.,1.)['confirmation_valid'])
    observer_receipt(data,mh)
    with (data/'analysis/old_pairs.csv').open('a') as f: f.write('tamper\n')
    rejects(lambda:verify_accepted(data,mh,cfg))

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(100,110));started=usage();BASE.mkdir()
    environment()
    check('dataset_t_intervals_zero_variance_N0_N1_strict_boundaries',numeric)
    check('PAVA_interpolation_plateau_nonmonotonic_unbracketed',selection)
    check('affine_full_ranks_direct_GEMM_ties_chunks',ranking)
    check('BR_old_frozen_oracle_and_analytic_roots',formula)
    check('supervisor_recomputes_full_fixture_and_accepts',protocol_good)
    check('PENDING_alone_not_accepted',lambda:require(not (fixture('pending')[0]/'ACCEPTED.json').exists()))
    check('missing_pipe_credential',lambda:rejects(lambda:credential('runner','a'*64)))
    check('hard_killed_runner_invalid',lambda:fault('runnerkill',lambda p:None,
        [dict(stage='runner',returncode=-9,killed=True),GOOD[1]]))
    check('hard_killed_analysis_invalid',lambda:fault('analysiskill',lambda p:None,
        [GOOD[0],dict(stage='analysis',returncode=-9,killed=True)]))
    check('total_cpu_exceeded',lambda:fault('cpu',lambda p:None,children=21001.))
    check('total_wall_exceeded',lambda:fault('wall',lambda p:None,wall=12601.))
    check('supervisor_cpu_exceeded',lambda:fault('supervisorcpu',lambda p:None,self_cpu=101.))
    check('missing_npz',lambda:fault('missing',lambda p:(p/'eval_128_0.npz').unlink()))
    check('corrupt_npz',lambda:fault('corrupt',lambda p:(p/'eval_128_0.npz').write_bytes(b'broken')))
    check('extra_artifact',lambda:fault('extra',lambda p:(p/'unplanned.npz').write_bytes(b'broken')))
    check('partial_retained',lambda:fault('partial',lambda p:(p/'left.partial').write_bytes(b'x')))
    check('event_truncated',lambda:fault('truncated',lambda p:(p/'events.jsonl').write_text((p/'events.jsonl').read_text()[:-10])))
    check('event_unknown_field',lambda:fault('eventfield',lambda p:(p/'events.jsonl').write_text((p/'events.jsonl').read_text().replace('"index": 0','"unknown": 1, "index": 0',1))))
    check('credential_mismatch',lambda:fault('credential',lambda p:overwrite_json(p/'start.json',lambda o:o.update(credential='d'*64))))
    check('locked_alpha_tamper',lambda:fault('locktamper',lambda p:overwrite_json(p/'lock.json',lambda o:o['cells'][0]['empirical'].update(alpha=123.))))
    check('candidate_flag_tamper',lambda:fault('flag',lambda p:overwrite_json(p/'analysis/report.json',lambda o:o['decisions'].update(coefficient_confirmation=True))))
    check('table_tamper',lambda:fault('table',lambda p:(p/'analysis/curves.csv').write_text('forged\n')))
    check('accepted_evidence_mutation_rejected',rejection_after_commit)
    check('atomic_commit_no_overwrite',lambda:rejects(lambda:commit(BASE/'good/eval_128_0.npz',dict(x=np.array([1])))))
    extra_checks(sys.modules[__name__])
    check('kernel_soft_CPU_hard_CPU_wall_kill',kernel_limits)
    report=dict(synthetic_only=True,training_or_query_draws=0,tests=RESULTS,
        passed=all(r['passed'] for r in RESULTS),start=started,end=usage())
    dump(OUT/'resume_tests.json',report)
    print(json.dumps(report,ensure_ascii=False))
    if not report['passed']: sys.exit(1)

from resume_extra import observer_receipt, extra_checks
if __name__=='__main__': main()
