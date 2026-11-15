"""完整计划大小的伪 rank 夹具：只检查持久化/分析/采信成本，不抽样。"""
import json
import sys
import time
import resource
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'audit_z4_checks'))
from checks import C,R,A,P,np,DEST,save

def main():
    start=time.process_time()
    p=subprocess.run([sys.executable,'-B',str(ROOT/'z4/entry.py'),'--verify-only'],
                     cwd=ROOT,capture_output=True,text=True,timeout=30)
    save('official_verify.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    assert p.returncode==0
    data=DEST/'full_volume';data.mkdir()
    cfg=C.config();mh=C.sha(ROOT/'Z4_FROZEN.sha256')
    creds={'runner':'b'*64,'analysis':'c'*64}
    C.dump(data/'launch.json',dict(manifest=mh,config=cfg,credentials=creds))
    C.dump(data/'start.json',dict(manifest=mh,config=cfg,credential=creds['runner'],
                                streams=C.sha(ROOT/'z4/streams.json')))
    ev=P.Events(data/'events.jsonl',mh,creds['runner']);done=[]
    for job in C.plan(cfg):
        ev.append('started',job=job);path=data/C.jobname(job)
        if job[0]=='cal':
            _,d,m,s=job;Q=cfg['q_cal'];rr=np.full((17,Q),d*d//4,dtype=np.int64)
            for k in range(17):rr[k,:round(.7*k/16*Q)]=1
            C.commit(path,dict(ids=np.arange(Q,dtype=np.int64),ranks=rr,
                metadata=np.array(json.dumps(dict(job=job,manifest=mh)))))
        elif job==['lock']:
            lock=R.lock_alphas(data,cfg,mh);C.dump(path,lock)
        else:
            _,d,s=job;Q=cfg['q_eval'];rr=np.full((2,3,4,2,Q),d*d//4,dtype=np.int64)
            aa=np.zeros((2,3,4));valid=np.ones((2,3,4),dtype=bool)
            for a,arm in enumerate(C.ARMS):
                for ti,target in enumerate(cfg['targets']):
                    for mi,method in enumerate(C.METHODS):
                        cell=next(c for c in lock['cells'] if (c['d'],c['method'],c['target'])==(d,method,target))
                        aa[a,ti,mi]=cell['empirical']['alpha'] if a==0 else cell['analytic']
                        rr[a,ti,mi,0,:round((.8-mi*.1)*Q)]=1
                        rr[a,ti,mi,1,:round(target*Q)]=1
            C.commit(path,dict(ids=np.arange(Q,dtype=np.int64),ranks=rr,alphas=aa,valid=valid,
                metadata=np.array(json.dumps(dict(job=job,manifest=mh,lock_sha256=C.sha(data/'lock.json'))))))
        done.append(job);ev.append('complete',job=job,sha256=C.sha(path))
    ev.append('stopped',reason='planned_complete',active=None,error=None);ev.close()
    C.dump(data/'runner_status.json',dict(reason='planned_complete',active=None,error=None,
           completed=done,event_tail=ev.prev,usage=C.usage(),wall=1.))
    t=time.process_time();report=A.make_report(data,mh,cfg);report_cpu=time.process_time()-t
    assert report['data_errors']==[]
    dest=data/'analysis';dest.mkdir();C.dump(dest/'report.json',report)
    files={'report.json':C.sha(dest/'report.json')}
    for name,key in [('curves.csv','curves'),('matching.csv','matching'),('old_pairs.csv','old_pairs'),
                     ('scaled_residuals.csv','scaled_residuals'),('availability.csv','availability'),('all_completed.csv','all_completed')]:
        with (dest/name).open('x') as f:f.write(A.csv_text(report[key]))
        files[name]=C.sha(dest/name)
    C.dump(dest/'PENDING.json',dict(manifest=mh,credential=creds['analysis'],files=files,
                                 usage=C.usage(),confirmation_valid=False))
    states=[dict(stage='runner',returncode=0,killed=False),dict(stage='analysis',returncode=0,killed=False)]
    t=time.process_time();status=P.publish(data,mh,cfg,states,10.,10.,1.);publish_cpu=time.process_time()-t
    assert status['confirmation_valid'],status
    C.dump(data/'SUPERVISOR_EXIT.json',dict(normal=True,manifest=mh,launch_status_sha256=C.sha(data/'launch_status.json')))
    C.dump(data/'EXIT_OBSERVED.json',dict(returncode=0,killed=False,manifest=mh,wall=10.,cpu=12.,self_cpu=1.,
          launch_status_sha256=C.sha(data/'launch_status.json'),receipt_sha256=C.sha(data/'SUPERVISOR_EXIT.json')))
    t=time.process_time();assert P.verify_accepted(data,mh,cfg)['confirmation_valid'];replay_cpu=time.process_time()-t
    bad=[]
    for line in (ROOT/'Z4_FROZEN.sha256').read_text().splitlines():
        h,rel=line.split('  ',1)
        if C.sha(ROOT/rel)!=h:bad.append(rel)
    save('volume_result.json',dict(jobs=len(done),npz_count=len(list(data.glob('*.npz'))),
         report_cpu=report_cpu,publish_cpu=publish_cpu,replay_cpu=replay_cpu,
         artifact_bytes=sum(f.stat().st_size for f in data.rglob('*') if f.is_file()),
         cpu_seconds=time.process_time()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
         frozen_mismatches=bad,formal_absent=not (ROOT/'z4/formal').exists(),formal_sampling=False,
         note='所有 rank 均为人为常量；此 ACCEPTED 只属于 audit_z4_checks/full_volume，不是科学结果。stage/资源字段为夹具值，未启动正式执行链。'))
    print(json.dumps(C.read(DEST/'volume_result.json'),ensure_ascii=False))

if __name__=='__main__':main()
