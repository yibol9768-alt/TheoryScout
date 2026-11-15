"""完整规模常量 rank 夹具和真实外层 39 秒 CPU 定时器的重放计时。"""
import json
import os
import resource
import signal
import subprocess
import sys
import time
from common_v2 import OUT,ROOT,config,dump,sha,read,usage
import test_resume as T
from protocol_v2 import publish,verify_accepted
from analyze_v2 import make_report
from resume_extra import observer_receipt

start=usage();T.BASE=OUT/'synthetic_volume';T.BASE.mkdir(exist_ok=True)
reused=(T.BASE/'full').exists()
if reused:
    p=T.BASE/'full';c=config();m='a'*64;build_cpu=None
    archive=p/'pre_timing_acceptance';archive.mkdir()
    for name in ['ACCEPTED.json','launch_status.json','SUPERVISOR_EXIT.json','EXIT_OBSERVED.json']:
        (p/name).rename(archive/name)
else:
    t=time.process_time();p,c,m=T.fixture('full',config());build_cpu=time.process_time()-t
t=time.process_time();wall=time.monotonic();report=make_report(p,m,c);report_cpu=time.process_time()-t;report_wall=time.monotonic()-wall
assert report['data_errors']==[] and report['decisions']['primary_all3_directions']
assert len(report['availability'])==5377
assert all(r['n']==(1280 if r['target']==.425 else 64) for r in report['old_pairs'])
t=time.process_time();wall=time.monotonic();status=publish(p,m,c,T.GOOD,10.,10.,1.);publish_cpu=time.process_time()-t;publish_wall=time.monotonic()-wall
assert status['confirmation_valid'],status
observer_receipt(p,m)
t=time.process_time();wall=time.monotonic();assert verify_accepted(p,m,c)['confirmation_valid'];replay_cpu=time.process_time()-t;replay_wall=time.monotonic()-wall
# 与 entry 相同的导入、冻结核验前准备和完整接受重放代码路径；冻结后另测真实 entry。
code='''import signal,time,resource
signal.signal(signal.SIGPROF,signal.SIG_DFL);signal.setitimer(signal.ITIMER_PROF,39.)
from common_v2 import OUT,config,usage
from protocol_v2 import verify_accepted
assert verify_accepted(OUT/'synthetic_volume/full','a'*64,config())['confirmation_valid']
print(usage())
'''
wall=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
child=subprocess.run([sys.executable,'-B','-c',code],cwd=OUT,capture_output=True,text=True,timeout=300)
cold_wall=time.monotonic()-wall;after=resource.getrusage(resource.RUSAGE_CHILDREN)
assert child.returncode==0,(child.returncode,child.stderr)
dump(OUT/'volume_result.json',dict(passed=True,jobs=5377,npz_count=len(list(p.glob('*.npz'))),
     fixture_build_and_first_report_cpu=build_cpu,report_cpu=report_cpu,publish_cpu=publish_cpu,replay_cpu=replay_cpu,
     report_wall=report_wall,publish_wall=publish_wall,replay_wall=replay_wall,fixture_reused_after_timeout=reused,
     timed_cold_replay=dict(returncode=child.returncode,stdout=child.stdout,stderr=child.stderr,wall=cold_wall,
         cpu=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime),
     artifact_bytes=sum(q.stat().st_size for q in p.rglob('*') if q.is_file()),
     start=start,end=usage(),training_or_query_draws=0,
     caveat='全部 rank 人工固定，stage/资源值为夹具，ACCEPTED 非科学结果；未启动正式 worker；冷重放计时此时尚不含未来冻结全包读取，冻结后另测'))
print(read(OUT/'volume_result.json'))
