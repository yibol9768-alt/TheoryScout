"""正式两阶段总预算监管；唯一命令入口，不允许恢复/重跑覆盖。"""
import resource
# 保留子进程可下调的RLIMIT硬限额；监管进程自身用内核profiling CPU计时器兜底。
resource.setrlimit(resource.RLIMIT_CPU,(21600,21600))
import signal
signal.signal(signal.SIGPROF,signal.SIG_DFL)
signal.setitimer(signal.ITIMER_PROF,9.)
import json
import os
from pathlib import Path
import subprocess
import sys
import time
START=time.monotonic()
signal.signal(signal.SIGALRM,signal.SIG_DFL)
signal.setitimer(signal.ITIMER_REAL,12600.)
from common import ROOT,OUT,verify,config,dump,usage
from publication import finalize, verify_accepted

def main():
    assert sys.argv[1:] in [['--formal-after-review'],['--verify-only'],['--verify-accepted']], '必须在人工审阅后显式启动'
    mh=verify();cfg=config()
    if sys.argv[1:]==['--verify-only']:
        print(json.dumps(dict(manifest_sha256=mh,verified=True,formal_sampling=False)))
        return
    if sys.argv[1:]==['--verify-accepted']:
        result=verify_accepted(OUT/'formal',mh,cfg)
        print(json.dumps(dict(verified=True,confirmation_valid=result['confirmation_valid'])))
        return
    assert cfg['runner_cpu_hard']==21340 and cfg['analysis_cpu_hard']==50 and cfg['total_cpu_hard']==21600
    data=OUT/'formal';data.mkdir()
    dump(data/'launch.json',dict(schema=1,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        manifest_sha256=mh,cpu_hard_split=[21340,50,10],preparation_cpu_reserve=200,
        wall_hard_total=12600,usage=usage()))
    env=dict(os.environ,Y3C_LAUNCHER='1')
    states=[]
    for script,deadline in [('run.py',START+12525),('analyze.py',START+12595)]:
        if script=='analyze.py': deadline=min(deadline,time.monotonic()+60)
        with (data/(script+'.stdout')).open('x') as stdout, (data/(script+'.stderr')).open('x') as stderr:
            proc=subprocess.Popen([sys.executable,'-B',str(OUT/script),'--formal-via-launcher'],
                cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            timed_out=False
            try: code=proc.wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out=True;proc.kill();code=proc.wait()
            state=dict(program=script,returncode=code,wall_hard_killed=timed_out)
            states.append(state)
            if script=='run.py': dump(data/'run_exit.json',state)
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    accepted=finalize(data,mh,cfg,dict(stages=states,wall_seconds=time.monotonic()-START,
        cpu_seconds_children=children.ru_utime+children.ru_stime,supervisor_usage=usage(),
        analysis_pending_exists=(data/'analysis/PENDING.json').is_file()),
        budget_probe=lambda:dict(wall_seconds=time.monotonic()-START,supervisor_usage=usage()))
    print(json.dumps(dict(stages=states,accepted=accepted)))

if __name__=='__main__': main()
