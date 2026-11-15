"""Sole future formal entry. This task only invokes --verify-only."""
import os
import resource
import secrets
import signal
import subprocess
import sys
import time
from common import OUT, config, verify, dump, sha, usage
from protocol import digest, publish, verify_accepted

def main():
    if sys.argv[1:] not in [['--verify-only'],['--formal'],['--verify-accepted']]: raise ValueError('explicit mode required')
    cfg=config();mh=verify()
    if sys.argv[1:]==['--verify-only']:
        print(dict(verified=True,manifest=mh,formal_sampling=False));return
    if sys.argv[1:]==['--verify-accepted']:
        print(dict(accepted=verify_accepted(OUT/'formal',mh,cfg)['confirmation_valid']));return
    start=time.monotonic()
    signal.signal(signal.SIGPROF,signal.SIG_DFL);signal.setitimer(signal.ITIMER_PROF,99.)
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.alarm(cfg['total_wall_hard'])
    data=OUT/'formal';data.mkdir() # permanent single attempt, including failed starts
    payloads={stage:dict(stage=stage,manifest=mh,supervisor_pid=os.getpid(),nonce=secrets.token_hex(32))
              for stage in ['runner','analysis']}
    dump(data/'launch.json',dict(manifest=mh,pid=os.getpid(),credentials={k:digest(v) for k,v in payloads.items()},
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),config=cfg))
    states=[]
    for stage,script,limit,deadline in [('runner','run.py',cfg['runner_cpu_hard'],start+cfg['runner_wall_hard']),
                                      ('analysis','analyze.py',cfg['analysis_cpu_hard'],start+12560)]:
        if stage=='analysis': deadline=min(deadline,time.monotonic()+cfg['analysis_wall_hard'])
        fd_r,fd_w=os.pipe()
        env=dict(os.environ,Z4_CREDENTIAL_FD=str(fd_r))
        # Set CPU limit before importing scientific libraries in the worker.
        code='import resource,runpy,sys;resource.setrlimit(resource.RLIMIT_CPU,('+str(limit)+','+str(limit)+'));sys.argv=['+repr(str(OUT/script))+',"--credential-worker"];runpy.run_path(sys.argv[0],run_name="__main__")'
        with (data/(stage+'.stdout')).open('x') as out,(data/(stage+'.stderr')).open('x') as err:
            proc=subprocess.Popen([sys.executable,'-B','-c',code],cwd=OUT,env=env,pass_fds=(fd_r,),
                stdout=out,stderr=err,start_new_session=True)
            os.close(fd_r)
            import json
            try:
                with os.fdopen(fd_w,'w') as f: json.dump(payloads[stage],f)
            except BrokenPipeError: pass
            killed=False
            try: rc=proc.wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                killed=True;os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
            states.append(dict(stage=stage,returncode=rc,killed=killed))
    u=resource.getrusage(resource.RUSAGE_CHILDREN)
    status=publish(data,mh,cfg,states,time.monotonic()-start,u.ru_utime+u.ru_stime,usage()['cpu'],
        budget_probe=lambda:(time.monotonic()-start,usage()['cpu']))
    # Receipt only survives a normally exiting supervisor; re-verification checks it too.
    dump(data/'SUPERVISOR_EXIT.json',dict(launch_status_sha256=sha(data/'launch_status.json'),normal=True,
        wall=time.monotonic()-start,cpu=usage()['cpu'],manifest=mh))
    print(dict(confirmation_valid=status['confirmation_valid']))

if __name__=='__main__': main()
