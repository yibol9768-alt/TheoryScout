"""唯一将来正式入口；外层观察者收割监管器，硬杀不产生正常退出凭据。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU, (21600, 21600))
import signal
signal.signal(signal.SIGPROF, signal.SIG_DFL)
signal.setitimer(signal.ITIMER_PROF, 39.)
import time
START = time.monotonic()
signal.signal(signal.SIGALRM, signal.SIG_DFL)
signal.alarm(12600)
import os
import subprocess
import sys
from common_v2 import OUT, config, verify, read, dump, sha, usage
from protocol_v2 import verify_accepted

def main():
    mode = sys.argv[1:]
    if mode not in [['--verify-only'], ['--verify-accepted'], ['--formal']]:
        raise ValueError('explicit mode required')
    cfg = config(); mh = verify()
    if mode == ['--verify-only']:
        if (OUT/'formal').exists(): raise ValueError('formal already exists')
        print(dict(verified=True, manifest=mh, formal_sampling=False)); return
    if mode == ['--verify-accepted']:
        print(dict(accepted=verify_accepted(OUT/'formal',mh,cfg)['confirmation_valid'])); return
    if (OUT/'formal').exists(): raise FileExistsError('permanent single attempt')
    env = dict(os.environ, Z4B_OBSERVER_PID=str(os.getpid()))
    proc = subprocess.Popen([sys.executable,'-B',str(OUT/'launch_v2.py'),'--formal'],
                            env=env, cwd=OUT)
    killed = False
    try: rc = proc.wait(timeout=max(.01,12600-5-(time.monotonic()-START)))
    except subprocess.TimeoutExpired:
        killed = True; proc.kill(); rc = proc.wait()
    data = OUT/'formal'
    if not data.is_dir(): raise RuntimeError('supervisor failed before launch')
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = children.ru_utime+children.ru_stime+usage()['cpu']
    dump(data/'EXIT_OBSERVED.json',dict(returncode=rc,killed=killed,manifest=mh,
        wall=time.monotonic()-START,cpu=cpu,self_cpu=usage()['cpu'],
        launch_status_sha256=sha(data/'launch_status.json') if (data/'launch_status.json').is_file() else None,
        receipt_sha256=sha(data/'SUPERVISOR_EXIT.json') if (data/'SUPERVISOR_EXIT.json').is_file() else None))
    if rc or killed: raise RuntimeError('supervisor abnormal exit; no confirmation')
    accepted=verify_accepted(data,mh,cfg)
    print(dict(confirmation_valid=accepted['confirmation_valid'],decisions=accepted['decisions']))

if __name__ == '__main__': main()
