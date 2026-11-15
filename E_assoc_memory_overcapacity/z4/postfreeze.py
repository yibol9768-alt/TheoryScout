"""冻结之后只读入口验证；收据新建于清单之外，避免自指。无正式模式。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(21600,21600))
import signal
signal.signal(signal.SIGPROF,signal.SIG_DFL)
signal.setitimer(signal.ITIMER_PROF,9.)
import subprocess
import sys
from common_v2 import ROOT,OUT,sha,dump,usage

assert not (OUT/'formal').exists()
p=subprocess.run([sys.executable,'-B',str(OUT/'entry.py'),'--verify-only'],
                 capture_output=True,text=True,timeout=30)
children=resource.getrusage(resource.RUSAGE_CHILDREN)
result=dict(passed=p.returncode==0,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
    manifest_sha256=sha(ROOT/'Z4_FROZEN.sha256'),formal_sampling=False,
    formal_directory_absent=not (OUT/'formal').exists(),usage=usage(),
    child_cpu=children.ru_utime+children.ru_stime,
    note='Post-freeze verification receipt intentionally excluded from its own manifest.')
dump(ROOT/'Z4_VERIFY.json',result)
print(result)
assert result['passed'] and result['formal_directory_absent']
