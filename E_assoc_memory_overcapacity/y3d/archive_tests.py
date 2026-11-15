"""归档本轮确定性重跑，历史副本保留。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,25))
import os,subprocess,json
from common import ROOT,OUT,sha,dump,usage
names=['synthetic_y3d','synthetic_rerun','protocol_rerun','final_rerun','audit_rerun','r3_rerun']
assert all((OUT/n).is_dir() for n in names)
for n in ['protocol_rerun','final_rerun','r3_rerun']:assert json.loads((OUT/n/'results.json').read_text())['passed']
path=OUT/'Y3D_test_fixtures.tar'
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
subprocess.run(['tar','--sort=name','--mtime=@0','--owner=0','--group=0','--numeric-owner','-cf',str(path),
                *[str((OUT/n).relative_to(ROOT)) for n in names]],cwd=ROOT,check=True)
with path.open('rb') as f:os.fsync(f.fileno())
u=resource.getrusage(resource.RUSAGE_CHILDREN)
dump(OUT/'Y3D_archive_status.json',dict(passed=True,directories=names,archive_sha256=sha(path),usage=usage(),
     cpu_seconds_children=u.ru_utime+u.ru_stime,no_formal_sampling=True))
print(json.dumps(dict(passed=True,usage=usage(),cpu_seconds_children=u.ru_utime+u.ru_stime)))
