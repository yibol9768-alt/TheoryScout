"""封装全部确定性夹具；避免正式启动逐个打开数万份重复测试文件。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(9,10))
import json,subprocess,os
from common import ROOT,OUT,dump,sha,usage
names=['synthetic_y3c','synthetic_final','protocol_v1','protocol_v2','final_edges']
assert all((OUT/n).is_dir() for n in names)
assert json.loads((OUT/'protocol_v2/results.json').read_text())['passed']
assert json.loads((OUT/'final_edges/results.json').read_text())['passed']
path=OUT/'test_fixtures.tar'
# 排他占位；tar写入的是本次新文件，失败时不作为成功归档采信。
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
subprocess.run(['tar','--sort=name','--mtime=@0','--owner=0','--group=0','--numeric-owner',
    '-cf',str(path),*[str((OUT/n).relative_to(ROOT)) for n in names]],cwd=ROOT,check=True)
with path.open('rb') as f:os.fsync(f.fileno())
u=resource.getrusage(resource.RUSAGE_CHILDREN)
dump(OUT/'archive_status.json',dict(passed=True,directories=names,archive_sha256=sha(path),
    usage=usage(),cpu_seconds_children=u.ru_utime+u.ru_stime,no_formal_sampling=True))
print(json.dumps(dict(passed=True,usage=usage(),cpu_seconds_children=u.ru_utime+u.ru_stime)))
