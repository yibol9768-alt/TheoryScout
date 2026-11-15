"""冻结后只读核验，并测包含完整冻结扫描的实际观察者入口代码。"""
import ast
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'z4b'))
from common_v2 import OUT,sha,read,dump,usage

start=usage();manifest=ROOT/'Z4B_FROZEN.sha256'
t=time.monotonic()
p=subprocess.run([sys.executable,'-B',str(OUT/'entry.py'),'--verify-only'],cwd=ROOT,capture_output=True,text=True,timeout=300)
dump(ROOT/'Z4B_VERIFY.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
     wall=time.monotonic()-t,manifest_sha256=sha(manifest),formal_sampling=False))
assert p.returncode==0,p.stderr
# 运行原样 entry.py 导入/定时器/verify/--verify-accepted 路径；仅将 OUT 指向合成全规模夹具。
# 配置与包核验函数仍绑定真实包；verify_accepted 的 manifest 参数映射为人工夹具的 a*64。
# 这不会执行 --formal，也不会绕过任何正式科学接受路径；只计时测试夹具。
code='''import runpy,sys
sys.path.insert(0,sys.argv[1])
import common_v2 as C,protocol_v2 as P
realout=C.OUT;realverify=C.verify;realconfig=C.config;replay=P.verify_accepted
def verify():
    result=realverify()
    return result
def wrapped(data,manifest,cfg):return replay(realout/'synthetic_volume/full','a'*64,cfg)
P.verify_accepted=wrapped
sys.argv=[str(realout/'entry.py'),'--verify-accepted']
runpy.run_path(sys.argv[0],run_name='__main__')
print(C.usage())
'''
before=resource.getrusage(resource.RUSAGE_CHILDREN);t=time.monotonic()
q=subprocess.run([sys.executable,'-B','-c',code,str(OUT)],cwd=ROOT,capture_output=True,text=True,timeout=300)
observer_wall=time.monotonic()-t
after=resource.getrusage(resource.RUSAGE_CHILDREN)
assert q.returncode==0,(q.returncode,q.stderr)
cpu=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
assert cpu<39
baseline=read(OUT/'baseline.json');bad=[rel for rel,h in baseline.items() if sha(ROOT/rel)!=h]
frozen_bad=[rel for line in manifest.read_text().splitlines() for h,rel in [line.split('  ',1)] if sha(ROOT/rel)!=h]
assert not bad and not frozen_bad
assert not (OUT/'formal').exists() and not (ROOT/'z4/formal').exists()
scan=read(ROOT/'Z4B_FREEZE_USAGE.json');volume=read(OUT/'volume_result.json');res=read(OUT/'resources.json')
assert scan['manifest_sha256']==sha(manifest)
forecast_cpu=res['runner_forecast_cpu']+volume['report_cpu']+volume['publish_cpu']+cpu+3*scan['scan_cpu']
forecast_wall=res['runner_forecast_wall_at_cpu_per_wall_1_8']+volume['report_wall']+volume['publish_wall']+observer_wall+3*scan['scan_wall']
dump(ROOT/'Z4B_FINAL_CHECK.json',dict(passed=True,manifest_sha256=sha(manifest),frozen_entries=len(manifest.read_text().splitlines()),
     changed_original=bad,changed_frozen=frozen_bad,formal_absent=True,formal_sampling=False,
     observer_entry_timed_replay=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr,cpu_seconds=cpu,
          wall=observer_wall,timer_cpu=39,includes_actual_entry_imports_and_full_freeze_verify=True,
          caveat='真实entry --verify-accepted代码，数据地址和夹具manifest仅映射到人工full夹具；不是正式运行收据'),
     all_tests_passed=read(OUT/'all_tests.json')['passed'],start=start,end=usage()))
dump(ROOT/'Z4B_EXECUTION_FORECAST.json',dict(manifest_sha256=sha(manifest),
     future_formal_cpu_seconds_approx=forecast_cpu,future_formal_cpu_hours_approx=forecast_cpu/3600,
     future_formal_wall_seconds_approx=forecast_wall,future_formal_wall_hours_approx=forecast_wall/3600,
     basis='运行器保守外推 + 报告/发布实测 + 含全包冻结验证的实际entry夹具重放 + 其余三次冻结扫描；不含准备历史',
     caveat='当次IO和1.8 CPU/墙钟秒假设，不是完成或峰值保证；硬限优先，扫描非冷缓存上界'))
print(dict(passed=True,observer_cpu=cpu,forecast_cpu_hours=forecast_cpu/3600,forecast_wall_hours=forecast_wall/3600,manifest=sha(manifest)))
