"""真实子进程在原分析main发布FINAL后遭SIGKILL的故障注入。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,21600))
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit_y3b_checks'
fixture=OUT/'analysis_killed_actual';data=fixture/'formal';data.mkdir(parents=True)
for p in (OUT/'target_fixture').glob('*.npz'):os.link(p,data/p.name)
for name in ['run_exit.json','runner_status.json','events.jsonl']:
    with (data/name).open('xb') as f:f.write((OUT/'state_control_normal/formal'/name).read_bytes())
code="""
import os,signal,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import analyze,common
common.environment()
analyze.OUT=Path(sys.argv[2])
analyze.verify=lambda:common.sha(common.ROOT/'Y3B_FROZEN.sha256')
sys.argv=['analyze.py','--formal-via-launcher']
os.environ['Y3B_LAUNCHER']='1'
analyze.main()
os.kill(os.getpid(),signal.SIGKILL)
"""
with (fixture/'child.log').open('x') as log:
    p=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'y3b'),str(fixture)],
                     cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=30)
result=json.loads((data/'analysis/FINAL.json').read_text())
assert p.returncode==-9 and result['confirmation_valid'] and result['d512_passed']
state=dict(analysis_returncode=p.returncode,FINAL_exists=True,
           confirmation_valid=result['confirmation_valid'],d512_passed=result['d512_passed'],
           injection='原分析main返回后立即SIGKILL；真实收集子进程returncode，未执行正式启动器')
with (OUT/'analysis_exit_injection.json').open('x') as f:json.dump(state,f,indent=2,ensure_ascii=False)
print(json.dumps(state,ensure_ascii=False))
