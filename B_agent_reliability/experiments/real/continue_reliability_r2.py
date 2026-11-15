"""One-shot orchestration; frozen experimental code handles all model calls."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent

def command(script,*args,log):
    with (ROOT/log).open('x') as f:
        p=subprocess.run([sys.executable,'-u',str(ROOT/script),*args],stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:
        raise SystemExit(f'{script} exited {p.returncode}; see {log}')

def main():
    with (ROOT/'SUPERVISOR_R2_STARTED.json').open('x') as f:
        f.write(json.dumps(dict(note='single orchestrator; no extra model budget'))+'\n')
    while True:
        s=json.loads((ROOT/'difficulty_r2_real/summary.json').read_text())
        if s.get('finished_utc'):
            break
        time.sleep(20)
    command('report_reliability_r2.py','r2',log='AUDIT_REPORT_DIFFICULTY_R2.log')
    command('audit_reliability_r2.py','r2',log='AUDIT_DIFFICULTY_R2.log')
    if s['halt']:
        blocked=dict(reason=s['halt'],r2_calls=s['calls'],calib_calls=0,not_started=True)
        (ROOT/'CALIB1_NOT_STARTED.json').write_text(json.dumps(blocked,indent=2)+'\n')
        with (ROOT/'CALIB1.md').open('x') as f:
            f.write('# CALIB1 结果\n\n不能进入容量/长度留出实验。R2触发全局停止，按预注册不得启动校准。\n\n'
                    f"R2停止原因：`{s['halt']}`；校准真实调用0次、计划新题未执行。\n\n"
                    '预注册完整性、目标率、30错误片段、修复与迁移门槛均未取得校准证据；不降低门槛、不补零。\n')
        print(json.dumps(blocked),flush=True)
        return
    command('reliability_r2.py','calib1',log='RUN_CALIB1.log')
    command('report_reliability_r2.py','calib1',log='AUDIT_REPORT_CALIB1.log')
    command('audit_reliability_r2.py','calib1',log='AUDIT_CALIB1.log')
    print('BOTH_PHASES_FINISHED',flush=True)

if __name__=='__main__':main()
