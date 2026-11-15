"""离线单入口：原版 v1/v2、v3 全测试及全部审计攻击回归。禁止覆盖旧运行结果。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'audit_regression'))
from common import resources

def main():
    start=time.time();res=resources();out=HERE/'regression_results'
    if any(out.glob('*_suite_results.json')):raise ValueError('use_a_fresh_copy_of_v3_for_repeat; preserve_existing_evidence')
    cases=['legacy_run_v1.py','legacy_run_suite.py','run_suite.py','attack_entries.py',
           'endpoint_escape.py','url_resolution.py','repeated_dispatch.py','attack_smoke.py','check_scores.py']
    results=[]
    # 子脚本自行安装所需网络钩子；不能预装 sitecustomize 的全禁网钩子，
    # 否则会连同旧合同要求的本地 HTTP mock 一并阻断。
    env=dict(os.environ)
    env.pop('PYTHONPATH',None)
    for name in cases:
        with (out/(name+'.log')).open('x') as log:
            p=subprocess.run([sys.executable,'-B',str(HERE/'audit_regression'/name)],env=env,stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(script=name,exit=p.returncode,log=str((out/(name+'.log')).relative_to(HERE))))
        print(name,p.returncode,flush=True)
    z=dict(passed=all(r['exit']==0 for r in results),seconds=time.time()-start,
           resources=res,results=results,no_real_llm=True,no_real_quota=True,
           network='仅旧版HTTP集成及v3独立测试客户端的127.0.0.1 mock；18319、外部、DNS均禁止')
    (out/'ALL_RESULTS.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')
    return 0 if z['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
