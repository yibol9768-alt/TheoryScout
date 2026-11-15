"""全离线回归：先验收命令额度与 M1，再运行 v5/v4/v1/v2/v3 及攻击回归。"""
import json,subprocess,sys,time
from pathlib import Path
from harness import OUT,ROOT,resources

def main():
    start=time.time();res=resources()
    jobs=[('command_quota',['command_quota_tests.py']),('boundary',['boundary_tests.py']),('resume',['resume_tests.py'])]
    jobs += [('suite_'+v,['harness.py','suite',v]) for v in ('v5','v4','v1','v2','v3')]
    jobs += [('ported_'+n[:-3],['harness.py','ported',n]) for n in
             ('attack_entries.py','endpoint_escape.py','url_resolution.py','repeated_dispatch.py','attack_smoke.py','check_scores.py')]
    results=[];here=Path(__file__).resolve().parent
    for label,args in jobs:
        print('START',label,flush=True)
        with (OUT/(label+'.log')).open('x') as log:
            p=subprocess.run([sys.executable,'-B',str(here/args[0]),*args[1:]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(job=label,exit=p.returncode,log=label+'.log'))
        print('DONE',label,p.returncode,flush=True)
        (OUT/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    z=dict(passed=all(x['exit']==0 for x in results),results=results,seconds=time.time()-start,resources=res,
           no_real_llm=True,no_real_quota=True,network='所有发送为本地记录桩；socket连接、DNS、sendto、bind均拒绝')
    (OUT/'ALL_RESULTS.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(z,ensure_ascii=False),flush=True)
    return 0 if z['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
