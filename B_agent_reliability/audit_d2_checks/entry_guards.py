"""离线入口审计：在 Engine 构造前截停，绝不创建真实客户端或真实账本。"""
import json
from pathlib import Path
import subprocess
import sys
from common import ROOT, dump_file, install_guards, resource_info

install_guards()
sys.path.insert(0,str(ROOT/'HOLDOUT_D_runner'))
import run
class StopBeforeEngine(Exception):pass
captures=[]
def stop_engine(out,**kw):
    captures.append(dict(out=str(out),**kw))
    raise StopBeforeEngine()
run.Engine=stop_engine

def cli(args):
    sys.argv=['run.py',*args]
    n=len(captures)
    try:run.main()
    except StopBeforeEngine:
        return dict(reached_engine_constructor=True,engine_options=captures[-1])
    except (SystemExit,ValueError) as ex:
        assert len(captures)==n
        return dict(reached_engine_constructor=False,exit=str(ex))
    raise AssertionError('unexpected execution')

launch=(ROOT/'HOLDOUT_D_runner/launch.sh').read_text()
start=launch.index('if [[');end=launch.index('\nfi',start)+len('\nfi')
shell_gate=launch[start:end]+'\nexit 0\n'
tests={}
for flag in ('--live','--liv','--li','--l'):
    args=[flag,'--quota-url','http://127.0.0.1:18319/AUDIT_NEVER_CONNECT']
    # 只执行启动脚本原样拒绝块；未执行 setsid/nohup/systemd-run 或写日志的后半段。
    p=subprocess.run(['bash','-c',shell_gate,'audit-launch',*args],capture_output=True,text=True)
    tests[flag]=dict(shell_rejected=p.returncode==2,shell_exit=p.returncode,cli=cli(args))
for label,args in {
    'default':[],
    'missing_quota':['--live'],
    'unfrozen_live':['--live','--test-unfrozen','--quota-url','http://127.0.0.1:18319/AUDIT_NEVER_CONNECT'],
    'alternate_ledger':['--live','--out',str(ROOT/'audit_d2_checks/fake_ledger'),'--quota-url','unused'],
    'unvalidated_provider_string':['--live','--quota-url','http://127.0.0.1:18319/AUDIT_NEVER_CONNECT'],
}.items():
    tests[label]=cli(args)
assert tests['--live']['shell_rejected']
assert all(not tests[f]['shell_rejected'] and tests[f]['cli']['engine_options']['mode']=='live' for f in ('--liv','--li','--l'))
dump_file('entry_guards.json',dict(resources=resource_info(),tests=tests,no_network=True,
    engine_constructed=False,real_ledger_written=False,network_clients_constructed=False))
print(json.dumps(tests,ensure_ascii=False))
