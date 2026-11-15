"""v4→v5 最小源码/路径变更及独立离线验收入口。"""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R=ROOT/'HOLDOUT_D_runner_v5'
def change(name,fn):
    p=R/name;p.write_text(fn(p.read_text()))
def main():
    for name in ('engine.py','evidence.py','run.py','score.py','launch.sh','test_mock.py','audit_regression/common.py'):
        change(name,lambda s:s.replace('HOLDOUT_D_runner_v4','HOLDOUT_D_runner_v5').replace('holdout-d-v4-','holdout-d-v5-'))
    for name in ('engine.py','evidence.py','audit_regression/check_scores.py'):
        # 旧清单仍须核验；新状态/人工门绑定 AMEND5。
        change(name,lambda s:s.replace("sha((ROOT/'HOLDOUT_D_AMEND4.sha256').read_bytes())","sha((ROOT/'HOLDOUT_D_AMEND5.sha256').read_bytes())").replace("digest((ROOT/'HOLDOUT_D_AMEND4.sha256').read_bytes())","digest((ROOT/'HOLDOUT_D_AMEND5.sha256').read_bytes())").replace("manifest = ROOT/'HOLDOUT_D_AMEND4.sha256'","manifest = ROOT/'HOLDOUT_D_AMEND5.sha256'").replace("manifest=ROOT/'HOLDOUT_D_AMEND4.sha256'","manifest=ROOT/'HOLDOUT_D_AMEND5.sha256'"))
    change('engine.py',lambda s:s.replace("    verify_manifest(ROOT/'HOLDOUT_D_AMEND4.sha256')","    verify_manifest(ROOT/'HOLDOUT_D_AMEND4.sha256')\n    verify_manifest(ROOT/'HOLDOUT_D_AMEND5.sha256')").replace("                good=a<.80 and b<.90","                now=self.clock()  # 查询完成后起算暂停，50秒超时不缩短300秒等待。\n                good=a<.80 and b<.90").replace("            except Exception:\n                self.wal.add('quota'","            except Exception:\n                now=self.clock()\n                self.wal.add('quota'"))
    change('score.py',lambda s:s.replace("'HOLDOUT_D_AMEND4.sha256'):","'HOLDOUT_D_AMEND4.sha256','HOLDOUT_D_AMEND5.sha256'):"))
    change('transport.py',lambda s:s.replace('import json, os, time, urllib.request, urllib.error','import json, os, subprocess, time, urllib.request, urllib.error').replace('\nclass MockServer:', '''
class CommandQuota:
    """固定项目命令；stdout 为五小时/周已用百分比，不接受自定义命令。"""
    def __init__(self,live=False):
        if live is not True:raise ValueError('quota_command_requires_--live')
        require_live_release(live)
    def __call__(self):
        require_live_release(True)
        z=subprocess.run(['python3','/home/liuyibo/auto-llm-driving/TheoryScout/tools/plus_quota.py'],
                         capture_output=True,text=True,timeout=50,check=True)
        a,b=map(float,z.stdout.split())
        if not (0<=a<=100 and 0<=b<=100):raise ValueError('quota_command_percent_out_of_range')
        return dict(five_hour=a/100,week=b/100,observed_unix=time.time())

class MockServer:'''))
    change('run.py',lambda s:s.replace('HTTPTransport, HTTPQuota, MockServer','HTTPTransport, HTTPQuota, CommandQuota, MockServer').replace("    ap.add_argument('--quota-url');ap.add_argument('--smoke-approval',type=Path)","    quotas=ap.add_mutually_exclusive_group()\n    quotas.add_argument('--quota-url')\n    quotas.add_argument('--quota-command',action='store_true',help='使用固定项目 plus_quota.py 命令查询额度；须 --live 和有效放行文件')\n    ap.add_argument('--smoke-approval',type=Path)").replace("    if z.live:require_live_release(z.live)","    if z.quota_command and not z.live:raise SystemExit('quota_command_requires_--live')\n    if z.live:require_live_release(z.live)").replace('if z.live and not z.quota_url:','if z.live and not (z.quota_url or z.quota_command):').replace('probe=HTTPQuota(z.quota_url,live=True) if z.live else lambda:dict(five_hour=0.,week=0.)','probe=(CommandQuota(live=True) if z.quota_command else HTTPQuota(z.quota_url,live=True)) if z.live else lambda:dict(five_hour=0.,week=0.)'))
    for name in ('HOLDOUT_D_cli_sentinel.py','audit_regression/entry_sentinel.py'):
        change(name,lambda s:s.replace('run.Engine=run.HTTPTransport=run.HTTPQuota=forbidden','run.Engine=run.HTTPTransport=run.HTTPQuota=run.CommandQuota=forbidden'))
    checks=R/'amend5_checks';checks.mkdir(exist_ok=True)
    for name in ('harness.py','sitecustomize.py','boundary_tests.py','resume_tests.py','run_all.py'):
        source=(R/'amend4_checks'/name).read_text()
        if name in ('harness.py','boundary_tests.py'):source=source.replace('HOLDOUT_D_runner_v4','HOLDOUT_D_runner_v5')
        if name=='harness.py':
            source=source.replace("('v3','v4')","('v3','v4','v5')").replace("OUT/'v4_tests'","OUT/'v5_tests'")
        if name=='run_all.py':
            source=source.replace("jobs=[('boundary'","jobs=[('command_quota',['command_quota_tests.py']),('boundary'")
            source=source.replace("('v4','v1','v2','v3')","('v5','v4','v1','v2','v3')")
            source=source.replace('先验收 M1，再运行 v4 继承测试、原版 v1/v2/v3 和攻击回归','先验收命令额度与 M1，再运行 v5/v4/v1/v2/v3 及攻击回归')
        if name=='sitecustomize.py':
            source=source.replace('D4:','D5:').replace('D4_','D5_')
            source=source.replace("    if event in ('socket.connect'", "    if event=='subprocess.Popen':\n        executable,argv,cwd,env=args\n        quota='/home/liuyibo/auto-llm-driving/TheoryScout/tools/plus_quota.py'\n        if quota in str(argv):\n            import shutil\n            resolved=shutil.which(os.fsdecode(executable),path=(env or os.environ).get('PATH'))\n            if resolved is None or Path(resolved).resolve()!=OUT/'command_bin/python3':\n                raise RuntimeError('D5_REAL_QUOTA_EXECUTION_BLOCKED')\n    if event in ('socket.connect'")
        (checks/name).write_text(source)
    (checks/'results').mkdir()
    change('validate_all.py',lambda s:s.replace("checks=HERE/'amend4_checks'","checks=HERE/'amend5_checks'"))
    print('v5 minimal edits complete')
if __name__=='__main__':main()
