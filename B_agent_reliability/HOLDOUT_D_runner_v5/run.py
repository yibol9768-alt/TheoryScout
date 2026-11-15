#!/usr/bin/env python3
"""D runner: 默认离线；--live 仍需额度适配和后续明确授权。"""
import argparse, json, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from engine import Engine, Paused, allowed_dir
from protocol import ROOT, dump
from evidence import require_live_release
from transport import HTTPTransport, HTTPQuota, CommandQuota, MockServer

def environment():
    cgroup=Path('/proc/self/cgroup').read_text().strip()
    rel=cgroup.split('0::')[-1].strip();cg=Path('/sys/fs/cgroup')/rel.lstrip('/')
    z=dict(pid=os.getpid(),sid=os.getsid(0),ppid=os.getppid(),stdin=os.readlink('/proc/self/fd/0'),
           tty=os.isatty(0),cgroup=cgroup,memory_max=(cg/'memory.max').read_text().strip(),
           swap_max=(cg/'memory.swap.max').read_text().strip(),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    if str(z['sid'])==os.environ.get('HOLDOUT_D_PARENT_SID') or not os.environ.get('HOLDOUT_D_PARENT_SID'):raise ValueError('not_independent_session')
    z['original_sid']=int(os.environ['HOLDOUT_D_PARENT_SID'])
    if z['tty'] or z['stdin']!='/dev/null' or z['memory_max']!='2147483648' or z['swap_max']!='0':raise ValueError('launch_contract_failed')
    if any(v not in ('1','2') for v in z['threads'].values()):raise ValueError('thread_contract_failed')
    return z

def execute(engine,transport,probe,*,mock=False,approval=None):
    if engine.mode=='live' and mock:raise ValueError('mock_scheduler_forbidden_live')
    if approval:engine.approve_smoke(approval)
    def task_loop(seed):
        while not engine.tasks[seed].terminal:
            if engine.halt:return
            try:engine.request(seed,transport,probe)
            except Paused as e:
                if engine.halt:return
                # Main loop never sleeps longer than 1 second; deadline is global and persistent.
                time.sleep(1)
    # Exactly two Python execution threads total: main thread + one worker.
    with ThreadPoolExecutor(max_workers=1) as pool:
        for first in range(116064001,116064145,2):
            if first>116064001 and not mock and not engine.smoke:
                engine.wal.add('smoke_wait',reason='first_two_in_N_need_human_review');break
            if engine.halt:break
            future=pool.submit(task_loop,first);task_loop(first+1);future.result()
    if all(engine.tasks[s].terminal for s in (116064001,116064002)) and not engine.smoke:
        p=engine.wal.out/'HOLDOUT_D_SMOKE_REVIEW_TEMPLATE.json'
        if not p.exists():
            with p.open('x') as f:f.write(dump(engine.smoke_review_template())+'\n')
    return engine.summary()

def main():
    ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--live',action='store_true',help='显式允许联网；本次交付禁止使用')
    ap.add_argument('--out',type=Path)
    quotas=ap.add_mutually_exclusive_group()
    quotas.add_argument('--quota-url')
    quotas.add_argument('--quota-command',action='store_true',help='使用固定项目 plus_quota.py 命令查询额度；须 --live 和有效放行文件')
    ap.add_argument('--smoke-approval',type=Path)
    ap.add_argument('--environment-only',action='store_true');ap.add_argument('--test-unfrozen',action='store_true',help=argparse.SUPPRESS)
    ap.add_argument('--entry-check',action='store_true',help=argparse.SUPPRESS)
    z=ap.parse_args()
    if z.quota_command and not z.live:raise SystemExit('quota_command_requires_--live')
    if z.live:require_live_release(z.live)
    if z.entry_check:return
    live_out=ROOT/'HOLDOUT_D_runner_v5/HOLDOUT_D_live'
    out=allowed_dir(z.out or (live_out if z.live else ROOT/'HOLDOUT_D_runner_v5/HOLDOUT_D_mock_run'))
    if z.live and out!=live_out:raise SystemExit('live_ledger_and_flock_path_is_fixed_no_budget_reset')
    if not z.live and out==live_out:raise SystemExit('mock_cannot_write_live_ledger')
    if z.live and (z.test_unfrozen or 'mock' in out.name.lower()):raise SystemExit('live_requires_frozen_distinct_output')
    if z.live and not (z.quota_url or z.quota_command):raise SystemExit('missing_qualified_project_quota_provider; stay_paused')
    engine=Engine(out,verify=not z.test_unfrozen,mode='live' if z.live else 'mock',live=z.live)
    try:
        if z.live and any(e['kind']=='init' and e['mode']!='live' for e in engine.wal.events):raise SystemExit('fixture_forbidden_live')
        launch=environment();engine.wal.add('launch',environment=launch,live=z.live)
        if z.environment_only:
            p=out/'HOLDOUT_D_LAUNCH_ENV.json'
            with p.open('x') as f:f.write(dump(dict(**launch,flock_held=True,no_network=True))+'\n')
            return
        transport=HTTPTransport(live=True) if z.live else MockServer(policy='fatal_at_8')
        probe=(CommandQuota(live=True) if z.quota_command else HTTPQuota(z.quota_url,live=True)) if z.live else lambda:dict(five_hour=0.,week=0.)
        approval=None
        if z.smoke_approval:
            p=z.smoke_approval.resolve()
            if not p.is_relative_to(ROOT/'HOLDOUT_D_runner_v5'):raise SystemExit('approval_must_be_inside_runner')
            approval=json.loads(p.read_text())
        print(dump(execute(engine,transport,probe,mock=not z.live,approval=approval)),flush=True)
    finally:engine.close()
if __name__=='__main__':main()
