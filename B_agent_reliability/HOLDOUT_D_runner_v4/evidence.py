"""入口及人工凭据的共同离线合同。模块不构造任何网络对象。"""
import json
from datetime import datetime, timezone
from pathlib import Path
from protocol import ROOT, dump, sha

RUNNER = ROOT/'HOLDOUT_D_runner_v4'
RELEASE = RUNNER/'HOLDOUT_D_LIVE_RELEASE.json'
FIRST = (116064001, 116064002)
CHECKS = ('prompt_contract', 'actual_error_writes', 'two_review_protocol',
          'terminal_outcomes', 'model_usage_guards', 'quota_budget', 'wal_replay')
REVIEW_DECLARATION='我已亲自审阅此快照绑定的首两题原始记录，并亲手填写本凭据。'
VERSION_FILES = ('engine.py', 'protocol.py', 'transport.py', 'run.py', 'launch.sh',
                 'evidence.py', 'score.py')

def version():
    files = {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in
             [*(RUNNER/n for n in VERSION_FILES), ROOT/'HOLDOUT_D_DESIGN.json',
              ROOT/'HOLDOUT_D_PREREG.sha256', ROOT/'HOLDOUT_D_AMEND1.sha256']}
    return sha(dump(files).encode())

def timestamp(value):
    if type(value) is not str:raise ValueError('timestamp_type')
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:raise ValueError('timestamp_needs_timezone')
    return dt.timestamp()

def require_live_release(explicit_live):
    # 此指定文件在本次离线交付中不存在；环境变量或其他路径不能替代。
    if explicit_live is not True:raise ValueError('live_requires_explicit_--live')
    if not RELEASE.is_file() or RELEASE.is_symlink():raise ValueError('live_release_file_missing')
    z = json.loads(RELEASE.read_text())
    keys = {'schema','explicit_user_authorization','provider_evidence','authorized_by',
            'authorized_at','runner_version','amend3_manifest_sha256'}
    if type(z) is not dict or set(z)!=keys:raise ValueError('live_release_schema')
    if z['schema']!='HOLDOUT_D_AMEND3_LIVE_RELEASE_V1' or z['explicit_user_authorization'] is not True:
        raise ValueError('live_release_authorization')
    if any(type(z[k]) is not str or not z[k].strip() for k in ('authorized_by','provider_evidence')):
        raise ValueError('live_release_human_and_provider_required')
    if timestamp(z['authorized_at'])>datetime.now(timezone.utc).timestamp():raise ValueError('future_release')
    if z['runner_version']!=version():raise ValueError('live_release_version')
    manifest = ROOT/'HOLDOUT_D_AMEND4.sha256'
    if not manifest.is_file() or z['amend3_manifest_sha256']!=sha(manifest.read_bytes()):
        raise ValueError('live_release_freeze')
    return z

def smoke_snapshot(events, runner_version):
    """唯一指定快照：首两题最后一个终局 action 的链 head，覆盖此前全部字节。"""
    terminals = {}
    for e in events:
        if e['kind']=='action':
            r=e['record'];s=r['seed']
            if s in FIRST and ((r['correct'] and r['step']==63) or (not r['correct'] and r['review']==2)):
                if s in terminals:raise ValueError('duplicate_smoke_terminal')
                terminals[s] = dict(seed=s, c=r['step']+int(r['correct']),
                                    status='success' if r['correct'] else 'unrepaired_after_2_reviews',
                                    terminal_seq=e['seq'], terminal_digest=e['digest'])
    if set(terminals)!=set(FIRST):raise ValueError('smoke_not_finished')
    end=max(t['terminal_seq'] for t in terminals.values())
    prefix=events[:end]
    if any(e['kind']=='reserve' and e['seed'] not in FIRST for e in prefix):raise ValueError('premature_third_task')
    if events[0].get('runner_version')!=runner_version:raise ValueError('smoke_version_mismatch')
    return dict(schema='HOLDOUT_D_AMEND3_SMOKE_SNAPSHOT_V1', runner_version=runner_version,
                amend3_manifest_sha256=events[0].get('amend3_manifest_sha256'),
                wal_seq=end, wal_digest=prefix[-1]['digest'],
                # 不仅覆盖终局，两题的 reserve/response/action 以及中间事件全部入 hash。
                wal_sha256=sha((''.join(dump(e)+'\n' for e in prefix)).encode()),
                tasks=[terminals[s] for s in FIRST])

def validate_smoke(approval, events, runner_version):
    if type(approval) is not dict or set(approval)!={'schema','approved','reviewer','reviewed_at','review_method','checked_items','snapshot','reviewer_declaration','snapshot_digest'}:
        raise ValueError('human_smoke_schema')
    snap=smoke_snapshot(events, runner_version)
    if dump(approval['snapshot'])!=dump(snap):raise ValueError('human_snapshot_mismatch')
    if events[0].get('mode')=='live':
        manifest=ROOT/'HOLDOUT_D_AMEND4.sha256'
        if not manifest.is_file() or snap['amend3_manifest_sha256']!=sha(manifest.read_bytes()):
            raise ValueError('human_frozen_manifest_mismatch')
    checks=approval['checked_items']
    if type(checks) is not dict or set(checks)!=set(CHECKS) or any(type(v) is not bool for v in checks.values()):
        raise ValueError('human_checklist_schema')
    if approval['schema']!='HOLDOUT_D_AMEND3_HUMAN_REVIEW_V1' or approval['approved'] is not True:
        raise ValueError('human_smoke_not_approved')
    if type(approval['reviewer']) is not str or not approval['reviewer'].strip():raise ValueError('human_reviewer_required')
    if approval['review_method']!='manual':raise ValueError('manual_review_required')
    if approval['reviewer_declaration']!=REVIEW_DECLARATION:raise ValueError('reviewer_declaration_required')
    if approval['snapshot_digest']!=sha(dump(snap).encode()):raise ValueError('review_snapshot_digest_mismatch')
    if any(v is not True for v in checks.values()):
        raise ValueError('human_checklist_incomplete')
    # 规范 JSON 比较还区分 True/1、浮点/整数，拒绝宽松 Python equality。
    reviewed=timestamp(approval['reviewed_at'])
    if reviewed<events[snap['wal_seq']-1]['utc'] or reviewed>datetime.now(timezone.utc).timestamp():
        raise ValueError('human_review_time')
    return snap
