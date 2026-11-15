"""监管器收割分析子进程后唯一可调用的确认提交协议。"""
import os
from common import dump, sha
from integrity import inspect, read_json, stage_ok, valid_usage

REPORTS = ['PENDING.json','data_hashes.json','invalid_files.json','integrity_errors.json',
           'start.json','endpoints.csv','diagnostics72.csv','secondary_old45.csv',
           'mechanism90.csv','availability.csv','all_completed.csv']
INPUTS = ['launch.json','start.json','run_exit.json','runner_status.json','events.jsonl']

def finalize(data, mh, cfg, status, budget_probe=None):
    """必须在wait()后调用；返回资格，不接受分析器自报的退出状态。"""
    output = data/'analysis'; errors = []
    def reject(reason): errors.append(reason)
    if (output/'FINAL_ACCEPTED.json').exists(): raise FileExistsError('accepted credential already exists')
    stages = status.get('stages',[])
    if len(stages)!=2 or not all(stage_ok(s,p) for s,p in zip(stages,['run.py','analyze.py'])):
        reject('both_subprocesses_must_exit_zero_without_hard_kill')
    def number(v): return type(v) in (int,float) and __import__('math').isfinite(v) and v >= 0
    try:
        candidate = read_json(output/'PENDING.json')
        if not (candidate['analysis_complete'] is True and candidate['confirmation_valid'] is False
                and candidate['candidate_confirmation_valid'] is True
                and candidate['publication_state']=='pending_supervisor'
                and candidate['manifest_sha256']==mh and not candidate['invalid_files']
                and not candidate['integrity_errors']): reject('candidate_invalid')
        if not valid_usage(candidate['usage']) or candidate['usage']['cpu_seconds'] > cfg['analysis_cpu_hard']:
            reject('analysis_cpu_budget_invalid')
        hashes = {p.name:sha(p) for p in sorted(data.glob('*.npz'))}
        if hashes != candidate['data_hashes'] or hashes != read_json(output/'data_hashes.json'):
            reject('data_changed_after_analysis')
        audit = inspect(data,mh,cfg,{tuple(k) for k in candidate['readable_cells']},hashes)
        if not audit['valid']: reject(dict(integrity_recheck=audit['errors']))
        if stages and audit['runner_exit'] != stages[0]: reject('supervisor_runner_exit_mismatch')
        report_hashes = {str((output/name).relative_to(data)):sha(output/name) for name in REPORTS}
        input_hashes = {name:sha(data/name) for name in INPUTS}
    except Exception as exc:
        reject(dict(invalid_pending_or_evidence=repr(exc)))
    # 把重新核验与哈希成本也计入监管预算，在最终提交前刷新实测值。
    if budget_probe is not None: status = dict(status, **budget_probe())
    supervisor = status.get('supervisor_usage',{})
    children = status.get('cpu_seconds_children'); wall = status.get('wall_seconds')
    if not (valid_usage(supervisor) and supervisor['cpu_seconds'] < 10
            and number(children) and children+supervisor['cpu_seconds'] <= cfg['formal_cpu_hard']
            and children+supervisor['cpu_seconds']+cfg['preparation_cpu_reserve'] <= cfg['total_cpu_hard']
            and number(wall) and wall <= cfg['total_wall_hard']): reject('supervisor_or_total_budget_invalid')
    # launch_status是先发布的不可覆盖证据；不存在最终凭据时仍不得采信。
    status = dict(status, publication_eligible=not errors, publication_errors=errors,
                  manifest_sha256=mh)
    dump(data/'launch_status.json',status)
    if errors: return False
    accepted = dict(schema=1, publication_state='accepted', analysis_complete=True,
                    confirmation_valid=True, manifest_sha256=mh,
                    **candidate['candidate_decisions'],
                    runner_exit=stages[0], analysis_exit=stages[1],
                    evidence_sha256=dict(**input_hashes,**report_hashes,
                                         **hashes,**{'launch_status.json':sha(data/'launch_status.json')}))
    # 先fsync完整候选凭据，再排他link发布；半写或SIGKILL不会留下最终路径。
    pending = output/'FINAL_ACCEPTED.pending'
    dump(pending,accepted)
    if budget_probe is not None:
        last = budget_probe()
        u = last.get('supervisor_usage',{})
        w = last.get('wall_seconds')
        if not (valid_usage(u) and u['cpu_seconds'] < 10 and number(w) and w <= cfg['total_wall_hard']
                and children+u['cpu_seconds'] <= cfg['formal_cpu_hard']
                and children+u['cpu_seconds']+cfg['preparation_cpu_reserve'] <= cfg['total_cpu_hard']):
            dump(data/'publication_aborted.json',dict(reason='budget_exceeded_before_atomic_publish',last_probe=last))
            return False
    os.link(pending,output/'FINAL_ACCEPTED.json')
    fd = os.open(output,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    pending.unlink()
    return True

def verify_accepted(data, mh, cfg):
    """READY采信条件的机器校验；损坏、无凭据、预算/退出异常均拒绝。"""
    accepted = read_json(data/'analysis/FINAL_ACCEPTED.json')
    assert not (data/'publication_aborted.json').exists()
    assert accepted['schema']==1 and accepted['publication_state']=='accepted'
    assert accepted['analysis_complete'] is True and accepted['confirmation_valid'] is True
    assert accepted['manifest_sha256']==mh
    expected = set(INPUTS+['launch_status.json']) | {'analysis/'+n for n in REPORTS} | {p.name for p in data.glob('*.npz')}
    assert set(accepted['evidence_sha256'])==expected
    for rel,h in accepted['evidence_sha256'].items():
        assert sha(data/rel)==h, rel
    status = read_json(data/'launch_status.json')
    assert status['publication_eligible'] is True and status['publication_errors']==[]
    assert status['manifest_sha256']==mh
    assert len(status['stages'])==2
    assert all(stage_ok(s,p) for s,p in zip(status['stages'],['run.py','analyze.py']))
    assert accepted['runner_exit']==status['stages'][0] and accepted['analysis_exit']==status['stages'][1]
    assert 0 <= status['wall_seconds'] <= cfg['total_wall_hard']
    u=status['supervisor_usage'];c=status['cpu_seconds_children']
    assert valid_usage(u) and u['cpu_seconds']<10 and 0<=c
    assert c+u['cpu_seconds']<=cfg['formal_cpu_hard']
    assert c+u['cpu_seconds']+cfg['preparation_cpu_reserve']<=cfg['total_cpu_hard']
    candidate=read_json(data/'analysis/PENDING.json')
    assert candidate['candidate_confirmation_valid'] is True and candidate['confirmation_valid'] is False
    assert valid_usage(candidate['usage']) and candidate['usage']['cpu_seconds']<=cfg['analysis_cpu_hard']
    assert accepted['primary_both_passed']==candidate['candidate_decisions']['primary_both_passed']
    assert accepted['d512_passed']==candidate['candidate_decisions']['d512_passed']
    assert accepted['full72_equivalent']==candidate['candidate_decisions']['full72_equivalent']
    return accepted
