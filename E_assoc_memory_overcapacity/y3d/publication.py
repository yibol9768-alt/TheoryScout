"""监管器收割分析子进程后唯一可调用的确认提交协议。"""
import csv
import math
import os
from common import dump, sha, jobs, METHODS, TASKS, DIMS, PAIRS
from integrity import inspect, read_json, stage_ok, valid_usage

REPORTS = ['PENDING.json','data_hashes.json','invalid_files.json','integrity_errors.json',
           'start.json','endpoints.csv','diagnostics72.csv','secondary_old45.csv',
           'mechanism90.csv','availability.csv','all_completed.csv']
INPUTS = ['launch.json','start.json','run_exit.json','runner_status.json','events.jsonl']


def report_inventory(output):
    """分析完成时的内容摘要；CSV 行数不含表头，JSON 不设表格行数。"""
    tables = {name for name in REPORTS if name.endswith('.csv')}
    assert {p.name for p in output.glob('*.csv')} == tables, 'report_set_mismatch'
    result = {}
    for name in REPORTS[1:]:  # PENDING 不能包含自身哈希
        path = output/name
        rows = None
        if name in tables:
            with path.open(newline='') as f:
                reader = csv.DictReader(f)
                assert reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), name
                rows = 0
                for row in reader:
                    assert None not in row and all(v is not None for v in row.values()), name
                    rows += 1
        result[name] = dict(sha256=sha(path), rows=rows)
    return result

def validate_reports(data, candidate, cfg):
    """从报告数值和实际文件集合重算资格/判定，不采用候选布尔量。"""
    output = data/'analysis'
    assert report_inventory(output) == candidate['report_inventory'], 'analysis_report_inventory_mismatch'
    plan = set(jobs(cfg['blocks']))
    names = {f'd{d}_r{ir}_s{s}.npz':(d,ir,s) for d,ir,s in plan}
    files = {p.name for p in data.glob('*.npz')}
    assert files <= set(names), 'npz_outside_plan'
    cells = {names[n] for n in files}
    assert isinstance(candidate['readable_cells'],list) and all(isinstance(k,list) and len(k)==3 and all(type(v) is int for v in k) for k in candidate['readable_cells']), 'invalid_readable_cells'
    assert candidate['readable_cells'] == [list(k) for k in sorted(cells)], 'readable_cells_mismatch'
    blocks = {s for s in range(cfg['blocks']) if all((d,ir,s) in cells for d in DIMS for ir in range(3))}
    expected = dict(complete_cells=len(cells), complete_blocks=len(blocks),
                    ungrouped_complete_cells=sum(s not in blocks for d,ir,s in cells),
                    planned_complete=cells == plan, diagnostic_family=72, confirm_family=cfg['confirm_family'])
    for key, value in expected.items():
        assert type(candidate[key]) is type(value) and candidate[key] == value, key
    def rows(name):
        with (output/name).open(newline='') as f: return list(csv.DictReader(f))
    def boolean(value):
        assert value in ('True','False'), 'invalid_csv_boolean'
        return value == 'True'
    def keyed(name, fields, wanted):
        table = rows(name); found = {}
        for row in table:
            key = tuple(int(row[k]) if k in ('d','ir','seed') else row[k] for k in fields)
            assert key not in found, name+'_duplicate_key'
            found[key] = row
        assert set(found) == wanted, name+'_coverage'
        return found
    availability = keyed('availability.csv', ('d','ir','seed'), plan)
    for key, row in availability.items():
        present = key in cells
        assert boolean(row['complete']) == present and boolean(row['complete_block']) == (key[2] in blocks)
        assert (row['state'] == 'complete') == present
    completed = keyed('all_completed.csv', ('d','ir','seed','method','task'),
                      {(*k,m,t) for k in cells for m in METHODS for t in TASKS})
    assert len(completed) == 10*len(cells)  # 完整计划必须恰好 2880 行
    for key, row in completed.items():
        assert boolean(row['complete_block']) == (key[2] in blocks)
        for field in ('recall','D','V','var_D','var_V','cov_DV'):
            assert math.isfinite(float(row[field])), 'nonfinite_completed_value'
    def bounds(row, n):
        assert int(row['n']) == n, 'sample_count_mismatch'
        if n < 2:
            assert row['lower'] == row['upper'] == '', 'unavailable_interval_mismatch'
            return None, None
        lower, upper = float(row['lower']), float(row['upper'])
        assert math.isfinite(lower) and math.isfinite(upper) and lower <= upper, 'invalid_interval'
        return lower, upper
    labels = ['three_dim_old','three_dim_new','d512_rminus_half_GD_minus_Sign_old']
    endpoints = keyed('endpoints.csv', ('endpoint',), {(k,) for k in labels})
    passed = []
    target_n = sum((cfg['target_d'],cfg['target_ir'],s) in cells for s in range(cfg['blocks']))
    for i, label in enumerate(labels):
        row = endpoints[label,]
        n = len(blocks) if i < 2 else target_n
        eligible = n >= cfg['min_blocks'] if i < 2 else cfg['d512_confirmatory'] and n == cfg['blocks']
        lower, upper = bounds(row, n)
        passed.append(bool(eligible and lower is not None and lower > 0))
        assert boolean(row['eligible']) == eligible and boolean(row['passed']) == passed[-1], 'endpoint_decision_mismatch'
    diag_keys = {(d,ir,task,a+'-'+b if b else a) for d in DIMS for ir in range(3) for task in TASKS
                 for a,b in (PAIRS if task == 'old' else [(m,None) for m in METHODS])}
    diagnostics = keyed('diagnostics72.csv', ('d','ir','task','label'), diag_keys)
    equivalences = []
    for (d,ir,task,label), row in diagnostics.items():
        n = sum((d,ir,s) in cells for s in range(cfg['blocks']))
        lower, upper = bounds(row, n)
        equivalent = lower is not None and lower > -cfg['equivalence_band'] and upper < cfg['equivalence_band']
        assert boolean(row['equivalent']) == equivalent, 'diagnostic_decision_mismatch'
        equivalences.append(equivalent)
    keyed('secondary_old45.csv', ('d','ir','method'), {(d,ir,m) for d in DIMS for ir in range(3) for m in METHODS})
    keyed('mechanism90.csv', ('d','ir','method','task'), {(d,ir,m,t) for d in DIMS for ir in range(3) for m in METHODS for t in TASKS})
    decisions = dict(primary_both_passed=all(passed[:2]), d512_passed=passed[2], full72_equivalent=all(equivalences))
    assert set(candidate['candidate_decisions']) == set(decisions), 'candidate_decision_fields'
    for key, value in decisions.items():
        assert type(candidate['candidate_decisions'][key]) is bool and candidate['candidate_decisions'][key] is value, 'candidate_decision_mismatch:'+key
        assert candidate[key] is False, 'pending_must_not_confirm'
    return decisions

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
        decisions = validate_reports(data,candidate,cfg)
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
                    **decisions,
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
    assert candidate['analysis_complete'] is True and candidate['publication_state']=='pending_supervisor'
    assert candidate['manifest_sha256']==mh and not candidate['invalid_files'] and not candidate['integrity_errors']
    hashes = {p.name:sha(p) for p in sorted(data.glob('*.npz'))}
    assert hashes == candidate['data_hashes'] == read_json(data/'analysis/data_hashes.json')
    decisions = validate_reports(data,candidate,cfg)
    audit = inspect(data,mh,cfg,{tuple(k) for k in candidate['readable_cells']},hashes)
    assert audit['valid'] and audit['runner_exit']==status['stages'][0]
    for key,value in decisions.items():
        assert type(accepted[key]) is bool and accepted[key] is value, 'accepted_decision_mismatch:'+key
    return accepted
