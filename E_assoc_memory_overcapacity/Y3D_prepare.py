"""仅修改从 Y3C 复制的新目录；不启动抽样。"""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'y3d'
assert OUT.is_dir()
for p in OUT.glob('*.py'):
    source=p.read_text()
    p.write_text(source.replace('Y3C','Y3D').replace('y3c','y3d'))
p=OUT/'analyze.py'
s=p.read_text().replace('from integrity import inspect','from integrity import inspect\nfrom publication import report_inventory')
s=s.replace("    result['usage']=usage()", "    result['report_inventory']=report_inventory(output)\n    result['usage']=usage()")
p.write_text(s)
p=OUT/'publication.py'
s=p.read_text().replace('import os','import csv\nimport math\nimport os',1)
s=s.replace('from common import dump, sha','from common import dump, sha, jobs, METHODS, TASKS, DIMS, PAIRS')
helpers='''
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

'''
s=s.replace('\ndef finalize(', '\n'+helpers+'def finalize(',1)
s=s.replace("        report_hashes = {str((output/name).relative_to(data)):sha(output/name) for name in REPORTS}","        decisions = validate_reports(data,candidate,cfg)\n        report_hashes = {str((output/name).relative_to(data)):sha(output/name) for name in REPORTS}")
s=s.replace("**candidate['candidate_decisions'],","**decisions,")
old="""    assert accepted['primary_both_passed']==candidate['candidate_decisions']['primary_both_passed']
    assert accepted['d512_passed']==candidate['candidate_decisions']['d512_passed']
    assert accepted['full72_equivalent']==candidate['candidate_decisions']['full72_equivalent']"""
new="""    assert candidate['analysis_complete'] is True and candidate['publication_state']=='pending_supervisor'
    assert candidate['manifest_sha256']==mh and not candidate['invalid_files'] and not candidate['integrity_errors']
    hashes = {p.name:sha(p) for p in sorted(data.glob('*.npz'))}
    assert hashes == candidate['data_hashes'] == read_json(data/'analysis/data_hashes.json')
    decisions = validate_reports(data,candidate,cfg)
    audit = inspect(data,mh,cfg,{tuple(k) for k in candidate['readable_cells']},hashes)
    assert audit['valid'] and audit['runner_exit']==status['stages'][0]
    for key,value in decisions.items():
        assert type(accepted[key]) is bool and accepted[key] is value, 'accepted_decision_mismatch:'+key"""
assert old in s
p.write_text(s.replace(old,new))
