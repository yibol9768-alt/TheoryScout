"""Y3C数据前冻结；不改旧文件，预算与统计规范不变。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(12,15))
import ast,json,os,time
from pathlib import Path
from common import ROOT,OUT,sha,dump,usage,environment,config

def main():
    assert not (OUT/'formal').exists() and not (ROOT/'y3b/formal').exists()
    manifest=ROOT/'Y3C_FROZEN.sha256';assert not manifest.exists()
    assert environment()==json.loads((OUT/'environment.json').read_text())
    baseline=json.loads((OUT/'original_baseline.json').read_text())
    changed=[rel for rel,h in baseline.items() if not (ROOT/rel).is_file() or sha(ROOT/rel)!=h]
    assert not changed,changed
    added=[]
    for base,dirs,names in os.walk(ROOT):
        dirs[:]=[n for n in dirs if Path(base)/n != OUT]
        for name in names:
            p=Path(base)/name;rel=str(p.relative_to(ROOT))
            if p.suffix!='.log' and rel not in baseline and not p.name.startswith('Y3C_'):added.append(rel)
    assert not added,added
    for p in OUT.glob('*.py'):ast.parse(p.read_text())
    last=json.loads((OUT/'final_edges/results.json').read_text());assert last['passed']
    for name,h in last['source_sha256'].items():
        if name not in ['freeze.py','archive_tests.py','test_protocol.py']:assert sha(OUT/name)==h,name
    synthetic=json.loads((OUT/'synthetic_final_tests.json').read_text())
    assert synthetic['passed'] and len(synthetic['checks'])==59
    assert synthetic['checks']==json.loads((ROOT/'y3b/synthetic_v4_tests.json').read_text())['checks']
    old=json.loads((ROOT/'Y3B_FREEZE_AUDIT.json').read_text())
    # 用systemd的实测记录替代原失败冻结70秒保守上界，不改任何预算。
    # 原账本保留；原成功冻结/verify亦用scope实测CPU（含进程收尾）。
    prior_upper=old['prior_cpu_seconds']-70+3.046+3.411+1.066
    # systemd scope的CPUAccounting含失败测试及其真实子进程；日志证据另存。
    measured_initial=3.872+46.436+51.152+0.1  # snapshot、两次全套协议进程、prlimit管理进程
    protocol=json.loads((OUT/'protocol_v2/results.json').read_text());assert protocol['passed']
    for name,h in protocol['source_sha256'].items():
        if name not in ['freeze.py','archive_tests.py']:assert sha(OUT/name)==h,name
    archive=json.loads((OUT/'archive_status.json').read_text());assert archive['passed']
    test_cpu=sum(s['usage']['cpu_seconds']+s['cpu_seconds_children']
                 for s in [json.loads((OUT/'synthetic_y3c_tests.json').read_text()),synthetic,last,protocol,archive])
    expanded={OUT/n for n in archive['directories']}
    # 展开夹具全量封装到冻结tar；结果JSON另列，方便无需解包审阅。
    regular=[]
    for base,dirs,names in os.walk(OUT):
        dirs[:]=[n for n in dirs if Path(base)/n not in expanded]
        regular.extend(Path(base)/n for n in names)
    paths=sorted(regular
        +[OUT/'protocol_v1/FAILED.json',OUT/'protocol_v2/results.json',OUT/'final_edges/results.json']
        +[p for p in ROOT.glob('Y3C_*') if p.is_file()])
    assert all(not p.is_symlink() for p in paths)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    assert hashes['y3c/test_fixtures.tar']==archive['archive_sha256']
    # 哈希、旧文件复核已完成；余下写账本/清单由5秒收尾预留覆盖。
    upper=prior_upper+measured_initial+test_cpu+usage()['cpu_seconds']+9+5
    assert upper<config()['preparation_cpu_reserve'],upper
    audit=dict(schema=1,no_formal_sampling=True,no_llm_api=True,
        original_files_checked=len(baseline),original_files_changed=changed,unexpected_new_files=added,
        original_59_checks_passed=True,protocol_checks=len(protocol['checks']),final_edge_checks=len(last['checks']),
        audit_fixture_count=13,real_analysis_exit_injections=last['injections'],
        test_failures_preserved=['y3c/protocol_v1/FAILED.json','y3c/protocol_v2/FAILED.json'],
        statistics_predictions_parameters_seeds_budget_unchanged=True,
        prior_preparation_upper=prior_upper,measured_initial_scopes_cpu_seconds=measured_initial,
        revision_test_and_archive_cpu_seconds=test_cpu,freeze_cpu_hard_limit=15,
        verify_cpu_upper=9,tail_cpu_reserve=5,preparation_total_upper=upper,preparation_cpu_limit=200,
        freeze_usage=usage(),test_evidence_archive='y3c/test_fixtures.tar',
        expanded_test_directories_not_individually_listed=archive['directories'],
        legacy_cpu_refinement='原失败冻结70秒保守计费替换为systemd scope实测3.046秒；成功冻结3.411、原verify1.066，原账本不变。',
        limitations='本地哈希与排他发布，不是WORM或可信时间戳；未实测正式资源峰值。')
    dump(ROOT/'Y3C_FREEZE_AUDIT.json',audit)
    hashes['Y3C_FREEZE_AUDIT.json']=sha(ROOT/'Y3C_FREEZE_AUDIT.json')
    with manifest.open('x') as f:
        for rel,h in sorted(hashes.items()):f.write(h+'  '+rel+'\n')
        f.flush();os.fsync(f.fileno())
    os.chmod(manifest,0o444)
    print(json.dumps(dict(frozen_files=len(hashes),manifest_sha256=sha(manifest),preparation_total_upper=upper,usage=usage())))
if __name__=='__main__':main()
