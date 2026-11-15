"""汇总实际执行结果；不重新计算统计判决。"""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit_y3c_checks'
def read(p):return json.loads(p.read_text())
protocol=read(OUT/'protocol_replay/results.json')
synthetic=read(OUT/'synthetic_results.json')
old=read(ROOT/'y3b/synthetic_v4_tests.json')
faults=read(OUT/'independent_faults/results.json')
false=read(OUT/'false_confirmation/results.json')
verify_only=json.loads((OUT/'verify_only.log').read_text().splitlines()[-1])
assert verify_only['verified'] is True and verify_only['formal_sampling'] is False
raws=[x for x in protocol['outcomes'] if x['case'].startswith('audit_')]
assert len(raws)==13 and all(not x['accepted'] for x in raws)
assert synthetic['checks']==old['checks'] and len(synthetic['checks'])==59
injections=[]
for mode in ['after_pending','before_pending','nonzero']:
    name='actual_analysis_'+mode
    row=next(x for x in protocol['outcomes'] if x['case']==name)
    status=read(OUT/'protocol_replay'/name/'formal/launch_status.json')
    injections.append({'mode':mode,'actual_returncode':status['stages'][1]['returncode'],'accepted':row['accepted']})
assert [x['actual_returncode'] for x in injections]==[-9,-9,7]
assert all(not x['accepted'] for x in injections)
assert false['accepted'] and false['verification_succeeded']
assert not any(false['original_decisions'].values()) and all(false['published_decisions'].values())
assert all(x['passed']=='False' for x in false['endpoints_untouched'])
result=dict(verdict='需修正，当前不可正式运行',protocol_checks=len(protocol['checks']),
    protocol_outcomes=len(protocol['outcomes']),raw_fixtures=len(raws),raw_outcomes=raws,
    synthetic_checks=len(synthetic['checks']),synthetic_labels_identical=True,
    actual_analysis_exit_injections=injections,new_fault_results=faults['results'],
    false_confirmation_original=false['original_decisions'],false_confirmation_published=false['published_decisions'],
    verify_only=verify_only,
    independent_fixture_manifest_sha256=faults['manifest_sha256'],
    protocol_fixture_manifest_binding='Y3B,复用原审计NPZ',independent_fixture_binding='真实Y3C清单；确定性重封装metadata')
name=sys.argv[1] if len(sys.argv)>1 else 'summary'
assert name.replace('_','').isalnum()
with (OUT/(name+'.json')).open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps({k:v for k,v in result.items() if k not in ['raw_outcomes','new_fault_results']},ensure_ascii=False))
