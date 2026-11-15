import json
from harness import OUT,resources
def read(name):return json.loads((OUT/name).read_text())
final=read('final_results.json');baseline=read('baseline.json')
summary=dict(resources=resources(),old_files=len(baseline['files']),final=final,suites={})
for v in ('v1','v2','v3'):
    z=read(v+'_results.json');summary['suites'][v]={k:z[k] for k in ('tests','passed','failures','errors')};summary['suites'][v]['max_threads']=max([z['resources']['threads'],*z['thread_counts']])
z=read('boundary_results.json');summary['boundary']=dict(tests=z['tests'],passed=z['passed'],failures=len(z['failures']),errors=len(z['errors']),max_threads=max([z['resources']['threads'],*z['thread_counts']]))
wal=OUT/'boundary/pause_normal_request/HOLDOUT_D_WAL.jsonl'
summary['normal_request_event_order']=[{k:e[k] for k in ('seq','kind','call','status','until') if k in e} for e in map(json.loads,wal.read_text().splitlines())]
smoke=read('ported_attack_smoke/smoke_results.json')
summary['smoke_fields']={k:len(v) if isinstance(v,list) else v for k,v in smoke.items() if k not in ('resources','forged_current_snapshot')}
(OUT/'review_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2,ensure_ascii=False))
