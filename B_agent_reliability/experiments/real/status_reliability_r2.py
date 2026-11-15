"""Read-only local progress snapshot; no model calls."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
out=root/('calib1_real' if (root/'calib1_real').exists() else 'difficulty_r2_real')
s=json.loads((out/'summary.json').read_text())
raw=(out/'budget.jsonl').read_text()
b=[json.loads(x) for x in raw.splitlines()[:-1 if not raw.endswith('\n') else None]]
quotas=[{k:x[k] for k in ('calls','five_hour','week')} for x in b if x['event']=='quota']
v=dict(phase=s['phase'],ended=len(s['results']),planned=len(s['schedule']),
       complete=sum(x['status']=='success' for x in s['results']),
       calls=sum(x['event']=='reserve' for x in b),
       service_failures=sum(x['event']=='outcome' and x['failed'] for x in b),
       service_exhausted=sum(x['status']=='service_retry_exhausted' for x in s['results']),
       cooldowns=sum(x['event']=='cooldown' for x in b),
       seed_errors_finished=sum(x['seed_errors'] for x in s['results']),
       natural_actions_finished=sum(x['advances'] for x in s['results']),
       halt=s['halt'],last_quota=quotas[-1] if quotas else None,
       service_boundaries=[{k:x[k] for k in ('attempted','recent_count','failures')} for x in b if x['event']=='service_boundary'],
       finished=s.get('finished_utc'))
print(json.dumps(v,ensure_ascii=False))
log=(root/'SUPERVISOR_R2.log').read_text()
if log:print(log[-1500:])
