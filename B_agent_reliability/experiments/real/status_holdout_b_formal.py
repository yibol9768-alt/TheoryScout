"""Read-only progress for the frozen formal run."""
import collections
import json
from pathlib import Path

out = Path(__file__).resolve().parent / 'holdout_b_real'
summary = json.loads((out / 'summary.json').read_text())
events = []
for line in (out / 'budget.jsonl').read_text().splitlines():
    try:
        events.append(json.loads(line))
    except json.JSONDecodeError:
        pass
reserves = [x for x in events if x['event'] == 'reserve']
quotas = [x for x in events if x['event'] == 'quota']
print(json.dumps(dict(stage=summary['stage'], completed=len(summary['results']),
    calls=len(reserves), service_errors=sum(x['event'] == 'outcome' and x['failed'] for x in events),
    cooldowns=sum(x['event'] == 'cooldown' for x in events),
    lengths={str(length): dict(collections.Counter(x['status'] for x in summary['results'] if x['length'] == length)) for length in (16, 64)},
    last_quota=quotas[-1] if quotas else None, halt=summary['halt']), ensure_ascii=False))
