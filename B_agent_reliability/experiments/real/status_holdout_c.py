"""Read-only progress view; never prints prompts, responses or credentials."""
import collections
import json
from pathlib import Path

root = Path(__file__).resolve().parent / 'holdout_c_real'
summary = json.loads((root / 'summary.json').read_text())
counts = collections.Counter()
active = {}
last_control = None
for line in (root / 'budget.jsonl').read_text().splitlines():
    row = json.loads(line)
    event = row['event']
    if event == 'reserve':
        counts['attempts'] += 1
        active[row['call']] = row['task']
    elif event == 'outcome':
        counts['failed' if row['failed'] else 'returned'] += 1
        active.pop(row['call'], None)
    elif event in ('pause', 'quota', 'quota_failed'):
        last_control = row
print('stage', summary['stage'], 'terminal', len(summary['results']), '/40', dict(counts))
for length in (16, 64):
    rows = [row for row in summary['results'] if row['length'] == length]
    print('L', length, 'terminal', len(rows), 'success', sum(row['status'] == 'success' for row in rows))
for call, name in active.items():
    with (root / (name + '.jsonl')).open() as stream:
        last = None
        for line in stream:
            row = json.loads(line)
            if row['event'] == 'request' and row['call'] == call:
                last = row
    if last:
        print('active', name, 'call', call, 'step', last['step'], 'review', last['review'], 'utc', last['utc'])
if last_control:
    print('last_control', json.dumps(last_control, ensure_ascii=False))
