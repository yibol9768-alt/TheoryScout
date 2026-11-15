"""Pilot 2: fixed paired schedule, serial requests, fail-closed frozen sources."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from pilot import ROOT, MODEL, URL, MAX_COMPLETION_TOKENS, Task, Mock, Real, Trace, run_one

SEEDS = (3301, 3307, 3313, 3319, 3323, 3329)
FROZEN = ('pilot.py', 'pilot2.py', 'test_pilot.py', 'test_pilot2.py')

def schedule():
    result = []
    for family in ('kv', 'sheet'):
        for i, seed in enumerate(SEEDS):
            k = 2 if i < 3 else 3
            # Alternate execution order to limit systematic time/order confounding.
            for injected in ((False, True) if i % 2 == 0 else (True, False)):
                result.append(dict(family=family, seed=seed, K=k, injected=injected))
    return result

def hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FROZEN}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=('mock', 'real'))
    args = ap.parse_args()
    frozen = json.loads((ROOT/'FREEZE_PILOT2.json').read_text())
    if hashes() != frozen['sha256']: raise SystemExit('source freeze mismatch')
    prereg = ROOT/'PREREG_PILOT2.md'
    if not prereg.is_file(): raise SystemExit('missing preregistration')
    out = ROOT/('pilot2_'+args.mode)
    out.mkdir(exist_ok=False)
    client = Real() if args.mode == 'real' else Mock()
    budget = dict(calls=0, limit=500, halt=None)
    summary = dict(mode=args.mode, model_requested=MODEL if args.mode=='real' else 'mock',
                   endpoint=URL if args.mode=='real' else None,
                   started_utc=datetime.now(timezone.utc).isoformat(),
                   concurrency=1, global_call_cap=500, per_run_call_cap=20,
                   max_completion_tokens=MAX_COMPLETION_TOKENS,
                   frozen_sha256=hashes(),
                   prereg_sha256=hashlib.sha256(prereg.read_bytes()).hexdigest(),
                   schedule=schedule(), results=[], calls=0, halt=None)
    def save():
        summary.update(calls=budget['calls'], halt=budget['halt'],
                       updated_utc=datetime.now(timezone.utc).isoformat())
        (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    save()
    for row in schedule():
        if budget['halt']: break
        trace = Trace(out/f"{row['family']}_{row['seed']}_K{row['K']}_{'injected' if row['injected'] else 'clean'}.jsonl")
        r = run_one(Task(row['family'], row['seed'], 16), client, trace,
                    injected=row['injected'], k=row['K'], call_limit=20, budget=budget)
        summary['results'].append(r)
        if (r['status'] in ('model_mismatch', 'transport_or_response_error', 'missing_usage', 'global_call_limit')
                or r['status'].startswith('http_')):
            budget['halt'] = r['status']
        save()
        print(json.dumps({key:r[key] for key in ('family','seed','K','injected','status','calls','usage')}), flush=True)
    summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()

if __name__=='__main__': main()
