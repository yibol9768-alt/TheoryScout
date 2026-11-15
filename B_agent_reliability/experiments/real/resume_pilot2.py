"""One continuation of never-attempted slots after a transport stop; no retries."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pilot import ROOT, MODEL, Real, Mock, Task, Trace, run_one
from pilot2 import hashes, schedule

EXTRA=('resume_pilot2.py','test_resume_pilot2.py')
def extra_hashes():
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in EXTRA}

def remaining(base):
    if base['halt']!='transport_or_response_error':
        raise ValueError('continuation only after original transport stop')
    if base['schedule']!=schedule(): raise ValueError('schedule mismatch')
    done=base['results']
    for planned,actual in zip(schedule(),done):
        if any(planned[k]!=actual[k] for k in planned): raise ValueError('prefix mismatch')
    if not done or done[-1]['status']!='transport_or_response_error': raise ValueError('missing failed slot')
    return schedule()[len(done):]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=('mock','real'));args=ap.parse_args()
    base_path=ROOT/'pilot2_real/summary.json';base=json.loads(base_path.read_text())
    todo=remaining(base)
    frozen=json.loads((ROOT/'FREEZE_PILOT2.json').read_text())
    ext=json.loads((ROOT/'FREEZE_RESUME_PILOT2.json').read_text())
    if hashes()!=frozen['sha256'] or extra_hashes()!=ext['sha256']: raise SystemExit('freeze mismatch')
    amendment=ROOT/'AMENDMENT_PILOT2.md'
    amendment_hash=hashlib.sha256(amendment.read_bytes()).hexdigest()
    if amendment_hash!=ext['amendment_sha256']:raise SystemExit('amendment mismatch')
    prior_calls=0
    for r in base['results']:
        events=[json.loads(x) for x in (ROOT/r['trace']).read_text().splitlines()]
        prior_calls+=sum(e['event']=='request' for e in events)
        for e in events:
            if e['event']=='response' and e['response'].get('model')!=MODEL:
                raise SystemExit('prior model mismatch: continuation prohibited')
    if prior_calls!=base['calls']:raise SystemExit('prior request count mismatch')
    out=ROOT/('pilot2_resume_'+args.mode);out.mkdir(exist_ok=False)
    client=Real() if args.mode=='real' else Mock()
    budget=dict(calls=prior_calls,limit=500,halt=None)
    result=dict(mode=args.mode,started_utc=datetime.now(timezone.utc).isoformat(),
                base_summary_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                prior_calls=prior_calls,schedule=todo,results=[],calls=0,halt=None,
                concurrency=1,global_call_cap=500,amendment_sha256=amendment_hash,
                continuation_sha256=extra_hashes())
    def save():
        result.update(calls=budget['calls']-prior_calls,halt=budget['halt'],updated_utc=datetime.now(timezone.utc).isoformat())
        (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    save()
    for row in todo:
        if budget['halt']: break
        trace=Trace(out/f"{row['family']}_{row['seed']}_K{row['K']}_{'injected' if row['injected'] else 'clean'}.jsonl")
        r=run_one(Task(row['family'],row['seed'],16),client,trace,
                  injected=row['injected'],k=row['K'],call_limit=20,budget=budget)
        result['results'].append(r)
        if r['status'] in ('model_mismatch','transport_or_response_error','missing_usage','global_call_limit') or r['status'].startswith('http_'):
            budget['halt']=r['status']
        save()
        print(json.dumps({k:r[k] for k in ('family','seed','K','injected','status','calls','usage')}),flush=True)
    result['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    if args.mode=='real':
        merged=dict(base)
        merged.update(results=base['results']+result['results'],calls=budget['calls'],halt=budget['halt'],
                      finished_utc=result['finished_utc'],continuation=result,
                      original_halt=base['halt'],original_finished_utc=base['finished_utc'])
        with (ROOT/'SUMMARY_PILOT2.json').open('x') as f:json.dump(merged,f,ensure_ascii=False,indent=2);f.write('\n')

if __name__=='__main__':main()
