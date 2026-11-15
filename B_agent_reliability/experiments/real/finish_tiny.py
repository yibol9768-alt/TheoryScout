"""Finish only the three unattempted slots after the first per-run token violation.
No request retries and no reruns; fixed original pilot.py / prompt / task protocol.
"""
import hashlib
import json
from datetime import datetime, timezone
from pilot import ROOT, Real, Task, Trace, run_one

def main():
    out=ROOT/'tiny_real'
    summary=json.loads((out/'summary.json').read_text())
    assert len(summary['results'])==1 and summary['results'][0]['status']=='provider_token_limit'
    assert summary['calls']==2 and summary['halt']=='provider_token_limit'
    assert hashlib.sha256((ROOT/'pilot.py').read_bytes()).hexdigest()==summary['source_sha256']
    # Exclusive permanent marker: even interruption must never silently reattempt a slot.
    with (out/'continuation_once.json').open('x') as f:
        json.dump({'reason':'token-limit guard is per-run; complete unattempted slots only',
                   'previous_halt':summary['halt'], 'previous_calls':2,
                   'slots':[['kv',2207],['sheet',1105],['sheet',2207]]},f,indent=2)
    summary['management_note']='First token violation stopped batch; resumed only three unattempted slots, no rerun. Further token violations fail their run. Model mismatch stops batch.'
    budget={'calls':summary['calls'],'halt':None}
    summary['halt']=None
    client=Real()
    for family,seed in [('kv',2207),('sheet',1105),('sheet',2207)]:
        r=run_one(Task(family,seed),client,Trace(out/f'{family}_{seed}.jsonl'),
                  injected=seed==2207,budget=budget)
        summary['results'].append(r)
        summary['calls']=budget['calls']
        summary['finished_utc']=datetime.now(timezone.utc).isoformat()
        if r['status']=='model_mismatch' or r['status'].startswith('http_') or r['status']=='transport_or_response_error':
            summary['halt']=r['status']
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:r[k] for k in ['family','seed','status','calls','usage']}),flush=True)
        if summary['halt']: break
    summary['continuation_source_sha256']=hashlib.sha256(__file__.encode()).hexdigest() if False else hashlib.sha256((ROOT/'finish_tiny.py').read_bytes()).hexdigest()
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__': main()
