"""Final administrative completion: attempt each remaining slot once, never retry."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pilot import ROOT, MODEL, Real, Mock, Task, Trace, run_one
from pilot2 import hashes, schedule

FILES=('finish_pilot2.py','test_finish_pilot2.py')
def source_hashes():return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES}
def todo(base):
    if base['halt']!='transport_or_response_error':raise ValueError('not a transport stop')
    if base['schedule']!=schedule():raise ValueError('schedule changed')
    done=base['results']
    if not done or len(done)>=24:raise ValueError('no remaining slots')
    for planned,observed in zip(schedule(),done):
        if any(planned[k]!=observed[k] for k in planned):raise ValueError('not a fixed prefix')
    return schedule()[len(done):]
def fatal(status):
    return status in ('model_mismatch','missing_usage','global_call_limit')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('mock','real'),required=True);args=ap.parse_args()
    base_path=ROOT/'SUMMARY_PILOT2.json';base=json.loads(base_path.read_text());rows=todo(base)
    core=json.loads((ROOT/'FREEZE_PILOT2.json').read_text())
    if hashes()!=core['sha256']:raise SystemExit('core freeze mismatch')
    amendment=ROOT/'AMENDMENT2_PILOT2.md';amend_hash=hashlib.sha256(amendment.read_bytes()).hexdigest()
    if args.mode=='real':
        f=json.loads((ROOT/'FREEZE_FINISH_PILOT2.json').read_text())
        if source_hashes()!=f['sha256'] or amend_hash!=f['amendment_sha256']:raise SystemExit('final freeze mismatch')
        if hashlib.sha256(base_path.read_bytes()).hexdigest()!=f['base_summary_sha256']:raise SystemExit('base changed')
    counted=0
    for r in base['results']:
        for line in (ROOT/r['trace']).read_text().splitlines():
            e=json.loads(line)
            counted+=e['event']=='request'
            if e['event']=='response' and e['response'].get('model')!=MODEL:raise SystemExit('model mismatch: prohibited')
    if counted!=base['calls']:raise SystemExit('prior requests disagree')
    out=ROOT/('pilot2_finish_'+args.mode);out.mkdir(exist_ok=False)
    budget=dict(calls=counted,limit=500,halt=None);client=Real() if args.mode=='real' else Mock()
    summary=dict(mode=args.mode,started_utc=datetime.now(timezone.utc).isoformat(),
                 prior_calls=counted,calls=0,results=[],schedule=rows,halt=None,concurrency=1,
                 base_summary_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                 amendment_sha256=amend_hash,source_sha256=source_hashes())
    def save():
        summary.update(calls=budget['calls']-counted,halt=budget['halt'])
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    save()
    for row in rows:
        if budget['halt']:break
        trace=Trace(out/f"{row['family']}_{row['seed']}_K{row['K']}_{'injected' if row['injected'] else 'clean'}.jsonl")
        r=run_one(Task(row['family'],row['seed'],16),client,trace,
                  injected=row['injected'],k=row['K'],call_limit=20,budget=budget)
        summary['results'].append(r)
        if fatal(r['status']):budget['halt']=r['status']
        save();print(json.dumps({k:r[k] for k in ('family','seed','K','injected','status','calls','usage')}),flush=True)
    summary['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    if args.mode=='real':
        merged=dict(base)
        merged.update(results=base['results']+summary['results'],calls=budget['calls'],
                      finished_utc=summary['finished_utc'],halt=budget['halt'],
                      final_continuation=summary,second_halt=base['halt'])
        with (ROOT/'FINAL_SUMMARY_PILOT2.json').open('x') as f:json.dump(merged,f,ensure_ascii=False,indent=2);f.write('\n')

if __name__=='__main__':main()
