"""Disclosed 408 timeout correction and exact-state continuation; frozen C unchanged."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor,as_completed
import holdout_c as h
import reliability_core as d

class Budget(h.Budget):
    def __init__(self,out,mock=False,resume=False):
        if not resume:
            super().__init__(out,mock,False);return
        self.lock=threading.RLock();self.real=not mock;self.halt=None
        ev=[json.loads(x) for x in (out/'budget.jsonl').read_text().splitlines()]
        self.calls=sum(x['event']=='reserve' for x in ev)
        self.successful=sum(x['event']=='outcome' and not x['failed'] for x in ev)
        self.failed=sum(x['event']=='outcome' and x['failed'] for x in ev)
        assert self.calls==self.successful+self.failed
        assert all(x.get('reason')=='http_408' for x in ev if x['event']=='halt')
        self.inflight=0;self.cap=2000;self.pause_until=0;self.quota_at=-1;self.need_probe=True
        self.journal=d.Journal.__new__(d.Journal);self.journal.f=(out/'budget.jsonl').open('a')

class Client(h.Client):
    def __init__(self,budget,mock=False):
        super().__init__(budget,mock)
        original=self.opener.open
        def opened(*a,**kw):
            try:return original(*a,**kw)
            except urllib.error.HTTPError as error:
                if error.code==408:raise urllib.error.URLError('http_408_timeout') from None
                raise
        self.opener.open=opened

class ContinuationTrace:
    def __init__(self,path):self.f=path.open('a');self.suppress=True
    def add(self,event,**kw):
        if not self.suppress:
            self.f.write(d.dump(dict(event=event,utc=d.now(),**kw))+'\n');self.f.flush()
            h.os.fsync(self.f.fileno())

# This module does not run old actions again: deterministic reconstruction returns
# the already recorded responses in memory, suppressing duplicate journal writes.
def continue_smoke():
    h.verify();out=h.OUT
    summary=json.loads((out/'summary.json').read_text())
    assert summary['halt']=='http_408' and len(summary['results'])==2
    saved=out/'STOP_408_ORIGINAL_SUMMARY.json'
    if not saved.exists():saved.write_bytes((out/'summary.json').read_bytes())
    partial=next(x for x in summary['results'] if x['status']=='http_408')
    path=out/(partial['id']+'.jsonl')
    ev=[json.loads(x) for x in path.read_text().splitlines()]
    responses={x['call']:x['response'] for x in ev if x['event']=='response'}
    requests=[x for x in ev if x['event']=='request' and x['call'] in responses]
    pending=[x for x in ev if x['event']=='request' and x['call'] not in responses]
    assert len(pending)==1 and pending[0]['call']==22
    # Archive original bytes before append, preserving original stop and failed request.
    for original in (path,out/'budget.jsonl'):
        archived=out/('STOP_408_ORIGINAL_'+original.name)
        assert not archived.exists();archived.write_bytes(original.read_bytes())
    journal=d.Journal.__new__(d.Journal);journal.f=(out/'budget.jsonl').open('a')
    journal.add('engineering_recovery',reason='408_is_authorized_timeout',original_call=22)
    journal.add('outcome',call=22,failed=True,successful=21,reconstructed_from='original_http_408_halt')
    # Account actual time since timeout, no unnecessary extra waiting.
    failed_time=datetime.fromisoformat(ev[-1]['utc']).timestamp()
    journal.f.close()
    budget=Budget(out,resume=True)
    budget.pause_until=time.monotonic()+max(0,failed_time+300-time.time())
    client=Client(budget)
    class Replay:
        def __init__(self):self.index=0;self.budget=budget
        def request(self,messages,trace,context,expected=None):
            if self.index<len(requests):
                x=requests[self.index];self.index+=1
                assert x['messages']==messages and all(x[k]==v for k,v in context.items())
                return responses[x['call']],x['call']
            if self.index==len(requests):
                x=pending[0]
                assert x['messages']==messages and all(x[k]==v for k,v in context.items())
                trace.suppress=False;self.index+=1
                trace.add('exact_state_recovery',original_call=x['call'],step=x['step'],review=x['review'])
            return client.request(messages,trace,context,expected)
    original_journal=d.Journal
    d.Journal=ContinuationTrace
    try:
        result=d.run_task({k:partial[k] for k in ('family','level','knob','seed','length')},Replay(),out)
    finally:d.Journal=original_journal
    summary['results']=[result if x['id']==result['id'] else x for x in summary['results']]
    summary.update(calls=budget.calls,successful_calls=budget.successful,service_failures=budget.failed,
                   halt=budget.halt,stage='stopped' if budget.halt else 'awaiting_human_smoke_review',
                   engineering_correction='AMENDMENT_408_HOLDOUT_C.md',updated_utc=d.now())
    h.write(out/'summary.json',summary);budget.journal.f.close()
    print(d.dump(result),flush=True);print('STAGE',summary['stage'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['smoke','formal'])
    args=ap.parse_args()
    if args.action=='smoke':continue_smoke()
    else:
        h.Budget=Budget;h.Client=Client;h.execute(resume=True)
