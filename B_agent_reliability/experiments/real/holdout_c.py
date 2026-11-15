"""Independent C batch; unchanged tasks/scoring, globally pausing transport."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import threading
import time
import holdout_b as b
import reliability_core as d
import reliability_r2 as r
ROOT=b.ROOT
OUT=ROOT/'holdout_c_real'
write=b.write
sha=b.sha
stats=b.stats
params=b.params
RATES=b.RATES
quantile=b.quantile

def schedule():
    return [dict(family='sheet',level=0,knob=2,seed=7100000+10000*L+rep,length=L)
            for rep in range(1,21) for L in (16,64)]

SOURCES=('PREREG_HOLDOUT_C.md','holdout_c.py','test_holdout_c.py','report_holdout_c.py',
         'PREDICTIONS_HOLDOUT_C.json','holdout_b.py','reliability_core.py','reliability_r2.py',
         'report_reliability_r2.py','report_holdout_b_fixed.py','difficulty.py')

def freeze():
    b.verify()
    assert (ROOT/'PREDICTIONS_HOLDOUT_C.json').read_bytes()==(ROOT/'PREDICTIONS_HOLDOUT_B.json').read_bytes()
    old=set(x['seed'] for x in b.schedule())
    old.update(x['seed'] for x in json.loads((ROOT/'calib1_real/summary.json').read_text())['results'])
    assert not old.intersection(x['seed'] for x in schedule())
    inputs=json.loads((ROOT/'FREEZE_HOLDOUT_B.json').read_text())['input_sha256']
    obj=dict(utc=d.now(),sha256={n:sha(ROOT/n) for n in SOURCES},input_sha256=inputs,
             b_result_sha256=sha(ROOT.parent.parent/'RESULTS_HOLDOUT_B.md'),
             b_summary_sha256=sha(ROOT/'holdout_b_real/summary.json'),schedule=schedule())
    with (ROOT/'FREEZE_HOLDOUT_C.json').open('x') as f:f.write(d.dump(obj)+'\n')
    print('FROZEN',sha(ROOT/'FREEZE_HOLDOUT_C.json'))

def verify():
    f=json.loads((ROOT/'FREEZE_HOLDOUT_C.json').read_text())
    assert f['sha256']=={n:sha(ROOT/n) for n in SOURCES}
    assert all(sha(ROOT/n)==v for n,v in f['input_sha256'].items())
    assert sha(ROOT.parent.parent/'RESULTS_HOLDOUT_B.md')==f['b_result_sha256']
    assert sha(ROOT/'holdout_b_real/summary.json')==f['b_summary_sha256']
    assert f['schedule']==schedule()
    return f

class Budget:
    def __init__(self,out,mock=False,resume=False):
        self.lock=threading.RLock();self.real=not mock;self.halt=None
        self.calls=0;self.successful=0;self.failed=0;self.inflight=0
        self.cap=2000;self.pause_until=0;self.quota_at=-1;self.need_probe=False
        self.journal=d.Journal.__new__(d.Journal)
        if resume:
            ev=[json.loads(s) for s in (out/'budget.jsonl').read_text().splitlines()]
            assert not any(x['event']=='halt' for x in ev)
            self.calls=sum(x['event']=='reserve' for x in ev)
            self.successful=sum(x['event']=='outcome' and not x['failed'] for x in ev)
            self.failed=sum(x['event']=='outcome' and x['failed'] for x in ev)
            assert self.calls==self.successful+self.failed
            self.journal.f=(out/'budget.jsonl').open('a')
        else:self.journal.f=(out/'budget.jsonl').open('x')
    def stop(self,reason):
        with self.lock:
            self.halt=reason;self.journal.add('halt',reason=reason,calls=self.calls)
    def quota_values(self):
        z=subprocess.run(['python3','/home/liuyibo/auto-llm-driving/TheoryScout/tools/plus_quota.py'],
                         capture_output=True,text=True,timeout=50,check=True)
        a,c=map(float,z.stdout.split())
        if not (0<=a<=100 and 0<=c<=100):raise ValueError()
        return a,c
    def quota(self):
        # Called with the global lock held and no model HTTP in flight.
        while True:
            if not self.real:a,c=0.,0.
            else:
                try:a,c=self.quota_values()
                except Exception:
                    self.journal.add('quota_failed',calls=self.calls,successful=self.successful)
                    self.journal.add('pause',reason='quota_unavailable',seconds=300)
                    time.sleep(300);continue
            self.journal.add('quota',calls=self.calls,successful=self.successful,five_hour=a,week=c)
            if a<80 and c<90:
                self.quota_at=self.successful;self.need_probe=False;return
            self.journal.add('pause',reason='quota_threshold',seconds=300)
            time.sleep(300)
    def reserve(self,ident,config):
        while True:
            with self.lock:
                if self.halt:raise d.Stop(self.halt)
                if self.calls>=3000:
                    self.stop('service_attempt_cap_3000');raise d.Stop(self.halt)
                if self.successful>=2000:
                    self.stop('successful_cap_2000');raise d.Stop(self.halt)
                boundary=(self.successful//100)!=(self.quota_at//100) or self.quota_at<0
                if self.inflight==0 and time.monotonic()>=self.pause_until and (boundary or self.need_probe):
                    self.quota()
                    boundary=False
                blocked=(time.monotonic()<self.pause_until or self.need_probe or boundary or
                         self.successful+self.inflight>=2000 or
                         self.successful%100+self.inflight>=100)
                if not blocked:
                    self.calls+=1;self.inflight+=1
                    self.journal.add('reserve',call=self.calls,task=ident,config=config,successful=self.successful)
                    return self.calls
            time.sleep(.2)
    def outcome(self,failed,call):
        with self.lock:
            self.inflight-=1
            if failed:
                self.failed+=1;self.pause_until=max(self.pause_until,time.monotonic()+300);self.need_probe=True
            else:self.successful+=1
            self.journal.add('outcome',call=call,failed=failed,successful=self.successful)
            if failed:self.journal.add('pause',reason='service_failure',seconds=300,call=call)

import urllib.request
import urllib.error
from pathlib import Path
class Client:
    def __init__(self, budget, mock=False):
        self.budget, self.mock = budget, mock
        if not mock and not os.environ.get('SOLKEY'):
            raise d.Stop('missing_SOLKEY')
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, messages, trace, context, expected=None):
        ident = Path(trace.f.name).stem
        config = ident.rsplit('_', 1)[0]
        attempt = 0
        while True:
            call = self.budget.reserve(ident, config)
            trace.add('request', call=call, attempt=attempt, model=d.MODEL, messages=messages, **context)
            started = time.monotonic()
            try:
                if self.mock:
                    response = {'model': d.MODEL, 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                                'choices': [{'finish_reason': 'stop', 'message': {'content': d.dump(expected)}}]}
                else:
                    body = d.dump(dict(model=d.MODEL, messages=messages, max_completion_tokens=4096)).encode()
                    req = urllib.request.Request(d.URL, body, {'Content-Type': 'application/json',
                                                            'Authorization': 'Bearer '+os.environ['SOLKEY']})
                    try:
                        with self.opener.open(req, timeout=90) as r:
                            raw = r.read()
                    except urllib.error.HTTPError as e:
                        if 500 <= e.code <= 599:
                            raise d.ServiceError(f'http_{e.code}') from None
                        self.budget.stop(f'http_{e.code}')
                        raise d.Stop(f'http_{e.code}') from None
                    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                        raise d.ServiceError('transport') from None
                    try:
                        response = json.loads(raw)
                    except (ValueError, UnicodeError):
                        self.budget.stop('invalid_api_json')
                        raise d.Stop('invalid_api_json') from None
                trace.add('response', call=call, response=response, seconds=time.monotonic()-started)
                if not isinstance(response, dict) or response.get('model') != d.MODEL:
                    self.budget.stop('model_mismatch')
                    raise d.Stop('model_mismatch')
                u = response.get('usage')
                if not isinstance(u, dict) or any(type(u.get(k)) is not int or u[k] < 0 for k in
                                                ('prompt_tokens', 'completion_tokens', 'total_tokens')):
                    self.budget.stop('missing_usage')
                    raise d.Stop('missing_usage')
                if u['completion_tokens'] > 4096 or u['prompt_tokens'] > 16000:
                    self.budget.stop('token_guard')
                    raise d.Stop('token_guard')
                try:
                    content = response['choices'][0]['message']['content']
                    finish = response['choices'][0]['finish_reason']
                    if not isinstance(content, str) or not isinstance(finish, str):
                        raise ValueError()
                except (ValueError, KeyError, IndexError, TypeError):
                    self.budget.stop('invalid_response_shape')
                    raise d.Stop('invalid_response_shape')
                self.budget.outcome(False, call)
                return response, call
            except d.ServiceError as e:
                trace.add('service_error', call=call, attempt=attempt, reason=str(e), seconds=time.monotonic()-started)
                self.budget.outcome(True, call)
                attempt += 1
        raise AssertionError()

def execute(resume=False,mock=False):
    frozen=verify();out=ROOT/'holdout_c_mock' if mock else OUT
    if resume:
        summary=json.loads((out/'summary.json').read_text())
        assert summary['stage']=='awaiting_human_smoke_review'
        gate=json.loads((out/'SMOKE_APPROVED.json').read_text())
        assert gate.get('approved') is True and gate.get('reviewer') and gate.get('checked_items')
    else:
        out.mkdir(exist_ok=False)
        summary=dict(started_utc=d.now(),phase='holdout_c',mode='mock' if mock else 'real',model=d.MODEL,
                     concurrency=2,cap=2000,service_attempt_cap=3000,schedule=schedule(),
                     frozen_sha256=frozen['sha256'],results=[],stage='smoke')
    budget=Budget(out,mock,resume);client=Client(budget,mock)
    def save():
        summary.update(calls=budget.calls,successful_calls=budget.successful,service_failures=budget.failed,
                       halt=budget.halt,updated_utc=d.now())
        write(out/'summary.json',summary)
    if resume:
        budget.journal.add('smoke_approved',approval=gate,calls=budget.calls)
        summary['stage']='formal';summary['smoke_approval']=gate
    save()
    cursor=len(summary['results']);limit=40 if resume or mock else 2
    with ThreadPoolExecutor(max_workers=2) as pool:
        while cursor<limit and not budget.halt:
            futures=[pool.submit(d.run_task,row,client,out) for row in schedule()[cursor:cursor+2]]
            for fut in as_completed(futures):
                try:result=fut.result()
                except Exception:
                    budget.stop('unexpected_local_exception');save()
                    raise RuntimeError('local failure; details suppressed') from None
                summary['results'].append(result);save();print(d.dump(result),flush=True)
            cursor+=2
    if budget.halt:summary['stage']='stopped'
    elif cursor==2 and not resume and not mock:summary['stage']='awaiting_human_smoke_review'
    else:summary['stage']='finished';summary['finished_utc']=d.now()
    if budget.successful%100==0 and not budget.halt:
        with budget.lock:budget.quota()
    save();budget.journal.f.close()
    print('STAGE',summary['stage'],'SUCCESSFUL',budget.successful,'ATTEMPTS',budget.calls,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','smoke','formal','mock'])
    z=ap.parse_args()
    if z.action=='freeze':freeze()
    else:execute(z.action=='formal',z.action=='mock')
