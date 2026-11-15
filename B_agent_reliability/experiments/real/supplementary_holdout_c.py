"""Offline engineering checks; never changes frozen C data or scoring."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import urllib.error
import holdout_c as h
import reliability_core as d
import report_holdout_c as report

def check_transport():
    with tempfile.TemporaryDirectory(dir=h.ROOT) as td:
        out=Path(td);budget=h.Budget(out,mock=True);client=h.Client(budget,mock=True);client.mock=False
        trace=d.Journal(out/'sheet_d0_test.jsonl');bodies=[];clock=[1000.]
        class Response:
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def read(self):
                return json.dumps(dict(model=d.MODEL,usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2),choices=[dict(finish_reason='stop',message=dict(content='{}'))])).encode()
        def opened(req,timeout):
            bodies.append(req.data)
            if len(bodies)==1:raise urllib.error.URLError('simulated')
            return Response()
        with patch.object(h.time,'monotonic',side_effect=lambda:clock[0]),patch.object(h.time,'sleep',side_effect=lambda x:clock.__setitem__(0,clock[0]+x)),patch.dict(h.os.environ,{'SOLKEY':'offline-test-placeholder'}),patch.object(client.opener,'open',side_effect=opened):
            client.request([dict(role='user',content='offline')],trace,dict(step=3,review=1))
        assert bodies[0]==bodies[1] and clock[0]>=1300
        assert (budget.calls,budget.successful,budget.failed)==(2,1,1)
        trace.f.close();budget.journal.f.close()

def replay(out):
    summary=json.loads((out/'summary.json').read_text());checks=0
    for result in summary['results']:
        ev=report.events(out/(result['id']+'.jsonl'))
        responses={x['call']:x['response'] for x in ev if x['event']=='response'}
        requests=[x for x in ev if x['event']=='request' and x['call'] in responses]
        class Replay:
            def __init__(self):self.i=0;self.budget=None
            def request(self,messages,trace,context,expected=None):
                x=requests[self.i];self.i+=1
                assert messages==x['messages'] and all(x[k]==v for k,v in context.items())
                return responses[x['call']],x['call']
        client=Replay()
        with tempfile.TemporaryDirectory(dir=h.ROOT) as td:
            got=d.run_task({k:result[k] for k in ('family','level','knob','seed','length')},client,Path(td))
            assert got==result and client.i==len(requests)
        checks+=len(requests)
    return checks

if __name__=='__main__':
    check_transport()
    print('transport identical-body/global-pause/accounting PASS')
    if (h.OUT/'summary.json').exists():print('full prompt replay',replay(h.OUT))
