from unittest.mock import patch
from common import *
guards();sys.path.insert(0,str(RUNNER))
from transport import HTTPTransport,URL
from protocol import Task
results=[];connects=[]
def before_connect(self):
    connects.append(dict(host=self.host,port=self.port));raise RuntimeError('AUDIT_BEFORE_HTTP_CONNECT')
for url in (URL,'http://127.0.0.1:18319@audit.invalid/v1/chat/completions',
            'http://127.0.0.1:18319@audit.invalid:80/v1/chat/completions'):
    count=len(connects)
    with patch('http.client.HTTPConnection.connect',before_connect):
        tr=HTTPTransport(live=True,test_mock_url=url)
        try:tr.send(Task(116064003).body())
        except Exception as ex:result=dict(url=url,error=repr(ex),connection_targets=connects[count:])
        else:raise AssertionError('unexpected network return')
    results.append(result)
save('url_resolution.json',dict(resources=resources(),results=results,note='Real urllib path; only HTTPConnection.connect replaced, no socket or DNS.'))
print(json.dumps(results,ensure_ascii=False))
