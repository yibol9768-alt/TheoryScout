"""默认 mock 协议服务器不建 socket；HTTP 仅显式 live=True 才构造。"""
import json, os, time, urllib.request, urllib.error
from protocol import MODEL, dump, oracle
from evidence import require_live_release
URL='http://127.0.0.1:18319/v1/chat/completions'
class HTTPTransport:
    def __init__(self,live=False,*,test_mock_url=None):
        if live is not True:raise ValueError('network_requires_explicit_--live')
        self.url=URL if test_mock_url is None else test_mock_url
        if test_mock_url is not None and not test_mock_url.startswith('http://127.0.0.1:'):raise ValueError('mock_must_be_loopback')
        self.testing=test_mock_url is not None
        if not self.testing:require_live_release(live)
        if not self.testing and not os.environ.get('SOLKEY'):raise ValueError('missing_SOLKEY')
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs):return None
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    def send(self,body,*,engine=None,reservation=None):
        if not self.testing:
            require_live_release(True)
            if engine is None or reservation is None or engine.mode!='live':
                raise ValueError('live_transport_requires_engine_reservation')
            with engine.lock:
                engine._send_gate(reservation['seed'])
                call=reservation['call']
                from protocol import sha
                if (engine.pending.get(call)!=reservation or call not in engine.active
                        or reservation['body_sha256']!=sha(body)):
                    raise ValueError('live_transport_reservation_mismatch')
        headers={'Content-Type':'application/json'}
        if not self.testing:headers['Authorization']='Bearer '+os.environ['SOLKEY']
        req=urllib.request.Request(self.url,body,headers)
        try:
            with self.opener.open(req,timeout=90) as r:return r.status,r.read(),None
        except urllib.error.HTTPError as e:
            # Persist error body but never authorization or response headers.
            return e.code,e.read(),'http'
        except (urllib.error.URLError,TimeoutError,ConnectionError,OSError):return 0,b'','connection_or_timeout'

class HTTPQuota:
    def __init__(self,url,live=False):
        if live is not True:raise ValueError('quota_network_requires_--live')
        require_live_release(live)
        if not url.startswith('http://127.0.0.1:18319/') or '/chat/completions' in url:raise ValueError('quota_must_be_separate_local_endpoint')
        self.url=url
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs):return None
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    def __call__(self):
        require_live_release(True)
        # This is an adapter contract, NOT a claim that the local service provides it.
        # A fresh separately validated provider is mandatory; no configured provider => pause.
        req=urllib.request.Request(self.url,headers={'Authorization':'Bearer '+os.environ['SOLKEY'],'Cache-Control':'no-cache'})
        with self.opener.open(req,timeout=50) as r:q=json.loads(r.read())
        if type(q.get('observed_unix')) not in (float,int) or not 0<=time.time()-q['observed_unix']<=60:raise ValueError('quota_not_fresh')
        return q

class MockServer:
    """内存协议服务器：接收同一序列化HTTP正文，返回(status,raw,reason)。"""
    def __init__(self,outcomes=None,policy='correct'):
        self.outcomes=list(outcomes or []);self.bodies=[];self.policy=policy;self.times=[]
    def send(self,body):
        self.bodies.append(body);self.times.append(time.time())
        if self.outcomes:
            x=self.outcomes.pop(0)
            if isinstance(x,BaseException):raise x
            if x=='connection':return 0,b'','connection'
            if x=='timeout':return 0,b'','timeout'
            if type(x) is int:return x,b'{"mock_error":true}','http'
            return x
        req=json.loads(body);p=json.loads(req['messages'][1]['content']);rv=len(req['messages'])==4
        expected=oracle(p['checkpoint'],p['operation'])
        content=dump(dict(writes=expected,detect=rv))
        if self.policy=='fatal_at_8' and p['step']==8:content='not-json'
        response=dict(model=MODEL,usage=dict(prompt_tokens=400,completion_tokens=80,total_tokens=480),
                      choices=[dict(message=dict(role='assistant',content=content),finish_reason='stop')])
        return 200,dump(response).encode(),None
