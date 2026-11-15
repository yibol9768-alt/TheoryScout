"""默认 mock 协议服务器不建 socket；HTTP 仅显式 live=True 才构造。"""
import json, os, subprocess, time, urllib.request, urllib.error
from urllib.parse import urlsplit
from protocol import MODEL, dump, oracle
from evidence import require_live_release
URL='http://127.0.0.1:18319/v1/chat/completions'
class HTTPTransport:
    testing=False
    def __init__(self,live=False,*,test_mock_url=None):
        # 兼容参数仅用于明确拒绝旧攻击入口，绝不授予网络豁免。
        if test_mock_url is not None:raise ValueError('test_url_forbidden_in_production_transport')
        if live is not True:raise ValueError('network_requires_explicit_--live')
        self.url=URL
        require_live_release(live)
        if not os.environ.get('SOLKEY'):raise ValueError('missing_SOLKEY')
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs):return None
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    def send(self,body,*,engine=None,reservation=None):
        require_live_release(True)
        if engine is None or reservation is None or engine.mode!='live':
            raise ValueError('live_transport_requires_engine_reservation')
        if self.url!=URL:raise ValueError('production_endpoint_is_fixed')
        # claim_dispatch 的 fsync 成功返回后才能构造 Request/进入 opener。
        engine.claim_dispatch(reservation,body)
        headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['SOLKEY']}
        req=urllib.request.Request(URL,body,headers)
        try:
            with self.opener.open(req,timeout=90) as r:return r.status,r.read(),None
        except urllib.error.HTTPError as e:
            # Persist error body but never authorization or response headers.
            return e.code,e.read(),'http'
        except (urllib.error.URLError,TimeoutError,ConnectionError,OSError):return 0,b'','connection_or_timeout'

class TestHTTPTransport:
    """独立测试客户端：只允许 49152–65535 的 IPv4 loopback HTTP mock。"""
    testing=True
    @staticmethod
    def validate_url(url):
        if type(url) is not str or any(c.isspace() for c in url):raise ValueError('invalid_test_url')
        p=urlsplit(url)
        if (p.scheme!='http' or p.hostname!='127.0.0.1' or p.username is not None
                or p.password is not None or p.port is None or not 49152<=p.port<=65535
                or p.fragment):raise ValueError('test_endpoint_forbidden')
        return url
    def __init__(self,url):
        self.url=self.validate_url(url)
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs):return None
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    def send(self,body):
        url=self.validate_url(self.url)  # 属性改变也不能扩大测试网络边界。
        req=urllib.request.Request(url,body,{'Content-Type':'application/json'})
        try:
            with self.opener.open(req,timeout=90) as r:return r.status,r.read(),None
        except urllib.error.HTTPError as e:return e.code,e.read(),'http'
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

class CommandQuota:
    """固定项目命令；stdout 为五小时/周已用百分比，不接受自定义命令。"""
    def __init__(self,live=False):
        if live is not True:raise ValueError('quota_command_requires_--live')
        require_live_release(live)
    def __call__(self):
        require_live_release(True)
        z=subprocess.run(['python3','/home/liuyibo/auto-llm-driving/TheoryScout/tools/plus_quota.py'],
                         capture_output=True,text=True,timeout=50,check=True)
        a,b=map(float,z.stdout.split())
        if not (0<=a<=100 and 0<=b<=100):raise ValueError('quota_command_percent_out_of_range')
        return dict(five_hour=a/100,week=b/100,observed_unix=time.time())

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
