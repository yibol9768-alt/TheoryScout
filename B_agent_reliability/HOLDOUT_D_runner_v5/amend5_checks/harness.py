import ast, io, json, os, sys, threading, types, unittest
from pathlib import Path
from email.message import Message
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'results'
def resources():
    rel=Path('/proc/self/cgroup').read_text().strip().split('0::')[-1]
    cg=Path('/sys/fs/cgroup')/rel.lstrip('/')
    z=dict(cgroup=rel,memory_max=(cg/'memory.max').read_text().strip(),swap_max=(cg/'memory.swap.max').read_text().strip(),threads=len(list(Path('/proc/self/task').iterdir())))
    assert z['memory_max']=='2147483648' and z['swap_max']=='0' and z['threads']<=2,z
    return z
thread_counts=[]
_start=threading.Thread.start
def start(self,*a,**k):
    z=_start(self,*a,**k);n=len(list(Path('/proc/self/task').iterdir()));thread_counts.append(n);assert n<=2,n;return z
threading.Thread.start=start
http_records=[]
class Response(io.BytesIO):
    def __init__(self,status,body,headers=None):
        super().__init__(body);self.status=self.code=status;self.reason='D4_LOCAL_STUB';self.msg=self.headers=Message()
        for k,v in (headers or {}).items():self.headers[k]=v
    def info(self):return self.headers
    def getheader(self,k,default=None):return self.headers.get(k,default)
def stub_request(self,method,url,body=None,headers=None,**kw):
    self._d4=(method,url,body,headers);http_records.append(dict(host=self.host,port=self.port,method=method,path=url))
def stub_getresponse(self):
    method,path,body,headers=self._d4
    from transport import MockServer
    status=302 if path=='/redirect' else 408 if path=='/408' else 200
    return Response(status,MockServer().send(body)[1],{'Location':'http://127.0.0.1:18319/NEVER'} if status==302 else {})
import http.client
http.client.HTTPConnection.request=stub_request
http.client.HTTPConnection.getresponse=stub_getresponse
def engine_module(runner):
    sys.path.insert(0,str(runner))
    source=(runner/'engine.py').read_text().replace("ROOT/"+repr(runner.name),'Path('+repr(str(OUT))+')')
    m=types.ModuleType('engine');m.__file__=str(runner/'engine.py');sys.modules['engine']=m
    exec(compile(source,m.__file__,'exec'),m.__dict__);return m
def test_module(runner,base):
    path=runner/'test_mock.py';tree=ast.parse(path.read_text(),str(path))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE' for t in node.targets):node.value=ast.parse(repr(str(base)),mode='eval').body;node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[node.value],keywords=[])
        if isinstance(node,ast.FunctionDef) and node.name=='http_server':
            # Preserve HTTPTests assertions and fork lifecycle; replace listener with a port-only stub.
            node.body=ast.parse('pipe.send(49152)').body
    ast.fix_missing_locations(tree)
    m=types.ModuleType('test_mock');m.__file__=str(path);sys.modules['test_mock']=m
    exec(compile(tree,str(path),'exec'),m.__dict__);return m
def install_common(destination):
    runner=ROOT/'HOLDOUT_D_runner_v5';path=runner/'audit_regression/common.py'
    sys.path.append(str(runner/'audit_regression'))
    m=types.ModuleType('common');m.__file__=str(path);sys.modules['common']=m;exec(compile(path.read_text(),str(path),'exec'),m.__dict__)
    m.ROOT=ROOT;m.OUT=destination;m.RUNNER=runner;m.resources=resources;m.guards=lambda **kw:None
    def modules():return engine_module(m.RUNNER)
    m.modules=modules
    return m
def suite(version):
    runner=ROOT/('HOLDOUT_D_runner' if version=='v1' else 'HOLDOUT_D_runner_'+version)
    engine_module(runner);m=test_module(runner,OUT/(version+'_tests'))
    mods=[m]
    if version!='v1':mods.append(__import__('test_amend2'))
    if version in ('v3','v4','v5'):mods.append(__import__('test_amend3'))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(x) for x in mods))
    z=dict(tests=result.testsRun,passed=result.wasSuccessful(),failures=[(str(t),e) for t,e in result.failures],errors=[(str(t),e) for t,e in result.errors],resources=resources(),thread_counts=thread_counts,http_stub_records=http_records)
    (OUT/(version+'_results.json')).write_text(json.dumps(z,indent=2,ensure_ascii=False)+'\n');return 0 if result.wasSuccessful() else 1
if __name__=='__main__':
    resources();kind=sys.argv[1]
    if kind=='suite':raise SystemExit(suite(sys.argv[2]))
    original=kind=='original';name=sys.argv[2];dest=OUT/(('original_' if original else 'ported_')+name.removesuffix('.py'));dest.mkdir()
    common=install_common(dest)
    # Child entry shims also execute this wrapper, retaining global write/network guards.
    (dest/'entry_sentinel.py').write_text('import sys\nsys.path.insert(0,'+repr(str(Path(__file__).resolve().parent))+')\nimport harness\nharness.install_common(__import__("pathlib").Path(__file__).parent)\nexec(compile(harness.ROOT.joinpath("audit_d3_checks/entry_sentinel.py").read_text(),"entry_sentinel.py","exec"))\n')
    (dest/'sitecustomize.py').write_text((Path(__file__).resolve().parent/'sitecustomize.py').read_text().replace("OUT=Path(__file__).resolve().parent/'results'",'OUT=Path('+repr(str(OUT))+')'))
    if not original:
        engine_module(common.RUNNER)
        test_module(common.RUNNER,dest/'all_tests')
    path=ROOT/('audit_d3_checks' if original else 'HOLDOUT_D_runner_v5/audit_regression')/name
    source=path.read_text()
    if not original and name=='attack_entries.py':
        source=source.replace("str(RUNNER/'audit_regression/entry_sentinel.py')","str(OUT/'entry_sentinel.py')").replace("str(RUNNER/'audit_regression')","str(OUT)")
    if not original and name=='check_scores.py':
        source=source.replace("base=OUT/'all_tests'",'base=Path('+repr(str(OUT/'v5_tests'))+')')
    namespace={'__name__':'__main__','__file__':str(path)}
    try:exec(compile(source,str(path),'exec'),namespace)
    except Exception as ex:
        if not original:raise
        expected={'attack_entries.py':'test_url_forbidden_in_production_transport','endpoint_escape.py':'test_url_forbidden_in_production_transport','url_resolution.py':'test_url_forbidden_in_production_transport','repeated_dispatch.py':'dispatch_already_started'}
        assert str(ex)==expected[name],repr(ex)
        z=dict(script=str(path.relative_to(ROOT)),source_sha256=__import__('hashlib').sha256(path.read_bytes()).hexdigest(),expected_v3_rejection=str(ex),resources=resources(),http_stub_records=http_records,assertions_unchanged=True,
            cli_cases=len(namespace.get('results',[])) if name=='attack_entries.py' else None,opens=namespace.get('opens',namespace.get('reached',[])),attempts=getattr(namespace.get('e'),'attempts',None))
        (dest/'rejection.json').write_text(json.dumps(z,indent=2)+'\n');print(json.dumps(z))
    else:
        if original:raise AssertionError('Old exploit unexpectedly completed')
