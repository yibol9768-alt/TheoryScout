"""Independent D4 boundary tests. All HTTP is recorded locally; no sockets."""
import json,multiprocessing,os,threading,unittest,urllib.error,urllib.request
from unittest.mock import patch
from harness import ROOT,OUT,resources,engine_module,http_records,thread_counts,Response
engine=engine_module(ROOT/'HOLDOUT_D_runner_v4')
from transport import HTTPTransport,TestHTTPTransport,MockServer,URL
from protocol import Task,sha
from contextlib import ExitStack
S=116064001
def good():return dict(five_hour=0.,week=0.)
def projection():
    stack=ExitStack()
    stack.enter_context(patch.object(engine.Engine,'mode',new=property(lambda self:'live')))
    stack.enter_context(patch('engine.require_live_release',return_value=None))
    stack.enter_context(patch('transport.require_live_release',return_value=None))
    stack.enter_context(patch.dict(os.environ,{'SOLKEY':'D4_OFFLINE_FIXTURE'}))
    return stack
def tr(e,calls,status=200):
    class Stub:
        def open(self,request,timeout):
            ds=[x for x in map(json.loads,e.wal.path.read_text().splitlines()) if x['kind']=='dispatch_started']
            assert len([x for x in ds if x['body_sha256']==sha(request.data)])==1
            calls.append(request.full_url)
            return Response(status,MockServer().send(request.data)[1])
    t=object.__new__(HTTPTransport);t.url=URL;t.opener=Stub();return t
class Boundary(unittest.TestCase):
    def eng(self,name,**kw):return engine.Engine(OUT/'boundary'/name,verify=False,**kw)
    def test_01_ipv6_dns_numeric_aliases_reject_before_opener(self):
        hosts=['[::1]','[::ffff:127.0.0.1]','localhost','LOCALHOST','localhost.','127.1','127.0.1','2130706433','0x7f000001','0177.0.0.1','127.000.000.001','127.0.0.2','127.0.0.1.','127%2e0%2e0%2e1']
        with patch('urllib.request.build_opener',side_effect=AssertionError('OPENER_REACHED')):
            for host in hosts:
                with self.subTest(host=host):
                    with self.assertRaises(ValueError):TestHTTPTransport('http://'+host+':49152/x')
    def test_02_production_ports_and_authority_grammar(self):
        urls=[f'http://127.0.0.1:{p}/x' for p in (0,80,443,18319,49151,65536)]
        urls+=['http://u:p@127.0.0.1:49152/x','http://127.0.0.1:49152@elsewhere.invalid/x','http://127.0.0.1:49152/x#fragment','http://127.0.0.1:49152/x\n','https://127.0.0.1:49152/x']
        with patch('urllib.request.build_opener',side_effect=AssertionError('OPENER_REACHED')):
            for url in urls:
                with self.subTest(url=url):
                    with self.assertRaises(ValueError):TestHTTPTransport(url)
                    with self.assertRaises(ValueError):HTTPTransport(live=True,test_mock_url=url)
    def test_03_proxy_environment_cannot_change_destination(self):
        keys=['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']
        for transport_kind in ('test','production'):
            destinations=[]
            for dirty in (False,True):
                env={k:'http://proxy.invalid:18319' for k in keys} if dirty else {}
                env.update(NO_PROXY='',no_proxy='')
                with patch.dict(os.environ,env,clear=True),projection():
                    before=len(http_records)
                    if transport_kind=='test':
                        t=TestHTTPTransport('http://127.0.0.1:49152/ok');self.assertEqual(t.send(Task(S).body())[0],200)
                    else:
                        e=self.eng('proxy_'+str(dirty));r,b=e.reserve(S,good);t=HTTPTransport(live=True);self.assertEqual(t.send(b,engine=e,reservation=r)[0],200);e.close()
                    destinations.append(http_records[before:])
            self.assertEqual(destinations[0],destinations[1]);self.assertEqual(len(destinations[0]),1)
    def test_04_all_redirect_codes_never_issue_second_http(self):
        from harness import stub_getresponse
        for status in (301,302,303,307,308):
            for target in ('http://127.0.0.1:18319/NEVER','https://elsewhere.invalid/NEVER'):
                for production in (False,True):
                    def redirect(conn):return Response(status,b'D4_REDIRECT',{'Location':target})
                    with patch('http.client.HTTPConnection.getresponse',redirect),projection():
                        before=len(http_records)
                        if production:
                            e=self.eng(f'redirect_{status}_{production}_{sha(target.encode())[:8]}');r,b=e.reserve(S,good);t=HTTPTransport(live=True)
                            code,raw,reason=t.send(b,engine=e,reservation=r);e.receive(r['call'],code,raw,reason)
                            self.assertEqual((e.halt,e.attempts,e.success),('other_http',1,0));e.close()
                        else:code,raw,reason=TestHTTPTransport('http://127.0.0.1:49152/start').send(Task(S).body())
                        self.assertEqual((code,reason),(status,'http'));self.assertEqual(len(http_records)-before,1)
    def test_05_two_transport_instances_share_atomic_send_right(self):
        for iteration in range(12):
            e=self.eng('race_'+str(iteration));calls=[];a,b=tr(e,calls),tr(e,calls);barrier=threading.Barrier(2);result=[]
            with projection():
                r,body=e.reserve(S,good)
                def send(t):
                    barrier.wait()
                    try:t.send(body,engine=e,reservation=r);result.append('sent')
                    except engine.Paused as ex:result.append(str(ex))
                worker=threading.Thread(target=send,args=(a,));worker.start();send(b);worker.join(10)
                self.assertFalse(worker.is_alive());self.assertCountEqual(result,['sent','dispatch_already_started'])
                self.assertEqual((len(calls),e.attempts,len(e.pending)),(1,1,1))
            e.close()
    def test_06_crash_after_fsync_before_request_construction(self):
        path=OUT/'boundary'/'crash_before_Request'
        def child():
            e=engine.Engine(path,verify=False)
            with projection():
                r,b=e.reserve(S,good)
                with patch('urllib.request.Request',side_effect=lambda *a,**k:os._exit(91)):
                    tr(e,[]).send(b,engine=e,reservation=r)
        p=multiprocessing.get_context('fork').Process(target=child);p.start();p.join(10);self.assertEqual(p.exitcode,91)
        for _ in range(3):
            e=engine.Engine(path,verify=False);calls=[]
            self.assertEqual((e.attempts,e.success,len(e.pending),len(e.active),e.dispatched,e.halt),(1,0,1,0,{1},'unknown_inflight'))
            with projection():
                with self.assertRaises(engine.Paused):tr(e,calls).send(Task(S).body(),engine=e,reservation=e.pending[1])
                with self.assertRaises(engine.Paused):e.reserve(S,good)
            self.assertEqual(calls,[]);e.close()
    def test_07_crash_before_dispatch_retains_unknown_reserve(self):
        path=OUT/'boundary'/'crash_before_claim'
        def child():
            e=engine.Engine(path,verify=False);e.reserve(S,good);os._exit(93)
        p=multiprocessing.get_context('fork').Process(target=child);p.start();p.join(10);self.assertEqual(p.exitcode,93)
        e=engine.Engine(path,verify=False);self.assertEqual((e.attempts,len(e.pending),e.dispatched,e.halt),(1,1,set(),'unknown_inflight'))
        with self.assertRaises(engine.Paused):e.claim_dispatch(e.pending[1],Task(S).body())
        e.close()
    def test_08_body_mismatch_does_not_consume_valid_send_right(self):
        e=self.eng('body_mismatch');calls=[]
        with projection():
            r,b=e.reserve(S,good);t=tr(e,calls)
            with self.assertRaisesRegex(ValueError,'reservation_mismatch'):t.send(b+b' ',engine=e,reservation=r)
            self.assertEqual(e.dispatched,set());t.send(b,engine=e,reservation=r)
        self.assertEqual(len(calls),1);e.close()
    def test_09_dispatch_write_failure_before_bytes_never_resends(self):
        e=self.eng('write_failure');path=e.wal.out;calls=[]
        with projection():
            r,b=e.reserve(S,good)
            with patch.object(e.wal,'add',side_effect=OSError('D4_NO_BYTES_WRITTEN')):
                with self.assertRaises(OSError):tr(e,calls).send(b,engine=e,reservation=r)
            with self.assertRaises(engine.Paused):tr(e,calls).send(b,engine=e,reservation=r)
        self.assertEqual(calls,[]);e.close();e=engine.Engine(path,verify=False)
        self.assertEqual((len(e.pending),e.dispatched,e.halt),(1,set(),'unknown_inflight'));e.close()
    def test_10_production_url_mutation_rejects_before_claim(self):
        e=self.eng('production_mutation');calls=[]
        with projection():
            r,b=e.reserve(S,good);t=tr(e,calls);t.url='http://127.0.0.1:49152/ok'
            with self.assertRaisesRegex(ValueError,'endpoint_is_fixed'):t.send(b,engine=e,reservation=r)
        self.assertEqual((calls,e.dispatched),([],set()));e.close()
    def test_11_reserved_unsent_request_must_obey_new_service_pause(self):
        findings=[]
        for status in (408,500,503,599):
            e=self.eng('pause_direct_'+str(status),clock=lambda:1000.);calls=[]
            with projection():
                r1,b1=e.reserve(S,good);r2,b2=e.reserve(S+1,good)
                t=tr(e,calls,status);code,raw,reason=t.send(b1,engine=e,reservation=r1);e.receive(r1['call'],code,raw,reason)
                before=len(calls);error=None
                try:tr(e,calls).send(b2,engine=e,reservation=r2)
                except (engine.Paused,ValueError) as ex:error=str(ex)
                findings.append(dict(status=status,clock=e.clock(),pause_until=e.pause_until,need_probe=e.need_probe,attempts=e.attempts,success=e.success,pending=list(e.pending),dispatched=sorted(e.dispatched),sends_during_pause=len(calls)-before,error=error))
            e.close()
        (OUT/'pause_direct_findings.json').write_text(json.dumps(findings,indent=2)+'\n')
        self.assertEqual([x['sends_during_pause'] for x in findings],[0]*len(findings),'pre-reserved unsent HTTP must not start during global service pause')
    def test_12_normal_request_thread_interleaving_must_obey_pause(self):
        e=self.eng('pause_normal_request',clock=lambda:1000.);calls=[];reserved=threading.Event();release=threading.Event();errors=[]
        slow=tr(e,calls);fast=tr(e,calls,408)
        class Delayed:
            testing=False
            def send(self,body,**kw):
                reserved.set();assert release.wait(10),'D4_THREAD_DEADLINE'
                return slow.send(body,**kw)
        with projection():
            def worker():
                try:e.request(S+1,Delayed(),good)
                except BaseException as ex:errors.append(repr(ex))
            thread=threading.Thread(target=worker);thread.start();self.assertTrue(reserved.wait(10))
            try:
                e.request(S,fast,good);before=len(calls)
            finally:release.set()
            thread.join(10);self.assertFalse(thread.is_alive())
        finding=dict(clock=e.clock(),pause_until=e.pause_until,need_probe=e.need_probe,attempts=e.attempts,successful=e.success,failed=e.failed,sends_during_pause=len(calls)-before,worker_errors=errors,dispatched=sorted(e.dispatched),quota_events=sum(x['kind']=='quota' for x in e.wal.events),halt=e.halt,wal=str(e.wal.path.relative_to(ROOT)))
        (OUT/'pause_normal_findings.json').write_text(json.dumps(finding,indent=2)+'\n');e.close()
        self.assertEqual(finding['sends_during_pause'],0,'normal Engine.request can be descheduled between reserve and production send')
if __name__=='__main__':
    resources();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Boundary))
    z=dict(tests=result.testsRun,passed=result.wasSuccessful(),errors=[(str(t),e) for t,e in result.errors],failures=[(str(t),e) for t,e in result.failures],resources=resources(),thread_counts=thread_counts,http_stub_records=http_records)
    (OUT/'boundary_results.json').write_text(json.dumps(z,indent=2,ensure_ascii=False)+'\n');raise SystemExit(0 if result.wasSuccessful() else 1)
