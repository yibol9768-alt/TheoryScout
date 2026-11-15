"""全部调用发往 mock 服务器；默认测试不建socket；唯一HTTP集成指向fork本地mock。"""
import base64, contextlib, json, multiprocessing, os, socket, threading, time, unittest
from pathlib import Path
from engine import Engine, Paused, IntegrityError, WAL
from protocol import ROOT, Task, dump, sha, guard, GuardError, _ns
from transport import MockServer, HTTPTransport, TestHTTPTransport
from run import execute
BASE=ROOT/'HOLDOUT_D_runner_v3'/('HOLDOUT_D_TEST_'+str(time.time_ns()))
BASE.mkdir()
S=116064001
class Clock:
    def __init__(self):self.t=2000000000.
    def __call__(self):return self.t
    def advance(self,x):self.t+=x

def mock_response(body):return MockServer().send(body)[1]
class Tests(unittest.TestCase):
    def eng(self,name,fixture=None,clock=None):return Engine(BASE/name,verify=False,fixture=fixture,clock=clock or Clock())
    def good(self):return dict(five_hour=0.,week=0.)
    def test_8999_atomic_concurrent(self):
        e=self.eng('8999',dict(success=8999));barrier=threading.Barrier(2);out=[]
        def reserve(seed):
            barrier.wait()
            try:out.append(e.reserve(seed,self.good))
            except Paused:pass
        thread=threading.Thread(target=reserve,args=(S,));thread.start();reserve(S+1);thread.join()
        self.assertEqual(len(out),1);self.assertEqual(e.success+len(e.pending),9000)
        req,body=out[0];e.receive(req['call'],200,mock_response(body));self.assertEqual(e.success,9000)
        with self.assertRaises(Paused):e.reserve(S+1,self.good)
        e.close();e=self.eng('8999');self.assertEqual(e.success,9000);e.close()
    def test_100_boundary(self):
        e=self.eng('100',dict(success=98));probes=[]
        def probe():probes.append((e.success,len(e.active)));return self.good()
        r1,b1=e.reserve(S,probe);r2,b2=e.reserve(S+1,probe)
        e.receive(r1['call'],200,mock_response(b1))
        with self.assertRaises(Paused):e.reserve(S,probe)
        self.assertEqual(probes,[(98,0)])
        e.receive(r2['call'],200,mock_response(b2));r3,_=e.reserve(S,probe)
        self.assertEqual(probes,[(98,0),(100,0)])
        self.assertEqual(e.success,100);e.close()
    def test_12000_attempts(self):
        e=self.eng('12000',dict(success=0,attempts=11999));r,b=e.reserve(S,self.good)
        self.assertEqual(r['call'],12000);e.receive(r['call'],503,b'mock')
        with self.assertRaises(Paused):e.reserve(S,self.good)
        self.assertEqual(e.success,0);e.close();e=self.eng('12000');self.assertEqual(e.attempts,12000);e.close()
    def test_concurrency_two_and_no_duplicate_logical_request(self):
        e=self.eng('concurrency')
        e.reserve(S,self.good)
        with self.assertRaises(Paused):e.reserve(S,self.good)
        e.reserve(S+1,self.good)
        with self.assertRaises(Paused):e.reserve(S+2,self.good)
        self.assertEqual(len(e.active),2);e.close()
    def test_quota_failure_exactly_at_hundred_boundary(self):
        c=Clock();e=self.eng('quota_100',dict(success=99),clock=c)
        r,b=e.reserve(S,self.good);e.receive(r['call'],200,mock_response(b))
        with self.assertRaises(Paused):e.reserve(S+1,lambda:dict(five_hour=.80,week=0))
        self.assertEqual(e.attempts,100);self.assertFalse(e.pending)
        with self.assertRaises(Paused):e.reserve(S+1,self.good)
        c.advance(300);e.reserve(S+1,self.good);self.assertEqual(e.attempts,101);e.close()
    def test_all_service_errors_wait_and_identical_retry(self):
        for error in ['connection','timeout',408,500,501,502,503,504,599]:
            with self.subTest(error=error):
                c=Clock();e=self.eng('service_'+str(error),clock=c);server=MockServer([error])
                e.request(S,server,self.good);self.assertEqual((e.success,e.attempts),(0,1))
                with self.assertRaises(Paused):e.reserve(S+1,self.good)
                c.advance(299.999)
                with self.assertRaises(Paused):e.reserve(S,self.good)
                c.advance(.001);e.request(S,server,self.good)
                self.assertEqual(server.bodies[0],server.bodies[1]);self.assertEqual(e.requests[2]['rid'],e.requests[1]['rid'])
                self.assertEqual(e.requests[2]['attempt'],2);self.assertEqual((e.success,e.attempts),(1,2));e.close()
    def test_last_failure_extends_and_drains(self):
        c=Clock();e=self.eng('two_failures',clock=c);r1,b=e.reserve(S,self.good);r2,b=e.reserve(S+1,self.good)
        e.receive(r1['call'],408);c.advance(10)
        with self.assertRaises(Paused):e.reserve(S,self.good)
        e.receive(r2['call'],599);self.assertEqual(e.pause_until,c()+300);c.advance(299)
        with self.assertRaises(Paused):e.reserve(S,self.good)
        c.advance(1);e.reserve(S,self.good);e.close()
    def test_quota_failures_thresholds_and_recovery(self):
        for label,q in [('five',dict(five_hour=.80,week=0)),('week',dict(five_hour=0,week=.90)),('failed',None),('nan',dict(five_hour=float('nan'),week=0))]:
            c=Clock();e=self.eng('quota_'+label,clock=c)
            def bad():
                if q is None:raise ValueError('mock quota unavailable')
                return q
            with self.assertRaises(Paused):e.reserve(S,bad)
            e.close();e=self.eng('quota_'+label,clock=c)
            with self.assertRaises(Paused):e.reserve(S,self.good)
            c.advance(299)
            with self.assertRaises(Paused):e.reserve(S,self.good)
            self.assertEqual(e.attempts,0);c.advance(1);e.reserve(S,self.good);e.close()
    def test_recovery_requires_new_quota_no_stale(self):
        e=self.eng('quota_recovery');e.quota(self.good);e.close();e=self.eng('quota_recovery')
        def fail():raise ValueError()
        with self.assertRaises(Paused):e.reserve(S,fail)
        self.assertEqual(e.attempts,0);e.close()
    def test_response_action_crash_and_two_recoveries(self):
        out='response_crash';e=self.eng(out);r,b=e.reserve(S,self.good);raw=mock_response(b)
        with self.assertRaises(SystemExit):e.receive(r['call'],200,raw,crash_after_response=True)
        before=e.wal.path.read_bytes();e.close()
        # Deliberately stale, fabricated summary must never affect replay.
        (BASE/out/'HOLDOUT_D_SUMMARY.json').write_text('{"successful":99999}')
        for i in range(2):
            e=self.eng(out);self.assertTrue(e.wal.path.read_bytes().startswith(before));self.assertEqual((e.success,e.attempts,e.tasks[S].step),(1,1,1))
            acts=[x for x in e.wal.events if x['kind']=='action'];self.assertEqual(len(acts),1)
            self.assertEqual(acts[0]['record']['after'],e.tasks[S].state);self.assertFalse(e.pending);e.close()
    def test_actual_process_exit_after_response_fsync(self):
        path=BASE/'process_crash'
        def child():
            e=Engine(path,verify=False,clock=Clock());r,b=e.reserve(S,self.good)
            try:e.receive(r['call'],200,mock_response(b),crash_after_response=True)
            except SystemExit:os._exit(91)  # No close/finally/summary; OS releases flock.
        p=multiprocessing.get_context('fork').Process(target=child);p.start();p.join(10)
        if p.is_alive():p.terminate();p.join();self.fail('crash child stuck')
        self.assertEqual(p.exitcode,91)
        for _ in range(2):
            e=Engine(path,verify=False,clock=Clock());self.assertEqual((e.success,e.attempts,e.tasks[S].step),(1,1,1))
            self.assertEqual(len([x for x in e.wal.events if x['kind']=='action']),1);e.close()
    def test_unknown_inflight_never_release_or_resend(self):
        e=self.eng('unknown');r,b=e.reserve(S,self.good);e.close()
        for _ in range(2):
            e=self.eng('unknown');self.assertEqual(e.success+len(e.pending),1)
            with self.assertRaises(Paused):e.reserve(S,self.good)
            self.assertEqual(e.attempts,1);e.close()
    def test_verified_raw_reconcile(self):
        e=self.eng('reconcile');r,b=e.reserve(S,self.good);e.close();e=self.eng('reconcile');raw=mock_response(b)
        proof=BASE/'HOLDOUT_D_mock_server_export.json';proof.write_text(dump(dict(rid=r['rid'],body_sha256=sha(b),raw=base64.b64encode(raw).decode(),raw_sha256=sha(raw),status=200,verified_by='MOCK ONLY',source='mock server journal')))
        e.reconcile_response(r['call'],proof);self.assertEqual((e.success,e.tasks[S].step),(1,1));e.close()
        e=self.eng('reconcile');self.assertEqual(e.tasks[S].step,1);self.assertFalse(e.pending);e.close()
    def test_transport_retry_across_two_recoveries(self):
        c=Clock();e=self.eng('retry_recovery',clock=c);server=MockServer([408]);e.request(S,server,self.good);body=server.bodies[0];e.close()
        for _ in range(2):
            e=self.eng('retry_recovery',clock=c)
            self.assertEqual(e.tasks[S].body(),body)
            with self.assertRaises(Paused):e.reserve(S,self.good)
            self.assertEqual(e.attempts,1);e.close()
        c.advance(300);e=self.eng('retry_recovery',clock=c);e.request(S,server,self.good)
        self.assertEqual(server.bodies[-1],body);self.assertEqual(e.requests[2]['attempt'],2);e.close()
    def test_guards_count_success_and_global_stop(self):
        for label,change in [('model',lambda z:z.update(model='other')),('usage',lambda z:z.pop('usage')),('tokens',lambda z:z['usage'].update(prompt_tokens=16001)),('envelope',lambda z:z.update(choices=[]))]:
            e=self.eng('guard_'+label);r,b=e.reserve(S,self.good);z=json.loads(mock_response(b));change(z);e.receive(r['call'],200,dump(z).encode())
            self.assertEqual(e.success,1);self.assertIsNotNone(e.halt);self.assertEqual(e.tasks[S].step,0)
            with self.assertRaises(Paused):e.reserve(S+1,self.good)
            e.close()
        e=self.eng('invalid_json');r,b=e.reserve(S,self.good);e.receive(r['call'],200,b'badjson');self.assertEqual(e.success,1);self.assertTrue(e.halt);e.close()
    def test_other_http_safely_stops(self):
        for status in (301,400,401,429):
            e=self.eng('http_'+str(status));r,b=e.reserve(S,self.good);e.receive(r['call'],status,b'mock')
            self.assertEqual(e.success,0);self.assertEqual(e.tasks[S].step,0);self.assertTrue(e.halt);e.close()
    def test_schema_numeric_and_truncated_are_actual_actions(self):
        e=self.eng('actions');t=e.tasks[S]
        for review in range(3):
            r,b=e.reserve(S,self.good);z=json.loads(mock_response(b))
            if review==0:z['choices'][0]['message']['content']='not json'
            elif review==1:
                a=json.loads(z['choices'][0]['message']['content']);a['writes']['r0']=(a['writes']['r0']+1)%997;z['choices'][0]['message']['content']=dump(a)
            else:z['choices'][0]['message']['content']='{';z['choices'][0]['finish_reason']='length'
            e.receive(r['call'],200,dump(z).encode())
            if review==1:self.assertEqual(t.state['r0'],a['writes']['r0'])
        self.assertEqual((e.success,t.step,t.status),(3,0,'unrepaired_after_2_reviews'));self.assertEqual(t.counts[0],[1,1,1,1,0]);self.assertIsNone(e.halt)
        with self.assertRaises(Paused):e.reserve(S,self.good)
        e.close();e=self.eng('actions');self.assertTrue(e.tasks[S].terminal);self.assertEqual(e.success,3);e.close()
    def test_flock_and_WAL_corruption(self):
        e=self.eng('flock')
        with self.assertRaises(BlockingIOError):self.eng('flock')
        e.close();p=BASE/'flock/HOLDOUT_D_WAL.jsonl';p.write_bytes(p.read_bytes()+b'{')
        with self.assertRaises(IntegrityError):self.eng('flock')
    def test_default_no_network(self):
        with self.assertRaises(ValueError):HTTPTransport()
        e=self.eng('mock_full');server=MockServer(policy='fatal_at_8');z=execute(e,server,self.good,mock=True)
        self.assertEqual(z['terminal'],144);self.assertEqual(z['successful'],144*11);self.assertEqual(z['attempts'],144*11)
        self.assertTrue(all(json.loads(b)['model']=='gpt-6-luna' for b in server.bodies));self.assertEqual(e.success,1584)
        self.assertTrue(all(t.status=='unrepaired_after_2_reviews' for t in e.tasks.values()));e.close()
        for _ in range(2):
            e=self.eng('mock_full');self.assertEqual(e.success,1584);self.assertTrue(all(t.terminal for t in e.tasks.values()));e.close()
    def test_human_smoke_gate(self):
        e=self.eng('smoke');server=MockServer(policy='fatal_at_8');z=execute(e,server,self.good,mock=False)
        self.assertEqual(z['terminal'],2);self.assertEqual(z['successful'],22)
        with self.assertRaises(Paused):e.approve_smoke(dict(approved=True))
        # E2：旧的自动填写审批正向断言已废止；测试不得伪造人工已审。
        template=e.smoke_review_template();self.assertFalse(template['approved'])
        with self.assertRaises(Paused):e.approve_smoke(template)
        self.assertFalse(e.smoke);e.close()
    def test_exact_prompts_old_trace(self):
        # Replay a complete C trace using pure protocol with its old seed and compare EVERY body/action.
        p=ROOT/'experiments/real/holdout_c_real/sheet_d0_7740001.jsonl'
        events=[json.loads(x) for x in p.read_text().splitlines()];t=Task(7740001)
        req={x['call']:x for x in events if x['event']=='request'};resp={x['call']:x for x in events if x['event']=='response'}
        n=0
        for x in events:
            if x['event']!='action':continue
            call=x['call'];self.assertEqual(json.loads(t.body())['messages'],req[call]['messages'])
            record=t.apply(resp[call]['response'])
            for k in ('before','after','expected','content','correct','error_type','n_after','n_before','step','review','action'):self.assertEqual(record[k],x[k])
            n+=1
        self.assertTrue(t.terminal);self.assertGreater(n,10)

# Actual urllib path, redirect refusal and HTTP status classification via a LOCAL MOCK SERVER.
def http_server(pipe):
    from http.server import HTTPServer,BaseHTTPRequestHandler
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=self.rfile.read(int(self.headers['Content-Length']))
            if self.path=='/redirect':self.send_response(302);self.send_header('Location','http://127.0.0.1:18319/NEVER');self.end_headers();return
            self.send_response(408 if self.path=='/408' else 200);self.end_headers();self.wfile.write(mock_response(body))
    for port in range(49152,65536):
        try:server=HTTPServer(('127.0.0.1',port),Handler);break
        except OSError:continue
    else:raise RuntimeError('test_port_range_exhausted')
    pipe.send(server.server_port)
    for _ in range(3):server.handle_request()
    server.server_close()

class HTTPTests(unittest.TestCase):
    def test_loopback_mock_http_only(self):
        parent,child=multiprocessing.Pipe();p=multiprocessing.get_context('fork').Process(target=http_server,args=(child,));p.start();port=parent.recv()
        try:
            for path,status in [('/ok',200),('/408',408),('/redirect',302)]:
                tr=TestHTTPTransport(f'http://127.0.0.1:{port}{path}');actual,raw,_=tr.send(Task(S).body());self.assertEqual(actual,status)
            p.join(5);self.assertEqual(p.exitcode,0)
        finally:
            if p.is_alive():p.terminate();p.join()

if __name__=='__main__':
    import sys
    start=time.time();suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),seconds=time.time()-start,
                artifacts=str(BASE.relative_to(ROOT)),no_real_llm=True,mock_http_only=True,thread_limit=2)
    with (BASE/'HOLDOUT_D_TEST_RESULTS.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(dump(report));sys.exit(0 if result.wasSuccessful() else 1)
