"""R1/R2 的网络前拒绝、一次性发送及真实进程崩溃回归。全部返回均为内存 fixture。"""
import json
import multiprocessing
import os
import threading
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from engine import Engine, Paused, IntegrityError
from transport import HTTPTransport, TestHTTPTransport, MockServer, URL
from protocol import Task, sha
from evidence import FIRST, REVIEW_DECLARATION, validate_smoke
from test_mock import BASE, Clock

S=FIRST[0]
def good():return dict(five_hour=0.,week=0.)

def projection():
    """仅投影 live 身份和放行文件；保留发送、预留、门控与 WAL 原实现。"""
    stack=ExitStack()
    stack.enter_context(patch.object(Engine,'mode',new=property(lambda self:'live')))
    stack.enter_context(patch('engine.require_live_release',return_value=None))
    stack.enter_context(patch('transport.require_live_release',return_value=None))
    stack.enter_context(patch.dict(os.environ,{'SOLKEY':'OFFLINE_FIXTURE_NOT_SECRET'}))
    return stack

class Response:
    status=200
    def __init__(self,body):self.body=body
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return MockServer().send(self.body)[1]

def transport(engine,opens,crash=False):
    class Sentinel:
        def open(self,req,timeout):
            call=engine.attempts
            # 从实际磁盘重新读取，确认网络边界之前持久记录恰有一次。
            ev=[json.loads(x) for x in engine.wal.path.read_text().splitlines()]
            ds=[e for e in ev if e['kind']=='dispatch_started' and e['call']==call]
            assert len(ds)==1 and ds[0]['body_sha256']==sha(req.data)
            assert not engine.lock._is_owned(), 'network under global lock'
            opens.append(call)
            if crash:os._exit(92)
            return Response(req.data)
    tr=object.__new__(HTTPTransport);tr.url=URL;tr.opener=Sentinel()
    return tr

class Amend3Tests(unittest.TestCase):
    def eng(self,name):return Engine(BASE/('v3_'+name),verify=False,clock=Clock())
    def test_R1_all_address_attacks_before_client_construction(self):
        targets=[URL,'http://127.0.0.1:18319/anything',
                 'http://127.0.0.1:18319@audit.invalid/v1/chat/completions',
                 'http://127.0.0.1:18319@audit.invalid:80/x',
                 'http://user@127.0.0.1:49152/x','http://127.0.0.1:49152@audit.invalid/x',
                 'https://127.0.0.1:49152/x','http://localhost:49152/x',
                 'http://127.0.0.1:49151/x','http://127.0.0.1:65536/x']
        with patch('urllib.request.build_opener',side_effect=AssertionError('CLIENT_CONSTRUCTED')):
            for url in targets:
                with self.subTest(url=url):
                    with self.assertRaises(ValueError):HTTPTransport(live=True,test_mock_url=url)
                    with self.assertRaises(ValueError):TestHTTPTransport(url)
        for port in (49152,65535):
            self.assertEqual(TestHTTPTransport.validate_url(f'http://127.0.0.1:{port}/x'),f'http://127.0.0.1:{port}/x')
    def test_R1_mock_third_task_and_recovery_cannot_use_production(self):
        e=self.eng('endpoint_escape');server=MockServer(policy='fatal_at_8')
        for seed in FIRST:
            while not e.tasks[seed].terminal:e.request(seed,server,good)
        for restart in (False,True):
            if restart:e=self.eng('endpoint_escape')
            before=e.attempts;tr=transport(e,[])
            with self.assertRaisesRegex(Paused,'mock_cannot_use_live_transport'):e.request(S+2,tr,good)
            # 旧 testing 属性即使被手工设为 True，生产 send 仍坚持 live Engine。
            tr.testing=True
            with patch('transport.require_live_release',return_value=None):
                with self.assertRaisesRegex(ValueError,'requires_engine_reservation'):
                    tr.send(Task(S+2).body(),engine=e,reservation={})
            self.assertEqual((e.attempts,e.tasks[S+2].step,e.smoke),(before,0,False));e.close()
    def test_R1_test_url_mutation_rechecked_before_request(self):
        tr=TestHTTPTransport('http://127.0.0.1:49152/x');tr.url=URL
        with patch('urllib.request.Request',side_effect=AssertionError('REQUEST_CONSTRUCTED')):
            with self.assertRaises(ValueError):tr.send(Task(S+2).body())
    def test_R2_sequential_duplicate_and_after_response(self):
        e=self.eng('repeat');opens=[];tr=transport(e,opens)
        with projection():
            r,b=e.reserve(S,good);status,raw,reason=tr.send(b,engine=e,reservation=r)
            for _ in range(2):
                with self.assertRaisesRegex(Paused,'dispatch_already_started'):tr.send(b,engine=e,reservation=r)
            self.assertEqual((len(opens),e.attempts,len(e.pending)),(1,1,1))
            e.receive(r['call'],status,raw,reason)
            with self.assertRaisesRegex(Paused,'dispatch_already_started'):tr.send(b,engine=e,reservation=r)
        e.close()
    def test_R2_two_threads_compete_for_one_reservation(self):
        e=self.eng('race');opens=[];tr=transport(e,opens);barrier=threading.Barrier(2);results=[]
        with projection():
            r,b=e.reserve(S,good)
            def send():
                barrier.wait()
                try:tr.send(b,engine=e,reservation=r);results.append('sent')
                except Paused as ex:results.append(str(ex))
            worker=threading.Thread(target=send);worker.start();send();worker.join(10)
            self.assertFalse(worker.is_alive());self.assertCountEqual(results,['sent','dispatch_already_started'])
            self.assertEqual((opens,e.attempts,len(e.pending)),([1],1,1))
        e.close()
    def test_R2_dispatch_crash_keeps_unknown_across_two_recoveries(self):
        path=BASE/'v3_dispatch_crash'
        def child():
            e=Engine(path,verify=False)
            with projection():
                r,b=e.reserve(S,good);transport(e,[],crash=True).send(b,engine=e,reservation=r)
        p=multiprocessing.get_context('fork').Process(target=child);p.start();p.join(10)
        self.assertEqual(p.exitcode,92)
        for _ in range(2):
            e=Engine(path,verify=False)
            self.assertEqual((e.attempts,len(e.pending),len(e.active),e.dispatched),(1,1,0,{1}))
            with projection():
                with self.assertRaises(Paused):transport(e,[]).send(Task(S).body(),engine=e,reservation=e.pending[1])
                with self.assertRaises(Paused):e.reserve(S,good)
            e.close()
    def test_R2_fsync_failure_never_reaches_network_or_reuses_right(self):
        e=self.eng('fsync_failure');opens=[];tr=transport(e,opens)
        with projection():
            r,b=e.reserve(S,good)
            with patch('engine.os.fsync',side_effect=OSError('fsync fixture')):
                with self.assertRaises(OSError):tr.send(b,engine=e,reservation=r)
            with self.assertRaises(Paused):tr.send(b,engine=e,reservation=r)
        self.assertEqual((opens,len(e.pending),e.halt),([],1,'unknown_inflight'));e.close()
    def test_R2_legal_retry_new_attempt_and_reservation(self):
        e=self.eng('retry');opens=[];tr=transport(e,opens)
        with projection():
            r,b=e.reserve(S,good);tr.send(b,engine=e,reservation=r);e.receive(r['call'],408)
            e.clock.advance(300)
            r2,b2=e.reserve(S,good);tr.send(b2,engine=e,reservation=r2)
            self.assertEqual((r2['call'],r2['attempt'],b2),(2,2,b))
            self.assertEqual(opens,[1,2]);self.assertEqual(e.dispatched,{1,2})
        e.close()
    def test_review_declaration_and_snapshot_digest_boundary(self):
        e=self.eng('declaration');server=MockServer(policy='fatal_at_8')
        for seed in FIRST:
            while not e.tasks[seed].terminal:e.request(seed,server,good)
        t=e.smoke_review_template()
        self.assertEqual(t['reviewer_declaration'],'');self.assertFalse(t['approved'])
        from datetime import datetime,timezone
        t.update(approved=True,reviewer='FORGED_FIXTURE_NOT_AUTHORIZATION',reviewed_at=datetime.now(timezone.utc).isoformat())
        t['checked_items']={k:True for k in t['checked_items']}
        with self.assertRaisesRegex(ValueError,'declaration'):validate_smoke(t,e.wal.events,e.runner_version)
        t['reviewer_declaration']=REVIEW_DECLARATION;t['snapshot_digest']='0'*64
        with self.assertRaisesRegex(ValueError,'snapshot_digest'):validate_smoke(t,e.wal.events,e.runner_version)
        e.close()
