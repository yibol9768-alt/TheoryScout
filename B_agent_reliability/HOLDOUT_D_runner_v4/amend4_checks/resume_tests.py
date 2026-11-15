"""M1 恢复验收：虚拟时钟、本地发送桩、同一预留不重复占预算。"""
import base64,json,unittest
from harness import ROOT,OUT,resources,thread_counts
from boundary_tests import engine,projection,tr,good,S

class Clock:
    def __init__(self):self.t=1000.
    def __call__(self):return self.t

class Resume(unittest.TestCase):
    def eng(self,name):
        c=Clock();e=engine.Engine(OUT/'resume'/name,verify=False,clock=c)
        self.addCleanup(e.close)
        return e,c
    def test_01_same_reservation_all_errors_and_quota_failures(self):
        for status in (0,408,500,501,502,503,504,599):
            with self.subTest(status=status),projection():
                e,c=self.eng('errors_'+str(status));calls=[];probes=[]
                def probe():probes.append(c());return good()
                a,ba=e.reserve(S,probe);b,bb=e.reserve(S+1,probe)
                code,raw,reason=tr(e,calls,status).send(ba,engine=e,reservation=a)
                e.receive(a['call'],code,raw,reason)
                before=e.wal.path.read_bytes();saved=dict(b)
                for now in (1000.,1299.999):
                    c.t=now
                    with self.assertRaises(engine.DispatchPaused):tr(e,calls).send(bb,engine=e,reservation=b)
                    with self.assertRaises(engine.Paused):e.request(S+1,tr(e,calls),probe)
                self.assertEqual(e.wal.path.read_bytes(),before)
                c.t=1300.
                with self.assertRaises(engine.DispatchPaused):tr(e,calls).send(bb,engine=e,reservation=b)
                with self.assertRaises(engine.Paused):e.request(S+1,tr(e,calls),lambda:dict(five_hour=.8,week=0.))
                self.assertEqual(e.probe_retry,1600.)
                # 即使 need_probe 被误清除，额度暂停仍在发送边界生效。
                e.need_probe=False
                with self.assertRaises(engine.DispatchPaused):tr(e,calls).send(bb,engine=e,reservation=b)
                e.need_probe=True;c.t=1599.999
                with self.assertRaises(engine.Paused):e.request(S+1,tr(e,calls),probe)
                self.assertEqual(probes,[1000.]);c.t=1600.
                e.request(S+1,tr(e,calls),probe)
                self.assertEqual(probes,[1000.,1600.])
                self.assertEqual((e.attempts,e.success,e.failed,len(calls),e.halt),(2,1,1,2,None))
                self.assertEqual(e.requests[b['call']],saved)
                self.assertEqual(base64.b64decode(saved['body']),bb)
                self.assertEqual(len([x for x in e.wal.events if x['kind']=='reserve']),2)
                self.assertEqual(len([x for x in e.wal.events if x['kind']=='dispatch_started' and x['call']==b['call']]),1)
    def test_02_actual_request_interleaving_then_resume(self):
        import threading
        e,c=self.eng('normal');calls=[];ready=threading.Event();release=threading.Event();errors=[]
        slow=tr(e,calls)
        class Delayed:
            testing=False
            def send(self,body,**kw):
                ready.set();assert release.wait(10)
                return slow.send(body,**kw)
        with projection():
            def worker():
                try:e.request(S+1,Delayed(),good)
                except BaseException as ex:errors.append(ex)
            t=threading.Thread(target=worker);t.start();self.assertTrue(ready.wait(10))
            try:e.request(S,tr(e,calls,408),good)
            finally:release.set()
            t.join(10);self.assertFalse(t.is_alive())
            self.assertEqual(len(errors),1);self.assertIsInstance(errors[0],engine.DispatchPaused)
            self.assertEqual((e.halt,e.attempts,e.active,len(calls)),(None,2,{1},1))
            saved=e.pending[1];c.t=1300.;e.request(S+1,tr(e,calls),good)
            self.assertIs(e.requests[1],saved)
            self.assertEqual((e.attempts,e.success,e.dispatched,len(calls)),(2,1,{1,2},2))
    def test_03_probe_drains_dispatched_but_not_unsent(self):
        e,c=self.eng('drain');calls=[]
        with projection():
            a,ba=e.reserve(S,good);b,bb=e.reserve(S+1,good)
            result=tr(e,calls).send(ba,engine=e,reservation=a);e.need_probe=True
            with self.assertRaises(engine.DispatchPaused):tr(e,calls).send(bb,engine=e,reservation=b)
            with self.assertRaises(engine.Paused):e.quota(lambda:self.fail('probe while dispatched'))
            e.receive(a['call'],*result);e.quota(good)
            tr(e,calls).send(bb,engine=e,reservation=b)
            self.assertEqual((e.attempts,len(calls)),(2,2))
    def test_04_last_failure_deadline_and_drain(self):
        e,c=self.eng('last_failure');calls=[]
        with projection():
            a,ba=e.reserve(S,good);b,bb=e.reserve(S+1,good)
            ra=tr(e,calls,408).send(ba,engine=e,reservation=a)
            rb=tr(e,calls,599).send(bb,engine=e,reservation=b)
            e.receive(a['call'],*ra);c.t=1300.
            with self.assertRaises(engine.Paused):e.quota(lambda:self.fail('not drained'))
            e.receive(b['call'],*rb);self.assertEqual(e.pause_until,1600.)
            c.t=1599.999
            with self.assertRaises(engine.Paused):e.quota(good)
            c.t=1600.;e.quota(good);self.assertFalse(e.need_probe)
    def test_05_paused_reservation_restart_stays_unknown(self):
        e,c=self.eng('restart');calls=[]
        with projection():
            a,ba=e.reserve(S,good);b,bb=e.reserve(S+1,good)
            e.receive(a['call'],*tr(e,calls,408).send(ba,engine=e,reservation=a))
            with self.assertRaises(engine.DispatchPaused):tr(e,calls).send(bb,engine=e,reservation=b)
        path=e.wal.out;e.close();self._cleanups.pop();c.t=1300.
        for _ in range(2):
            restored=engine.Engine(path,verify=False,clock=c)
            try:
                self.assertEqual((restored.halt,restored.active,restored.attempts),('unknown_inflight',set(),2))
                with self.assertRaises(engine.Paused):restored.quota(lambda:self.fail('unknown probe'))
                with self.assertRaises(engine.Paused):restored.request(S+1,tr(restored,calls),good)
                self.assertEqual(len(calls),1)
            finally:restored.close()
    def test_06_boundary_and_low_level_probe_reuse(self):
        e,c=self.eng('boundary');calls=[]
        with projection():
            a,ba=e.reserve(S,good);b,bb=e.reserve(S+1,good)
            e.quota_bucket=-1
            with self.assertRaises(engine.DispatchPaused):tr(e,calls).send(bb,engine=e,reservation=b)
            e.quota(good)
            tr(e,calls).send(bb,engine=e,reservation=b)
            self.assertEqual((e.attempts,len(e.pending),len(calls)),(2,2,1))

if __name__=='__main__':
    resources();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Resume))
    (OUT/'resume_results.json').write_text(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),failures=[(str(t),e) for t,e in r.failures],errors=[(str(t),e) for t,e in r.errors],resources=resources(),thread_counts=thread_counts),indent=2)+'\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
