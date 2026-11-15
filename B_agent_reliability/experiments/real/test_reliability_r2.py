import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import difficulty as old
import reliability_core as d
import reliability_r2 as r
import report_reliability_r2 as report

class Tests(unittest.TestCase):
    def test_inherited_generation_and_execution(self):
        self.assertEqual(d.COMMON,old.COMMON)
        self.assertEqual(d.RULES,old.RULES)
        for row in old.schedule():
            self.assertEqual(d.task(row),old.task(row))
            long=d.task(dict(row,length=32))
            self.assertEqual(long['prefix'][:17],old.task(row)['prefix'])
            self.assertEqual(report.oracle(long),long['prefix'])
        rows=r.r2_schedule()
        self.assertEqual(len({x['seed'] for x in rows}),48)
        self.assertTrue(set(x['seed'] for x in rows).isdisjoint(x['seed'] for x in old.schedule()))

    def test_actual_wrong_state_then_repair(self):
        with tempfile.TemporaryDirectory(dir=d.ROOT) as tmp:
            out=Path(tmp)
            b=r.Budget(out,'r2',True)
            c=r.Client(b,True)
            original=c.request
            counter=0
            def wrong_once(messages,trace,context,expected):
                nonlocal counter
                counter+=1
                expected=copy.deepcopy(expected)
                if counter==1:
                    key=next(iter(expected['writes']))
                    expected['writes'][key]=(expected['writes'][key]+1)%997
                return original(messages,trace,context,expected)
            c.request=wrong_once
            result=d.run_task(r.r2_schedule()[0],c,out)
            self.assertEqual((result['status'],result['seed_errors'],result['repaired']),('success',1,1))
            actions=[json.loads(s) for s in (out/(result['id']+'.jsonl')).read_text().splitlines() if json.loads(s)['event']=='action']
            self.assertNotEqual(actions[0]['after'],actions[0]['expected'])
            self.assertEqual(actions[0]['after'],actions[1]['before'])
            b.journal.f.close()

    def test_retry_exhaustion_cooldown_and_identical_body(self):
        class Failing:
            def __init__(self): self.bodies=[]
            def open(self,req,timeout):
                self.bodies.append(req.data)
                raise OSError()
        with tempfile.TemporaryDirectory(dir=d.ROOT) as tmp:
            out=Path(tmp)
            b=r.Budget(out,'r2',True)
            trace=d.Journal(out/'sheet_d0_1.jsonl')
            with patch.dict(d.os.environ,{'SOLKEY':'TEST_ONLY'}),patch.object(r.time,'sleep') as sleep:
                c=r.Client(b)
                c.opener=Failing()
                with self.assertRaisesRegex(d.Stop,'service_retry_exhausted'):
                    c.request([],trace,{'step':0,'review':0})
                self.assertEqual([x.args[0] for x in sleep.call_args_list],[10,30,60,120])
                self.assertEqual(b.calls,5)
                self.assertEqual(len(set(c.opener.bodies)),1)
                self.assertGreater(b.cooldown_until,r.time.monotonic()+590)
                self.assertIsNone(b.halt)
            trace.f.close()
            b.journal.f.close()

    def test_caps_quota_boundary_and_recent_window(self):
        with tempfile.TemporaryDirectory(dir=d.ROOT) as tmp:
            b=r.Budget(Path(tmp),'calib1',True)
            b.task_calls['a']=48
            with self.assertRaisesRegex(d.Stop,'task_cap_48'): b.reserve('a','c')
            b.config_calls['c']=2400
            with self.assertRaisesRegex(d.Stop,'config_cap_2400'): b.reserve('b','c')
            b.calls=4800
            with self.assertRaisesRegex(d.Stop,'hard_cap'): b.reserve('b','d')
            self.assertEqual(b.calls,4800)
            b.journal.f.close()
        with tempfile.TemporaryDirectory(dir=d.ROOT) as tmp:
            b=r.Budget(Path(tmp),'r2')
            with patch.object(r.subprocess,'run') as p:
                p.return_value.stdout='0 79'
                b.calls=100
                self.assertEqual(b.reserve('a','c'),101)
                b.calls=200
                p.return_value.stdout='80 1'
                with self.assertRaisesRegex(d.Stop,'quota_80'): b.reserve('a','c')
                self.assertEqual(b.calls,200)
            b.journal.f.close()
        with tempfile.TemporaryDirectory(dir=d.ROOT) as tmp:
            b=r.Budget(Path(tmp),'calib1',True)
            for i in range(20): b.outcome(i<4,i+1)
            b.boundary('c',10)
            self.assertEqual(b.halt,'service_boundary_stop')
            b.journal.f.close()

    def test_exact_reference(self):
        self.assertAlmostEqual(report.exact(0,10)[1],1-.025**.1,places=10)
        self.assertAlmostEqual(report.exact(10,10)[0],.025**.1,places=10)
        ci=report.exact(6,64)
        self.assertTrue(.035<ci[0]<.036 and .192<ci[1]<.194)

if __name__=='__main__': unittest.main()
