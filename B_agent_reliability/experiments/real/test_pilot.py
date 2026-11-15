import copy
import json
from pathlib import Path
import tempfile
import unittest
from pilot import Task, Executor, Mock, Real, Stop, Trace, run_one, prompt

class Tests(unittest.TestCase):
    def run_task(self, family='kv', injected=False, client=None, **kwargs):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'trace.jsonl'
            # run_one returns paths relative to ROOT; use ROOT child temp directory instead below
            from pilot import ROOT
            with tempfile.TemporaryDirectory(dir=ROOT) as actual:
                trace = Trace(Path(actual) / 'trace.jsonl')
                r = run_one(Task(family, 1105), client or Mock(), trace, injected, **kwargs)
                events = [json.loads(s) for s in trace.path.read_text().splitlines()]
                return r, events
    def test_mock_end_to_end(self):
        for family in ['kv', 'sheet']:
            for inject in [False, True]:
                with self.subTest(family=family, inject=inject):
                    r, events = self.run_task(family, inject)
                    self.assertTrue(r['success'])
                    self.assertEqual(r['calls'], 18 if inject else 16)
                    self.assertEqual(r['injections'], 2 if inject else 0)
                    self.assertEqual(len(r['final']['ledger']), 16)
                    self.assertEqual(len([e for e in events if e['event']=='response']), r['calls'])
    def test_paired_prefix(self):
        for f in ['kv', 'sheet']:
            self.assertEqual(Task(f, 88, 16).operations, Task(f, 88, 32).operations[:16])
            self.assertEqual(Task(f, 88, 16).prefix, Task(f, 88, 32).prefix[:17])
    def test_undo(self):
        ex = Executor(Task('sheet', 11))
        initial = ex.state.copy(); key = next(iter(initial))
        ex.apply({'op':'write','target':key,'value':999,'detect':False})
        self.assertNotEqual(initial, ex.state)
        ex.apply({'op':'undo','detect':True})
        self.assertEqual(initial, ex.state)
    def test_capacity(self):
        ex = Executor(Task('sheet',11), k=3)
        keys = list(ex.state)
        for key in keys[:2]: ex.apply({'op':'write','target':key,'value':100,'detect':True})
        with self.assertRaisesRegex(Stop,'undo_capacity'):
            ex.apply({'op':'write','target':keys[2],'value':100,'detect':True})
        self.assertEqual(len(ex.active),2)
    def test_invalid_actions(self):
        for a in [None,{}, {'op':'write','target':'missing','value':1,'detect':False},
                  {'op':'write','target':'k0','value':True,'detect':False},
                  {'op':'write','target':'k0','value':1,'detect':'false'},
                  {'op':'undo','detect':True}, {'op':'exec','detect':True}]:
            ex=Executor(Task('kv',11)); initial=ex.state.copy()
            self.assertFalse(ex.apply(a)); self.assertEqual(initial,ex.state)
            self.assertTrue(ex.invalid)
    def test_call_limit(self):
        class Bad(Mock):
            def complete(self,m):
                r=super().complete(m)
                r['choices'][0]['message']['content']='malformed'
                return r
        r,_=self.run_task(client=Bad(), call_limit=3)
        self.assertFalse(r['success']); self.assertEqual(r['status'],'call_limit')
        self.assertEqual(r['calls'],3)
    def test_global_cap(self):
        b={'calls':150,'halt':None}
        r,_=self.run_task(budget=b)
        self.assertEqual(r['status'],'global_call_limit'); self.assertEqual(r['calls'],0)
    def test_model_mismatch(self):
        class Wrong(Real):
            def __init__(self): pass
            def complete(self,m):
                r=Mock().complete(m); r['model']='wrong'; return r
        b={'calls':0,'halt':None}
        r,events=self.run_task(client=Wrong(),budget=b)
        self.assertEqual(r['status'],'model_mismatch'); self.assertEqual(r['calls'],1)
        self.assertEqual(b['halt'],'model_mismatch')
        self.assertTrue(any(e['event']=='response' for e in events))
    def test_usage_guard(self):
        class Oversize(Real):
            def __init__(self): pass
            def complete(self,m):
                r=Mock().complete(m); r['model']='gpt-6-luna'
                r['usage']['prompt_tokens']=2049
                return r
        r,_=self.run_task(client=Oversize())
        self.assertEqual(r['status'],'provider_token_limit')
    def test_final_cancellation_not_success(self):
        ex=Executor(Task('kv',11)); ex.state=ex.task.prefix[-1].copy()
        self.assertTrue(ex.final()['state_ok'])
        self.assertFalse(ex.final()['obligations_ok'])
    def test_ledger_tampering(self):
        t=Task('kv',11); ex=Executor(t)
        for state in t.prefix[1:]: ex.state=state.copy(); ex.commit()
        self.assertTrue(ex.final()['obligations_ok'])
        ex.ledger[0]['state']['k0'] += 1
        self.assertFalse(ex.final()['obligations_ok'])
    def test_multi_error_observed(self):
        ex=Executor(Task('sheet',11))
        keys=list(ex.state)
        for key in keys[:2]: ex.apply({'op':'write','target':key,'value':999,'detect':True})
        self.assertGreaterEqual(ex.n(),2)
    def test_recovery_clears_history(self):
        r,events=self.run_task(injected=True)
        repair=[e for e in events if e['event']=='request' and e['mode']=='repair']
        self.assertTrue(all(json.loads(e['messages'][1]['content'])['untrusted_history'] for e in repair))
        self.assertEqual(r['recoveries'],2)
        self.assertEqual(sum(e['event']=='commit' and e['repaired'] for e in events),2)
    def test_trace_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x'; t=Trace(p); t.close()
            with self.assertRaises(FileExistsError): Trace(p)

if __name__=='__main__': unittest.main(verbosity=2)
