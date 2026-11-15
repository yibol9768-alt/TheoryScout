import json
import unittest
from unittest.mock import patch
from pilot import Task, Executor, Mock, Real, MODEL, MAX_COMPLETION_TOKENS, parse_action, prompt
from test_pilot import Tests
from pilot2 import schedule

class Pilot2Tests(Tests):
    def test_schedule(self):
        rows=schedule()
        self.assertEqual(len(rows),24)
        self.assertEqual(len({(r['family'],r['seed']) for r in rows}),12)
        self.assertEqual(len({(r['family'],r['seed'],r['K']) for r in rows}),12)
        self.assertEqual(24*20,480)
        for family in ('kv','sheet'):
            for k in (2,3):
                self.assertEqual(sum(r['family']==family and r['K']==k for r in rows),6)
    def test_all_pilot2_mock(self):
        for family in ('kv','sheet'):
            for k in (2,3):
                for injected in (False,True):
                    r,_=self.run_task(family,injected,k=k,call_limit=20)
                    self.assertTrue(r['success'])
    def test_schema_explicit_and_strict(self):
        for family in ('kv','sheet'):
            ex=Executor(Task(family,3301))
            p=prompt(ex,'advance',[])
            self.assertEqual(json.loads(p[1]['content'])['legal_targets'],list(ex.state))
            self.assertIn('Valid JSON examples',p[0]['content'])
            for target in ('cell/k0',[],{},None):
                self.assertFalse(ex.apply(dict(op='write',target=target,value=7,detect=False)))
        with self.assertRaises(ValueError): parse_action('{"op":"write","op":"undo"}')
        ex=Executor(Task('kv',3301))
        ex.apply(dict(op='write',target='k0',value=7,detect=False))
        self.assertFalse(ex.apply(dict(op='undo',detect=False)))
    def test_wire_token_limit(self):
        class Reply:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self): return b'{}'
        class Opener:
            def open(self,req,timeout):
                body=json.loads(req.data)
                if body['max_completion_tokens']!=2048 or body['model']!=MODEL:
                    raise AssertionError('incorrect wire fields')
                return Reply()
        with patch.dict('os.environ',{'SOLKEY':'mock-key-never-real'}), patch('urllib.request.build_opener',return_value=Opener()):
            Real().complete([])
        self.assertEqual(MAX_COMPLETION_TOKENS,2048)
    def test_output_and_reasoning_guards(self):
        for tokens,reasoning,finish,expected in ((2049,2049,'stop','provider_token_limit'),
                (2048,2048,'length','completion_length_limit'),(100,2049,'stop','provider_token_limit')):
            class Oversize(Real):
                def __init__(self): pass
                def complete(self,m):
                    r=Mock().complete(m); r['model']=MODEL
                    r['usage'].update(completion_tokens=tokens,total_tokens=tokens,
                                      completion_tokens_details={'reasoning_tokens':reasoning})
                    r['choices'][0]['finish_reason']=finish
                    return r
            r,events=self.run_task(client=Oversize())
            self.assertEqual(r['status'],expected)
            self.assertEqual(r['work'],0)
            self.assertEqual(r['usage']['completion_tokens'],tokens)
    def test_injection_not_natural_error(self):
        r,events=self.run_task(injected=True)
        inj=[e for e in events if e['event']=='transition' and e['injection']]
        self.assertEqual(len(inj),2)
        self.assertTrue(all(e['model_action_n_after']==0 and e['n_after']==1 for e in inj))
    def test_500_hard_stop(self):
        b=dict(calls=499,limit=500,halt=None)
        r,_=self.run_task(budget=b)
        self.assertEqual(r['calls'],1)
        self.assertEqual(b['calls'],500)
        self.assertEqual(r['status'],'global_call_limit')

if __name__=='__main__': unittest.main(verbosity=2)
