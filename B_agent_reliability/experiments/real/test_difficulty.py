import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import difficulty as d


class Tests(unittest.TestCase):
    def test_sheet_sequential_and_formulas(self):
        t = {'family': 'sheet'}
        s = {f'r{i}': i+1 for i in range(6)}
        got = d.transition(t, s, {'instructions': [{'a': 0, 'b': 1, 'c': 2, 'm': 3, 'd': -10}]})
        # First r0=-6 mod997=991; r2=(3+991)=994.
        self.assertEqual([got[k] for k in ('r0', 'r2', 'F0', 'F1', 'F2')], [991, 994, 1, 2, 14])
        self.assertEqual(s['r0'], 1)

    def test_indirect_alias_and_negative_mod(self):
        s = {f'r{i}': 1 for i in range(8)}
        s['cursor'] = 0
        t = {'family': 'indirect', 'table': {str(i): [i, i] for i in range(8)}}
        got = d.transition(t, s, {'start': 0, 'delta': [-5]})
        self.assertEqual(got['r0'], (994+2*994) % 997)
        self.assertEqual(got['cursor'], got['r0'] % 3)

    def test_memory_alias(self):
        t = {'family': 'memory', 'knob': 4}
        s = {'r0': 2, 'r1': 3, 'r2': 4, 'r3': 5, 'checksum': 1}
        got = d.transition(t, s, {'a': 0, 'b': 0, 'shift': 1, 'm': 2, 'd': -10})
        self.assertEqual(got, dict(r0=993, r1=3, r2=4, r3=5, checksum=(7+993+9+993)%997))

    def test_schedule_new_tasks(self):
        rows = d.schedule()
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({r['seed'] for r in rows}), 48)
        for f in d.KNOBS:
            for level in range(4):
                self.assertEqual(sum(r['family'] == f and r['level'] == level for r in rows), 4)

    def test_schema(self):
        valid = '{"writes":{"r0":3},"detect":false}'
        self.assertIsNone(d.parse(valid, ['r0'])[1])
        for bad in ['{"writes":{"r0":true},"detect":false}',
                    '{"writes":{"r0":3,"r0":4},"detect":false}',
                    '{"writes":{"r0":3},"detect":false,"x":0}',
                    '{"writes":{"r0":-1},"detect":false}', valid.replace('r0', 'r1'), 'not json']:
            self.assertIsNone(d.parse(bad, ['r0'])[0])

    def test_mock_complete_and_real_wrong_action_repair(self):
        with tempfile.TemporaryDirectory(dir=d.ROOT) as tmp:
            out = Path(tmp)
            b = d.Budget(out, False)
            c = d.Client(b, mock=True)
            row = d.schedule()[0]
            original = c.request
            count = 0
            def one_error(messages, trace, context, expected):
                nonlocal count
                count += 1
                x = copy.deepcopy(expected)
                if count == 1:
                    k = next(iter(x['writes']))
                    x['writes'][k] = (x['writes'][k]+1)%997
                return original(messages, trace, context, x)
            c.request = one_error
            r = d.run_task(row, c, out)
            self.assertEqual((r['status'], r['advances'], r['seed_errors'], r['repaired']), ('success', 16, 1, 1))
            events = [json.loads(x) for x in (out/(r['id']+'.jsonl')).read_text().splitlines()]
            a = [x for x in events if x['event'] == 'action']
            self.assertNotEqual(a[0]['after'], a[0]['expected'])
            self.assertEqual(a[1]['before'], a[0]['after'])
            b.journal.f.close()

    def test_failed_repair_censors(self):
        with tempfile.TemporaryDirectory(dir=d.ROOT) as tmp:
            out = Path(tmp)
            b = d.Budget(out, False)
            c = d.Client(b, mock=True)
            original = c.request
            def wrong(messages, trace, context, expected):
                x = copy.deepcopy(expected)
                k = next(iter(x['writes']))
                x['writes'][k] = (x['writes'][k]+1)%997
                return original(messages, trace, context, x)
            c.request = wrong
            r = d.run_task(d.schedule()[0], c, out)
            self.assertEqual((r['advances'], r['seed_errors'], r['repair_reerrors']), (1, 1, 2))
            self.assertEqual(r['status'], 'unrepaired_after_2_reviews')
            b.journal.f.close()

    def test_hard_cap_and_quota_stop(self):
        with tempfile.TemporaryDirectory(dir=d.ROOT) as tmp:
            b = d.Budget(Path(tmp), False)
            b.calls = 1199
            self.assertEqual(b.reserve(), 1200)
            with self.assertRaises(d.Stop):
                b.reserve()
            self.assertEqual(b.calls, 1200)
            b.journal.f.close()
        with tempfile.TemporaryDirectory(dir=d.ROOT) as tmp:
            b = d.Budget(Path(tmp), True)
            with patch.object(d.subprocess, 'run') as run:
                run.return_value.stdout = '0 80'
                with self.assertRaises(d.Stop):
                    b.reserve()
            self.assertEqual(b.calls, 0)
            b.journal.f.close()

    def test_model_guard_and_retry(self):
        class FakeOpener:
            def __init__(self, responses):
                self.responses = iter(responses)
                self.bodies = []
            def open(self, req, timeout):
                self.bodies.append(json.loads(req.data))
                value = next(self.responses)
                if isinstance(value, Exception):
                    raise value
                self.value = value
                return self
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return json.dumps(self.value).encode()
        for sequence, calls, halt in [
            ([OSError(), {'model': 'wrong'}], 2, 'model_mismatch'),
            ([OSError(), OSError()], 2, None)]:
            with tempfile.TemporaryDirectory(dir=d.ROOT) as tmp:
                out = Path(tmp)
                b = d.Budget(out, False)
                trace = d.Journal(out/'trace.jsonl')
                opener = FakeOpener(sequence)
                with patch.dict(d.os.environ, {'SOLKEY': 'TEST_ONLY'}), patch.object(d.urllib.request, 'build_opener', return_value=opener):
                    c = d.Client(b)
                    with self.assertRaises(d.Stop):
                        c.request([], trace, {})
                self.assertEqual(b.calls, calls)
                self.assertEqual(b.halt, halt)
                self.assertEqual(opener.bodies[0], opener.bodies[1])
                self.assertEqual(opener.bodies[0]['model'], d.MODEL)
                self.assertEqual(opener.bodies[0]['max_completion_tokens'], 4096)
                trace.f.close()
                b.journal.f.close()


if __name__ == '__main__':
    unittest.main()
