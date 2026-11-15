import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import holdout_c as h
import reliability_core as d
import report_reliability_r2 as ref

class Tests(unittest.TestCase):
    def test_schedule_oracle_and_no_overlap(self):
        rows=h.schedule()
        self.assertEqual(len(rows),40)
        self.assertEqual(len({x['seed'] for x in rows}),40)
        old={x['seed'] for x in h.b.schedule()}
        old.update(x['seed'] for x in json.loads((h.ROOT/'calib1_real/summary.json').read_text())['results'])
        self.assertFalse(old.intersection(x['seed'] for x in rows))
        for row in rows:self.assertEqual(d.task(row)['prefix'],ref.oracle(d.task(row)))
    def test_failed_service_pauses_global_and_preserves_work(self):
        with tempfile.TemporaryDirectory(dir=h.ROOT) as tmp:
            b=h.Budget(Path(tmp),mock=True)
            now=[1000.]
            with patch.object(h.time,'monotonic',side_effect=lambda:now[0]),patch.object(h.time,'sleep',side_effect=lambda x:now.__setitem__(0,now[0]+x)):
                first=b.reserve('a','sheet');b.outcome(True,first)
                second=b.reserve('b','sheet')
                self.assertGreaterEqual(now[0],1300)
                self.assertEqual((b.successful,b.failed),(0,1))
                b.outcome(False,second)
                self.assertEqual((b.successful,b.calls),(1,2))
            b.journal.f.close()
    def test_quota_failure_and_both_thresholds_wait(self):
        with tempfile.TemporaryDirectory(dir=h.ROOT) as tmp:
            b=h.Budget(Path(tmp),mock=True);b.real=True
            with patch.object(b,'quota_values',side_effect=[ValueError(),(80,10),(0,90),(79,89)]),patch.object(h.time,'sleep') as sleep:
                b.quota();self.assertEqual(sleep.call_count,3)
                self.assertEqual(b.quota_at,0);self.assertIsNone(b.halt)
            b.journal.f.close()
    def test_caps_and_accounting(self):
        with tempfile.TemporaryDirectory(dir=h.ROOT) as tmp:
            b=h.Budget(Path(tmp),mock=True);b.successful=1999;b.quota_at=1900
            c=b.reserve('a','sheet');b.outcome(True,c)
            self.assertEqual(b.successful,1999)
            b.pause_until=0;b.need_probe=False
            c=b.reserve('a','sheet');b.outcome(False,c)
            b.calls=3000
            with self.assertRaisesRegex(d.Stop,'service_attempt_cap'):b.reserve('a','sheet')
            b.journal.f.close()

if __name__=='__main__':unittest.main()
