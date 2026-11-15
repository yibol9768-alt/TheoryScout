import collections
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import holdout_b as h
import reliability_core as d
import reliability_r2 as r
import report_reliability_r2 as reference

class HoldoutTests(unittest.TestCase):
    def test_finite_absorbing_chain_and_selective_second_review(self):
        c=dict(e=283,n=1279,r1_ok=222,r1_n=283,r2_ok=42,r2_n=61,before_e=50,before_n=167,after_e=233,after_n=1112)
        p=h.params(c)
        for length in (16,64):
            expected=(1-19/1279)**length
            self.assertAlmostEqual(h.predict(p,length)["main"],expected,places=13)
            # Independent enumeration of all 3 feedback paths for one step.
            restored=(1-p["error"])+p["error"]*p["r1"]+p["error"]*(1-p["r1"])*p["r2"]
            self.assertAlmostEqual(restored**length,expected,places=13)
        self.assertNotAlmostEqual((1-p["r1"])**2,(1-p["r1"])*(1-p["r2"]))
    def test_length_schedule_and_independent_gold(self):
        rows=h.schedule()
        self.assertEqual(len(rows),40);self.assertEqual(len({x["seed"] for x in rows}),40)
        self.assertEqual(collections.Counter(x["length"] for x in rows),{16:20,64:20})
        self.assertEqual([x["length"] for x in rows[:2]],[16,64])
        for row in rows:
            t=d.task(row);self.assertEqual(reference.oracle(t),t["prefix"])
            short=d.task(dict(row,length=16));self.assertEqual(short["prefix"],t["prefix"][:17])
    def test_cap_2000_and_restart_count_preserved(self):
        with tempfile.TemporaryDirectory(dir=h.ROOT) as tmp:
            out=Path(tmp);b=h.Budget(out,mock=True)
            for _ in range(17):b.reserve("a","sheet_d0")
            b.journal.f.close();restored=h.Budget(out,mock=True,resume=True)
            self.assertEqual(restored.calls,17);self.assertEqual(restored.reserve("b","sheet_d0"),18)
            restored.calls=2000
            with self.assertRaisesRegex(d.Stop,"hard_cap"):restored.reserve("b","sheet_d0")
            restored.journal.f.close()
    def test_feedback_stage_statistics_keep_detection_separate(self):
        ev=[]
        for stage,ok,detect in [(0,False,False),(1,False,False),(2,True,True),(0,False,False)]:
            ev.append(dict(event="action",review=stage,correct=ok,step=0,n_before=1,n_after=0 if ok else 2,action={"detect":detect}))
        c=h.stats(ev)
        self.assertEqual((c["e"],c["n"],c["r1_n"],c.get("r1_detect",0),c["r2_ok"]),(2,2,1,0,1))
        self.assertEqual((c["next_repaired_n"],c["next_repaired_e"]),(1,1))
    def test_predictive_distribution(self):
        p=dict(error=.2,r1=.75,r2=.5,before=.2,after=.2)
        result=h.predictive_interval([p],16,20)
        self.assertAlmostEqual(sum(result["pmf"]),1)
        lo,hi=result["counts"]
        self.assertGreaterEqual(sum(result["pmf"][lo:hi+1]),.975)
        self.assertAlmostEqual(h.predict(p,16)["main"],h.predict(p,16)["history_two_state"])

if __name__=="__main__":unittest.main()
