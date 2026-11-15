import copy
import unittest
from resume_pilot2 import remaining
from pilot2 import schedule

class ResumeTests(unittest.TestCase):
    def base(self):
        rows=copy.deepcopy(schedule()[:9])
        for row in rows: row['status']='success'
        rows[-1]['status']='transport_or_response_error'
        return dict(halt='transport_or_response_error',schedule=schedule(),results=rows)
    def test_failed_slot_not_retried(self):
        b=self.base();todo=remaining(b)
        self.assertEqual(todo,schedule()[9:]);self.assertEqual(len(todo),15)
        self.assertNotIn(schedule()[8],todo)
        self.assertLessEqual(147+len(todo)*20,500)
    def test_wrong_model_cannot_resume(self):
        b=self.base();b['halt']='model_mismatch'
        with self.assertRaises(ValueError):remaining(b)
    def test_prefix_cannot_change(self):
        b=self.base();b['results'][0]['seed']+=1
        with self.assertRaises(ValueError):remaining(b)
    def test_transport_failure_required(self):
        b=self.base();b['results'][-1]['status']='success'
        with self.assertRaises(ValueError):remaining(b)

if __name__=='__main__':unittest.main(verbosity=2)
