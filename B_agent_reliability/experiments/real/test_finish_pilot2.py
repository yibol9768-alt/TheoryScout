import copy
import unittest
from finish_pilot2 import todo,fatal
from pilot2 import schedule

class FinishTests(unittest.TestCase):
    def test_only_unattempted_suffix(self):
        b=dict(halt='transport_or_response_error',schedule=schedule(),results=copy.deepcopy(schedule()[:10]))
        self.assertEqual(todo(b),schedule()[10:]);self.assertEqual(len(todo(b)),14)
        self.assertLessEqual(159+14*20,500)
    def test_no_wrong_model_continuation(self):
        b=dict(halt='model_mismatch',schedule=schedule(),results=schedule()[:10])
        with self.assertRaises(ValueError):todo(b)
        self.assertTrue(fatal('model_mismatch'))
    def test_transport_failure_is_not_retried_or_global(self):
        self.assertFalse(fatal('transport_or_response_error'))
        self.assertFalse(fatal('http_status_500'))
        self.assertTrue(fatal('missing_usage'));self.assertTrue(fatal('global_call_limit'))
    def test_cannot_replace_a_seed(self):
        b=dict(halt='transport_or_response_error',schedule=schedule(),results=copy.deepcopy(schedule()[:10]))
        b['results'][2]['seed']=0
        with self.assertRaises(ValueError):todo(b)

if __name__=='__main__':unittest.main(verbosity=2)
