import sys
from common import *
guards()
sys.path.insert(0,str(RUNNER))
import run
def forbidden(*a,**k):raise AssertionError('CONSTRUCTED_BEFORE_REJECTION')
run.Engine=run.HTTPTransport=run.HTTPQuota=run.CommandQuota=forbidden
args=sys.argv[1:]
if args and args[0]=='-B':args=args[1:]
if args and args[0].endswith('run.py'):args=args[1:]
sys.argv=['run.py',*args]
try:run.main()
except (ValueError,SystemExit) as ex:
    print('REJECTED_BEFORE_CONSTRUCTION',repr(ex));raise SystemExit(2)
raise AssertionError('ENTRY_RETURNED_WITHOUT_REJECTION')
