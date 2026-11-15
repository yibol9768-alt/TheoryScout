"""子进程无网络哨兵：shell 与直接 Python 测试共用，不创建 Engine/客户端。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run
def forbidden(*args,**kwargs):raise AssertionError('CLIENT_OR_ENGINE_CONSTRUCTED')
run.Engine=run.HTTPTransport=run.HTTPQuota=forbidden
def audit(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):raise AssertionError('NETWORK_ATTEMPT')
sys.addaudithook(audit)
args=sys.argv[1:]
if args and args[0]=='-B':args=args[1:]
if args and args[0].endswith('run.py'):args=args[1:]
sys.argv=['run.py',*args]
try:run.main()
except (ValueError,SystemExit) as ex:
    print('BLOCKED_BEFORE_CLIENTS',str(ex));raise SystemExit(2)
raise AssertionError('unexpected_entry_success')
