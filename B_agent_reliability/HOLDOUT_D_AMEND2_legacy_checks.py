"""在只读/禁连接哨兵下重跑两套旧冻结数学与输入核验。"""
import os
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parent
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
        raise RuntimeError('legacy_validation_read_only')
    if event in ('socket.connect','socket.getaddrinfo','os.mkdir','os.remove','os.rmdir','os.rename'):
        raise RuntimeError('legacy_validation_no_mutation_or_network')
sys.addaudithook(audit)
for name,args in [('HOLDOUT_D_validate.py',[]),('HOLDOUT_D_AMEND1_validate.py',['--check-only'])]:
    sys.argv=[name,*args]
    runpy.run_path(str(ROOT/name),run_name='__main__')
