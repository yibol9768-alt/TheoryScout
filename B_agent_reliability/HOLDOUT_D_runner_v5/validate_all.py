"""在指定 systemd scope 内运行；输出独占创建，禁止覆盖证据。"""
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
checks=HERE/'amend5_checks'
env=dict(os.environ,PYTHONPATH=str(checks),PYTHONDONTWRITEBYTECODE='1')
os.execve(sys.executable,[sys.executable,'-B',str(checks/'run_all.py')],env)
