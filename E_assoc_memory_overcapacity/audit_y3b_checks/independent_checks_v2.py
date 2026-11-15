"""保留首次失败；仅修复审计脚本的导入限额与输出位置。"""
from pathlib import Path
import resource,signal
signal.signal(signal.SIGPROF,signal.SIG_DFL)
signal.setitimer(signal.ITIMER_PROF,90)
p=Path(__file__).with_name('independent_checks.py')
source=p.read_text()
source=source.replace('resource.setrlimit(resource.RLIMIT_CPU, (90, 100))',
                      'resource.setrlimit(resource.RLIMIT_CPU, (90, 21600))')
source=source.replace("OUT=ROOT/'audit_y3b_checks'", "OUT=ROOT/'audit_y3b_checks'/'independent_v2'; OUT.mkdir()")
source=source.replace('or OUT in p.parents or ROOT/', "or ROOT/'audit_y3b_checks' in p.parents or ROOT/")
exec(compile(source,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
