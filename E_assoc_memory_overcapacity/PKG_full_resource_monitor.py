import json,subprocess,sys,time
from pathlib import Path
unit=sys.argv[1];base=Path('supplement/full_experiments_recomputation');latest=None;maxthreads=0
while True:
    proc=subprocess.run(['systemctl','--user','show',unit,'-p','ControlGroup','--value'],capture_output=True,text=True)
    cg=proc.stdout.strip()
    if not cg:break
    path=Path('/sys/fs/cgroup')/cg.lstrip('/')
    try:
        values={k:(path/k).read_text().strip() for k in ['memory.max','memory.swap.max','memory.peak','memory.events']}
        assert values['memory.max']=='3221225472' and values['memory.swap.max']=='0'
        for pid in (path/'cgroup.procs').read_text().split():
            tasks=Path('/proc')/pid/'task'
            try:count=len(list(tasks.iterdir()))
            except FileNotFoundError:continue
            maxthreads=max(maxthreads,count)
            assert count<=2,(pid,count)
        latest=dict(resource_limits=values,max_threads_per_process=maxthreads)
        Path('PKG_full_resources.json').write_text(json.dumps(latest,indent=2))
    except FileNotFoundError:break
    if (base/'matched/result.json').exists() and (base/'confirmation/result.json').exists():break
    time.sleep(10)
assert latest is not None
print(json.dumps(latest))
