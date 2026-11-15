from pathlib import Path
import sys,runpy,json
project=Path.cwd().resolve();allowed=Path(sys.argv[1]).resolve();script=Path(sys.argv[2]).resolve();args=sys.argv[3:]
assert allowed.is_relative_to(project) and script.is_relative_to(allowed)
read_count=0

def audit(event,args):
    global read_count
    if event in ['open','os.listdir','os.scandir'] and args and isinstance(args[0],(str,bytes)):
        value=Path(args[0]).resolve()
        if value.is_relative_to(project):
            assert value.is_relative_to(allowed), 'Read outside relocated package: '+str(value)
            read_count+=1
sys.addaudithook(audit)
sys.path.insert(0,str(script.parent));sys.argv=[str(script),*args]
runpy.run_path(str(script),run_name='__main__')
print(json.dumps(dict(relocation_audit='PASS',package_accesses=read_count,original_project_accesses=0)))
