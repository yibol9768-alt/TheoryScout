from pathlib import Path
import subprocess,sys,json,time
base=Path('supplement').resolve();dest=base/'full_experiments_recomputation'
assert all(json.loads((dest/name/'result.json').read_text())['status']=='PASS' for name in ['matched','confirmation'])
job=['experiments_replay/recompute_tables.py','--matched-measurements','full_experiments_recomputation/matched/measurements.csv','--confirmation-measurements','full_experiments_recomputation/confirmation/measurements.csv','--output','full_experiments_recomputation/tables']
start=time.monotonic()
with (dest/'table_recompute.log').open('x') as log:p=subprocess.run([sys.executable,'-B','-u',*job],cwd=base,stdout=log,stderr=subprocess.STDOUT)
result=dict(command='python '+' '.join(job),seconds=time.monotonic()-start,exit_code=p.returncode)
(dest/'table_timings.json').write_text(json.dumps([result],indent=2));assert p.returncode==0,(dest/'table_recompute.log').read_text()
print(json.dumps(result))
