import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path('supplement/two_step_adam_certificate')))
from check_adam import box,reintegrate,signals,pos,encode
rows=json.loads(Path('supplement/runs/adam_regenerated/stabilizer_ordering.json').read_text())['leaves']
result=[]
for i in [0,512,1026]:
    row=rows[i];start=time.monotonic();a=box(*row['a']);e=box(*row['e']);v=reintegrate(a,e,64,4);s=signals(a,v);pos(s['N']);pos(s['J'])
    r=dict(index=i,seconds=time.monotonic()-start,N=encode(s['N']),J=encode(s['J']));result.append(r);print(r,flush=True)
Path('PKG_iv_probe.json').write_text(json.dumps(result,indent=2))
