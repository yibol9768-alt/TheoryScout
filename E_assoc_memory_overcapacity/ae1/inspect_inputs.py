import json
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parents[1]
for name in ['z4b/formal/cal_128_0_0.npz','z4b/formal/eval_128_0.npz']:
 p=r/name
 if not p.exists():
  print('missing', name);continue
 with np.load(p,allow_pickle=False) as f:
  print(name,{k:(f[k].shape,str(f[k].dtype)) for k in f.files})
  for k in f.files:
   if f[k].size<25: print(k,f[k].tolist())
o=json.loads((r/'z4b/formal/lock.json').read_text())
print('lock keys',list(o))
print('lock excluding sources',{k:v for k,v in o.items() if k!='source_hashes'})
o=json.loads((r/'z4b/formal/ACCEPTED.json').read_text());print('accepted keys',list(o))
print('raw W4 keys', list(json.loads((r/'experiments/w4/W4_raw.jsonl').read_text().splitlines()[0])))
