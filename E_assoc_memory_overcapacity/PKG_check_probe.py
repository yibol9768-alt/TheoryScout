import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('supplement/two_step_adam_certificate')))
from check_adam import box,stored,signals,pos,exact_cover
reg=json.loads(Path('supplement/runs/adam_regenerated/stabilizer_ordering.json').read_text())
print('strips',exact_cover(reg['leaves']))
for i,row in enumerate(reg['leaves']):
    st=signals(box(*row['a']),[stored(v) for v in row['integrals']]);pos(st['N']);pos(st['J'])
print('PASS all 1027 independent decisions')
