import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('supplement/two_step_adam_certificate')))
from check_adam import exact_cover,box,stored,signals,pos
base=Path('supplement/two_step_adam_certificate/reference')
reg=json.loads((base/'stabilizer_ordering.json').read_text());cases=[]
def reject(name,fn):
    try:fn()
    except (AssertionError,ValueError,ZeroDivisionError):cases.append(dict(case=name,rejected=True))
    else:raise AssertionError('Accepted corrupted input: '+name)
rows=copy.deepcopy(reg['leaves']);rows.pop();reject('missing_rectangle',lambda:exact_cover(rows))
rows=copy.deepcopy(reg['leaves']);rows.append(rows[0]);reject('overlapping_rectangle',lambda:exact_cover(rows))
rows=copy.deepcopy(reg['leaves']);rows[0]['a'][0]='1/10000';reject('interior_gap',lambda:exact_cover(rows))
x=copy.deepcopy(reg['leaves'][0]);x['integrals'][2]['lo']=str(-1000*2**160);x['integrals'][2]['hi']=str(-1000*2**160)
reject('flipped_derivative_integral',lambda:pos(signals(box(*x['a']),[stored(v) for v in x['integrals']])['N']))
x=copy.deepcopy(reg['leaves'][0]['integrals'][0]);x['power']=159;reject('wrong_fixed_point_scale',lambda:stored(x))
result=dict(status='PASS',cases=cases)
Path('PKG_negative_checks.json').write_text(json.dumps(result,indent=2))
Path('supplement/logs/negative_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
