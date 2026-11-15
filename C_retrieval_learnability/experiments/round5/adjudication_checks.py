"""Independent synthetic boundary checks of the frozen assessor; no experiment tuning."""
from copy import deepcopy
from run import grid
from assess import assess
import experiment as c

def fixture(category):
    values={'initialization':dict(ZL=False,ZM=False,SL=True,SM=True),
            'learning_rate':dict(ZL=False,ZM=True,SL=False,SM=True),
            'each_sufficient':dict(ZL=False,ZM=True,SL=True,SM=True),
            'joint_required':dict(ZL=False,ZM=False,SL=False,SM=True)}[category]
    rows=[]
    for q in grid():
        yes=values[q['variant']] if q['stage']=='A' else q['variant']!='b_original'
        rows.append(dict(config=q,status='completed',steps=c.T,requested=c.T,
                         final=dict(all_sign=yes,robust_sign=yes,min_zeta=.001 if yes else -.001,rank_fraction=1.),
                         late_loss=bool(q['stage']=='B' and q['variant']=='b_original')))
    return rows

checks={}
for category in ('initialization','learning_rate','each_sufficient','joint_required'):
    rows=fixture(category); a=assess(rows)
    assert a['gate'] and a['H_A'] and a['A_d6']['category']==category and a['A_d7']['category']==category
    bad=deepcopy(rows)
    for r in bad:
        if r['config']['stage']=='B' and r['config']['variant']=='b_stop80': r['final']['robust_sign']=False
    assert not assess(bad)['H_B1'] and not assess(bad)['gate'] and assess(bad)['H_B2']
    missing=rows[:-1]; assert not assess(missing)['gate']
    broken=deepcopy(rows); broken[0]['status']='budget_truncated'; assert not assess(broken)['gate']
    checks[category]=True
c.save(c.OUT/'adjudication_checks.json',dict(pass_all=True,checks=checks,description='Synthetic threshold and gate checks; does not train or alter frozen assessor.'))
print(checks)
