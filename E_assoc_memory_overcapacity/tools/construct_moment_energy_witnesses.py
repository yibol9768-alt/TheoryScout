"""Exact atomic witnesses compatible with C0 moment intervals, not its geometry.

The threshold remains the certified spherical order law. LP selects a finite
support; rational elimination reconstructs all probabilities exactly.
"""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime,timezone
from bisect import bisect_left,bisect_right
import argparse
import hashlib
import json
import time

import certify_weighted_chisquare as integer

S=integer.SCALE
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def midpoint(r):return F(int(r['lower_integer'])+int(r['upper_integer']),2*S)


def cdf_bound(row,d,energy,side):
    """CDF at d/sqrt(energy), d>=0, using exact rational squared comparisons."""
    assert d>=0 and energy>0
    if '_prepared_r39' not in row:
        row['_prepared_r39']=([F(t)**2 for t in row['t']],list(map(F,row['sweeps'][-1]['lower_CDF'])),list(map(F,row['sweeps'][-1]['upper_CDF'])))
    squared,lower,upper=row['_prepared_r39'];ratio=d*d/energy
    if side=='lower':
        index=bisect_right(squared,ratio)-1
        return F(0) if index<0 else lower[index]
    assert side=='upper'
    index=bisect_left(squared,ratio)
    return F(1) if index==len(squared) else upper[index]


def targets(root):
    moments=read(root/'revisions/r28_conditional_moments_v2/RESULT.json')['C0_moments']
    selected={key:midpoint(moments[key]) for key in ['mean_D','variance_D','third_central_D','fourth_central_D','mean_V','variance_V','covariance_DV']}
    d0=F(3834524,1000000);v0=F(1056992,1000000);x=selected['mean_D']-d0;y=selected['mean_V']-v0
    vd=selected['variance_D'];m3=selected['third_central_D'];m4=selected['fourth_central_D'];vv=selected['variance_V'];cv=selected['covariance_DV']
    root_mean=F(1027,1000);r31=read(root/'revisions/r31_mean_margin_v3/RESULT.json')
    interval=r31['energy']['mean_sqrt_energy'];assert F(int(interval['lower_integer']),S)<=root_mean<=F(int(interval['upper_integer']),S)
    values=[F(1),x,vd+x*x,m3+3*x*vd+x**3,m4+4*x*m3+6*x*x*vd+x**4,
            10*y,100*(vv+y*y),10*(cv+x*y),
            10000*(root_mean-1-(selected['mean_V']-1)/2+(vv+(selected['mean_V']-1)**2)/8)]
    return dict(selected_moments={k:str(v) for k,v in selected.items()},mean_sqrt_V=str(root_mean),
        D_reference=str(d0),V_reference=str(v0),moment_targets=list(map(str,values)))


def features(d,s,design):
    v=s*s;x=d-F(design['D_reference']);y=10*(v-F(design['V_reference']))
    return [F(1),x,x*x,x**3,x**4,y,y*y,x*y,10000*(s-1-(v-1)/2+(v-1)**2/8)]


def contrast_bounds(row,d,s,meanv):
    v=s*s
    if d==0 or v==meanv:return F(0),F(0)
    low=cdf_bound(row,d,v,'lower')-cdf_bound(row,d,meanv,'upper')
    high=cdf_bound(row,d,v,'upper')-cdf_bound(row,d,meanv,'lower')
    return (max(F(0),low),high) if v<meanv else (low,min(F(0),high))


def solve_exact(columns,target):
    count=len(target);assert len(columns)==count
    a=[[columns[j][i] for j in range(count)]+[target[i]] for i in range(count)]
    for j in range(count):
        pivot=next(i for i in range(j,count) if a[i][j]);a[j],a[pivot]=a[pivot],a[j]
        value=a[j][j];a[j]=[x/value for x in a[j]]
        for i in range(count):
            if i!=j:
                value=a[i][j];a[i]=[x-value*y for x,y in zip(a[i],a[j])]
    return [row[-1] for row in a]


def point_constraints(root,candidates):
    lower=read(root/'revisions/r26_grid_nodes/RESULT.json');upper=read(root/'revisions/r36_right_tail/RESULT.json')
    rows=[];inset=F(1,10**7)
    for t,h in zip(lower['t'],lower['merged_heights']):
        values=[-F(d>=s*F(t)) for d,s in candidates]
        rows.append(dict(t=t,side='lower',kind='normalized_score',values=values,original_rhs=-F(h),rhs=-F(h)-inset))
    for t,h in upper['all_point_uppers'].items():
        values=[F(d>=s*F(t)) for d,s in candidates]
        rows.append(dict(t=t,side='upper',kind='normalized_score',values=values,original_rhs=F(h),rhs=F(h)-inset))
    for node in read(root/'revisions/r32_frozen_energy/RESULT.json')['nodes']:
        t=node['threshold'];side=node['side'];h=F(node['C0_survival_bound'])
        sign=-1 if side=='lower' else 1
        values=[sign*F(d>=F(t)) for d,s in candidates]
        rows.append(dict(t=t,side=side,kind='D_survival',values=values,original_rhs=sign*h,rhs=sign*h-inset))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic()
    groups={'R28':['revisions/r28_conditional_moments_v2/RESULT.json'],
            'R31':['revisions/r31_mean_margin_v3/RESULT.json'],
            'R23':['revisions/r23_cdf_dense8/RESULT.json'],
            'R26':['revisions/r26_grid_nodes/RESULT.json'],
            'R36':['revisions/r36_right_tail/RESULT.json'],
            'R37':['revisions/r37_joint_radial_v2/RESULT.json'],
            'R32':['revisions/r32_frozen_energy/RESULT.json']}
    hashes={}
    for rev,names in groups.items():
        receipt=read(root/f'revisions/{rev}_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
        for name in names:assert sha(root/name)==manifest['files'][name];hashes[name]=sha(root/name)
    design=targets(root);design.update(source_hashes=hashes,created_utc=datetime.now(timezone.utc).isoformat(),
        probability_laws_are_not_C0_spherical_quadratic_realizations=True,
        selected_midpoints_not_asserted_to_be_exact_C0_moments=True)
    save(args.out/'DESIGN.json',design)
    ds={F(j,5) for j in range(5,37)}
    for node in read(root/'revisions/r32_frozen_energy/RESULT.json')['nodes']:
        t=F(node['threshold']);floor=t.numerator*10000//t.denominator
        ds.update(F(floor+j,10000) for j in [0,1])
    candidates=[(d,F(k,100)) for d in sorted(ds) for k in range(90,116)]
    columns=[features(d,s,design) for d,s in candidates];target=list(map(F,design['moment_targets']))
    inequalities=point_constraints(root,candidates)
    import numpy as np
    from scipy.optimize import linprog
    matrix=np.array([[float(col[j]) for col in columns] for j in range(9)]);rhs=np.array(list(map(float,target)))
    upper_matrix=np.array([[float(v) for v in c['values']] for c in inequalities]);upper_rhs=np.array([float(c['rhs']) for c in inequalities])
    meanv=F(design['selected_moments']['mean_V']);rows=[]
    for row in read(root/'revisions/r23_cdf_dense8/RESULT.json')['rows']:
        bounds=[contrast_bounds(row,d,s,meanv) for d,s in candidates]
        witnesses=[]
        for direction,index,sign in [('positive',0,-1),('negative',1,1)]:
            objective=np.array([sign*float(v[index]) for v in bounds])
            fit=linprog(objective,A_eq=matrix,b_eq=rhs,A_ub=upper_matrix,b_ub=upper_rhs,bounds=(0,None),method='highs',options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9));assert fit.success,fit.message
            support=[j for j,x in enumerate(fit.x) if x>1e-9]
            selected=[];basis=matrix[:,support]
            for k,residual in enumerate(fit.ineqlin.residual):
                if residual<1e-8 and len(basis)<len(support):
                    proposal=np.vstack([basis,upper_matrix[k,support]])
                    if np.linalg.matrix_rank(proposal,tol=1e-10)>len(basis):selected.append(k);basis=proposal
            assert len(basis)==len(support)
            exact_columns=[columns[j]+[inequalities[k]['values'][j] for k in selected] for j in support]
            weights=solve_exact(exact_columns,target+[inequalities[k]['rhs'] for k in selected]);assert min(weights)>0 and sum(weights)==1
            for constraint in inequalities:assert sum(w*constraint['values'][j] for j,w in zip(support,weights))<=constraint['original_rhs']
            atoms=[dict(D=str(candidates[j][0]),sqrt_V=str(candidates[j][1]),probability=str(w)) for j,w in zip(support,weights)]
            lo=sum(w*bounds[j][0] for j,w in zip(support,weights));hi=sum(w*bounds[j][1] for j,w in zip(support,weights))
            witnesses.append(dict(direction=direction,atoms=atoms,selected_active_point_constraints=selected,
                signed_energy_contrast_interval=list(map(str,[lo,hi])),approximate=[float(lo),float(hi)],sign_certified=lo>0 if direction=='positive' else hi<0))
            print(row['L'],direction,'contrast',float(lo),float(hi),flush=True)
        rows.append(dict(L=row['L'],witnesses=witnesses))
    save(args.out/'RESULT.json',dict(rows=rows,source_hashes=hashes,candidate_atoms=len(candidates),
        all_26_lower_and_20_upper_score_point_constraints_satisfied=True,
        all_8_R32_D_survival_constraints_satisfied=True,
        selected_moments=design['selected_moments'],mean_sqrt_V=design['mean_sqrt_V'],
        same_spherical_order_threshold_as_actual_model=True,actual_C0_geometry_not_realized=True,
        actual_C0_energy_remainder_not_determined=True,new_samples=0,new_matrix_or_probability_integrals=0,
        independent_proof_review=False,wall_seconds=time.monotonic()-start))


if __name__=='__main__':main()
