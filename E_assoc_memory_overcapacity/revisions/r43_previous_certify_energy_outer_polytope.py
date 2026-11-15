"""Actual-C0 energy contrast bounds from a containing cell-mass polytope.

Tail mass, all moment intervals and every cell/event envelope are outward.
Floating LP selects multipliers only; a rational simplex residual certifies.
"""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime,timezone
import argparse
import hashlib
import json
import math
import time

import certify_weighted_chisquare as integer
from construct_moment_energy_witnesses import cdf_bound

S=integer.SCALE
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def restore(r):return integer.I.raw(int(r['lower_integer']),int(r['upper_integer']))
def lo(r):return F(r.lo,S)
def hi(r):return F(r.hi,S)


def linear_range(a,b,c,box):
    dl,dh,vl,vh=box
    return min(a*dl,a*dh)+min(b*vl,b*vh)+c,max(a*dl,a*dh)+max(b*vl,b*vh)+c


def square_min(a,b,c,box):
    x,y=linear_range(a,b,c,box)
    return F(0) if x<=0<=y else min(x*x,y*y)


def kernel_bounds(row,box,meanv,negative):
    dl,dh,vl,vh=box;ml,mh=meanv
    if dh<=0:return -negative,negative,F(0),negative
    assert dl>=0
    lower=cdf_bound(row,dl,vh,'lower')-cdf_bound(row,dh,ml,'upper')
    upper=cdf_bound(row,dh,vl,'upper')-cdf_bound(row,dl,mh,'lower')
    if vh<=ml:lower=max(F(0),lower)
    if vl>=mh:upper=min(F(0),upper)
    assert lower<=upper
    return lower,upper,cdf_bound(row,dl,vh,'lower'),cdf_bound(row,dh,vl,'upper')


def build(root):
    inputs=read(root/'revisions/r28_conditional_moments_v2/INPUTS.json');moment_source=read(root/'revisions/r28_conditional_moments_v2/RESULT.json');mom=moment_source['C0_moments']
    bad=read(root/'revisions/r27_recall_upper/BAD_EVENTS.json');n=inputs['n'];m=n-1
    assert moment_source['n']==n and moment_source['q']==inputs['q']
    assert bad['n']==n
    eta=F(inputs['eta']);nu=F(inputs['N_operator_perturbation']);kappa=F(inputs['common_K_norm'])
    ev=F(n*n,m)*(nu+2*kappa*eta);delta=F(bad['union_upper'])
    assert F(bad['center'])-F(bad['radius'])-n*eta> -3 and F(bad['center'])+F(bad['radius'])+n*eta<10
    assert F(bad['l0'])**2-ev>F(19,25) and F(bad['u0'])**2+ev<F(139,100)
    ds=[F(-3),F(-2),F(-1),F(0),F(2)]+[F(j,20) for j in range(41,141)]+[F(10)]
    vs=[F(j,100) for j in range(76,140)]
    boxes=[(a,b,c,d) for a,b in zip(ds,ds[1:]) for c,d in zip(vs,vs[1:])];count=len(boxes)+1
    negative=F(1,1<<100);mean=restore(mom['mean_V']);meanv=(lo(mean),hi(mean))
    cdf_source=read(root/'revisions/r23_cdf_dense8/RESULT.json');cdf=cdf_source['rows'];kernels={}
    sphere=root/'revisions/r21_sphere_cdf/RESULT.json'
    assert sha(sphere)==cdf_source['source_hashes']['revisions/r21_sphere_cdf/RESULT.json'] and read(sphere)['m']==m
    for row in cdf:assert (1<<100)*sum(math.comb(m,j) for j in range(row['L']))<=(1<<m)
    for row in cdf:kernels[row['L']]=[kernel_bounds(row,box,meanv,negative) for box in boxes]+[(F(-1),F(1),F(0),F(1))]
    rows=[]
    def add(label,a,b):
        assert len(a)==count;rows.append(dict(label=label,coefficients=a,rhs=F(b)))
    add('tail_mass_upper',[F(0)]*len(boxes)+[F(1)],delta)
    def event_bounds(kind,t,c=None):
        low=[];upper=[]
        for dl,dh,vl,vh in boxes:
            if kind=='normalized':
                l=dl>=0 and dl*dl>=t*t*vh;u=dh>=0 and dh*dh>=t*t*vl
            elif kind=='score':l=dl>=t;u=dh>=t
            else:
                assert c>F(20,n)
                l=c*dl-(dl*dl+m*vh)/n>=t;u=c*dh-(dh*dh+m*vl)/n>=t
            low.append(F(l));upper.append(F(u))
        return low+[F(0)],upper+[F(1)]
    lower=read(root/'revisions/r26_grid_nodes/RESULT.json')
    for index,(t,h) in enumerate(zip(lower['t'],lower['merged_heights'])):
        a,b=event_bounds('normalized',F(t));add(f'normalized_lower_{index}',[-x for x in b],-F(h))
    for index,(t,h) in enumerate(read(root/'revisions/r36_right_tail/RESULT.json')['all_point_uppers'].items()):
        a,b=event_bounds('normalized',F(t));add(f'normalized_upper_{index}',a,F(h))
    for index,node in enumerate(read(root/'revisions/r32_frozen_energy/RESULT.json')['nodes']):
        a,b=event_bounds('score',F(node['threshold']));value=F(node['C0_survival_bound'])
        if node['side']=='lower':add(f'D_survival_lower_{index}',[-x for x in b],-value)
        else:add(f'D_survival_upper_{index}',a,value)
    old_joint=read(root/'revisions/r41_joint_pencils/RESULT.json');new_joint=read(root/'revisions/r42_joint_thresholds/RESULT.json')
    assert old_joint['n']==new_joint['n']==n and old_joint['q']==new_joint['q']==inputs['q']
    assert old_joint['source_hashes']['revisions/r11_anchor_certificate_v2/actual_construction.bin']==inputs['source_hashes']['revisions/r11_anchor_certificate_v2/actual_construction.bin']
    joints=[dict(c=r['c'],y='0',interval=r['probability_C0_positive']) for r in old_joint['rows']]
    joints += [dict(c=r['c'],y=r['y'],interval=r['C0_survival_interval']) for r in new_joint['rows']]
    for index,node in enumerate(joints):
        a,b=event_bounds('joint',F(node['y']),F(node['c']));l,u=map(F,node['interval'])
        add(f'joint_lower_{index}',[-x for x in b],-l);add(f'joint_upper_{index}',a,u)
    for previous in read(root/'revisions/r37_joint_radial_v2/RESULT.json')['rows']:
        values=kernels[previous['L']]
        add(f'recall_upper_L{previous["L"]}',[v[2] for v in values],F(previous['conditional_recall_upper']))
        add(f'recall_lower_L{previous["L"]}',[-v[3] for v in values],-F(previous['conditional_recall_lower']))
    d0=F(3834524,1000000);v0=F(1056992,1000000)
    md=restore(mom['mean_D']);mv=restore(mom['mean_V']);vd=restore(mom['variance_D']);vv=restore(mom['variance_V']);cov=restore(mom['covariance_DV'])
    for a,b in [(1,0),(0,10),(1,5),(1,-5),(1,10),(1,-10)]:
        c=-a*d0-b*v0;expect=a*a*vd+2*a*b*cov+b*b*vv+(a*md+b*mv+c).square()
        add(f'moment_square_{a}_{b}',[square_min(F(a),F(b),c,box) for box in boxes]+[F(0)],hi(expect))
    offset=md-d0;fourth=restore(mom['fourth_central_D'])+4*offset*restore(mom['third_central_D'])+6*offset.square()*vd+offset.square().square()
    add('fourth_D_moment',[square_min(F(1),F(0),-d0,box)**2 for box in boxes]+[F(0)],hi(fourth))
    for label,a,b,center,expect,second in [('D',F(1),F(0),d0,md-d0,vd+(md-d0).square()),('V',F(0),F(1),v0,mv-v0,vv+(mv-v0).square())]:
        tolerance=hi((second*delta).sqrt());bounds=[linear_range(a,b,-center,box) for box in boxes]
        add('centered_mean_upper_'+label,[x[0] for x in bounds]+[F(0)],hi(expect)+tolerance)
        add('centered_mean_lower_'+label,[-x[1] for x in bounds]+[F(0)],-lo(expect)+tolerance)
    design=dict(n=n,box_D=[str(ds[0]),str(ds[-1])],box_V=[str(vs[0]),str(vs[-1])],D_grid=list(map(str,ds)),V_grid=list(map(str,vs)),
        good_box_cells=len(boxes),tail_cell=True,tail_mass_upper=str(delta),V_transfer_error=str(ev),negative_threshold_upper=str(negative),
        mean_V_interval=list(map(str,meanv)),constraints=len(rows),constraint_labels=[r['label'] for r in rows])
    return design,rows,kernels


def simplex_dual(cost,rows,multipliers):
    residual=cost.copy();constant=F(0)
    for index,weight in multipliers:
        weight=F(weight);assert weight>=0 and 0<=index<len(rows)
        row=rows[index];constant-=weight*row['rhs']
        for j,x in enumerate(row['coefficients']):residual[j]+=weight*x
    minimum=min(residual)
    return constant+minimum,dict(constant=str(constant),minimum_simplex_residual=str(minimum),minimum_index=residual.index(minimum))


def frozen_recall_bounds(nodes,row,meanv):
    """R33 monotone survival transfer, here using squared RMS comparisons."""
    ml,mh=meanv
    lower=sorted((F(x['threshold']),F(x['C0_survival_bound'])) for x in nodes if x['side']=='lower')
    upper=sorted((F(x['threshold']),F(x['C0_survival_bound'])) for x in nodes if x['side']=='upper')
    levels=[max(v for _,v in lower[j:]) for j in range(len(lower))]+[F(0)];bottom=F(0)
    for j,(x,_) in enumerate(lower):bottom+=(levels[j]-levels[j+1])*cdf_bound(row,x,mh,'lower')
    previous=F(1);top=F(0)
    for x,value in upper:
        current=min(previous,value);top+=(previous-current)*cdf_bound(row,x,ml,'upper');previous=current
    top+=previous;assert 0<=bottom<=top<=1
    return bottom,top


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic()
    names=['revisions/r28_conditional_moments_v2/INPUTS.json','revisions/r28_conditional_moments_v2/RESULT.json','revisions/r27_recall_upper/BAD_EVENTS.json','revisions/r23_cdf_dense8/RESULT.json','revisions/r26_grid_nodes/RESULT.json','revisions/r36_right_tail/RESULT.json','revisions/r32_frozen_energy/RESULT.json','revisions/r37_joint_radial_v2/RESULT.json','revisions/r41_joint_pencils/RESULT.json','revisions/r42_joint_thresholds/RESULT.json','tools/certify_weighted_chisquare.py','tools/construct_moment_energy_witnesses.py']
    names.append('revisions/r21_sphere_cdf/RESULT.json')
    hashes={name:sha(root/name) for name in names};design,rows,kernels=build(root)
    save(args.out/'DESIGN.json',dict(**design,created_utc=datetime.now(timezone.utc).isoformat(),source_hashes=hashes))
    import numpy as np
    from scipy.optimize import linprog
    matrix=np.array([[float(x) for x in r['coefficients']] for r in rows]);rhs=np.array([float(r['rhs']) for r in rows]);answer=[]
    for length,values in kernels.items():
        proofs={}
        for side,index,sign in [('lower',0,1),('upper',1,-1)]:
            cost=[sign*v[index] for v in values]
            fit=linprog(np.array(list(map(float,cost))),A_ub=matrix,b_ub=rhs,A_eq=np.ones((1,len(cost))),b_eq=[1.],bounds=(0,None),method='highs')
            assert fit.success,fit.message
            multipliers=[(j,str(F.from_float(float(-v)))) for j,v in enumerate(fit.ineqlin.marginals) if v<0]
            bound,parts=simplex_dual(cost,rows,multipliers)
            proofs[side]=dict(multipliers=multipliers,parts=parts,certified_bound=str(sign*bound),float_optimum_diagnostic=float(sign*fit.fun))
        low=F(proofs['lower']['certified_bound']);high=F(proofs['upper']['certified_bound']);assert -1<=low<=high<=1
        cdf=next(row for row in read(root/'revisions/r23_cdf_dense8/RESULT.json')['rows'] if row['L']==length)
        qrl,qru=frozen_recall_bounds(read(root/'revisions/r32_frozen_energy/RESULT.json')['nodes'],cdf,list(map(F,design['mean_V_interval'])))
        prior=next(row for row in read(root/'revisions/r37_joint_radial_v2/RESULT.json')['rows'] if row['L']==length)
        unpaired=[F(prior['conditional_recall_lower'])-qru,F(prior['conditional_recall_upper'])-qrl]
        final=[max(low,unpaired[0]),min(high,unpaired[1])];assert final[0]<=final[1]
        answer.append(dict(L=length,cell_dual_interval=list(map(str,[low,high])),frozen_recall_interval=list(map(str,[qrl,qru])),
            unpaired_interval=list(map(str,unpaired)),actual_C0_energy_contrast_interval=list(map(str,final)),approximate=list(map(float,final)),proofs=proofs))
        print('ACTUAL C0 ENERGY',length,*map(float,final),flush=True)
    save(args.out/'RESULT.json',dict(rows=answer,source_hashes=hashes,design_cells=design['good_box_cells'],constraints=len(rows),
        exact_simplex_dual_residual_included=True,actual_joint_distribution_contained_not_atomic_model=True,
        D_V_independence_assumed=False,new_samples=0,new_matrices=0,independent_proof_review=False,
        no_certified_LP_optimality_claim=True,wall_seconds=time.monotonic()-start))


if __name__=='__main__':main()
