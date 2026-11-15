"""Actual-C0 shared-threshold energy flips: two directions and total mass.

Uses the frozen R45 containing polytope. Floating LP only proposes duals.
"""
from fractions import Fraction as F
from pathlib import Path
from datetime import datetime,timezone
from itertools import combinations
import argparse
import time

import certify_two_sided_density_energy as energy

base=energy.base

def envelopes(lower,upper):
    assert -1<=lower<=upper<=1
    return dict(positive=(max(F(0),lower),max(F(0),upper)),
                negative=(max(F(0),-upper),max(F(0),-lower)),
                total=(F(0) if lower<=0<=upper else min(abs(lower),abs(upper)),max(abs(lower),abs(upper))))

def halfplanes(bounds,net):
    p,n,t=[list(map(F,bounds[k])) for k in ['positive','negative','total']];lo,hi=map(F,net)
    return [(F(1),F(0),p[1]),(F(-1),F(0),-p[0]),(F(0),F(1),n[1]),(F(0),F(-1),-n[0]),
            (F(1),F(1),t[1]),(F(-1),F(-1),-t[0]),(F(1),F(-1),hi),(F(-1),F(1),-lo)]

def polygon(bounds,net):
    constraints=halfplanes(bounds,net);vertices=set()
    for (a,b,c),(d,e,f) in combinations(constraints,2):
        det=a*e-b*d
        if det:
            p,n=(c*e-b*f)/det,(a*f-c*d)/det
            if all(x*p+y*n<=z for x,y,z in constraints):vertices.add((p,n))
    assert vertices,'Empty joint outer polygon'
    points=sorted(vertices)
    projected={name:[str(min(fn(p,n) for p,n in points)),str(max(fn(p,n) for p,n in points))]
               for name,fn in [('positive',lambda p,n:p),('negative',lambda p,n:n),('total',lambda p,n:p+n),('net',lambda p,n:p-n)]}
    # min(p,n) is affine on each side of the diagonal; extrema occur at
    # polygon vertices or at intersections of an edge with p=n.
    candidates=set(vertices)
    for a,b,c in constraints:
        if a+b:
            x=c/(a+b)
            if all(u*x+v*x<=w for u,v,w in constraints):candidates.add((x,x))
    cancelled=[2*min(p,n) for p,n in candidates]
    assert 0<=min(cancelled)<=max(cancelled)<=1
    return dict(halfplanes=[[str(x) for x in row] for row in constraints],vertices=[[str(x) for x in point] for point in points],
                projected_intervals=projected,cancelled_total_flip_mass_interval=list(map(str,[min(cancelled),max(cancelled)])),
                outer_enclosure_not_attainable_true_model_region=True)

def input_hashes(root):
    prior=base.read(root/'revisions/r45_two_sided_energy/RESULT.json')['source_hashes']
    names=list(prior)+['tools/certify_energy_flip_components.py','tools/certify_two_sided_density_energy.py',
        'tools/certify_sphere_density_lower.py','tools/certify_shared_cdf_energy.py','tools/certify_sphere_order_density.py',
        'tools/certify_sphere_threshold_cdf.py','tools/certify_energy_outer_polytope.py',
        'revisions/r44_sphere_density.json','revisions/r45_sphere_density_lower_v2.json',
        'revisions/r45_two_sided_energy/RESULT.json','revisions/r45_two_sided_energy/TWO_SIDED.json']
    hashes={n:base.sha(root/n) for n in names}
    for n,h in prior.items():assert hashes[n]==h
    receipt=base.read(root/'revisions/R45_HISTORY.json');manifest=base.read(root/receipt['path']/'manifest.json')
    for n,h in hashes.items():
        if n!='tools/certify_energy_flip_components.py':assert manifest['files'][n]==h,n
    return hashes

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic()
    hashes=input_hashes(root);design,rows,kernels=energy.build(root)
    old_design=base.read(root/'revisions/r45_two_sided_energy/DESIGN.json')
    for k,v in design.items():assert old_design[k]==v,k
    base.save(a.out/'DESIGN.json',dict(**design,created_utc=datetime.now(timezone.utc).isoformat(),source_hashes=hashes))
    import numpy as np
    from scipy.optimize import linprog
    matrix=np.array([[float(v) for v in r['coefficients']] for r in rows]);rhs=np.array([float(r['rhs']) for r in rows]);answers=[]
    prior=base.read(root/'revisions/r45_two_sided_energy/TWO_SIDED.json')
    for length,values in kernels.items():
        cells=[envelopes(v[0],v[1]) for v in values];components={}
        for name in ['positive','negative','total']:
            proofs={}
            for side,index,sign in [('lower',0,1),('upper',1,-1)]:
                cost=[sign*v[name][index] for v in cells]
                fit=linprog(np.array(list(map(float,cost))),A_ub=matrix,b_ub=rhs,A_eq=np.ones((1,len(cost))),b_eq=[1.],bounds=(0,None),method='highs')
                assert fit.success,fit.message
                weights=[(j,str(F.from_float(float(-v)))) for j,v in enumerate(fit.ineqlin.marginals) if v<0]
                bound,parts=base.simplex_dual(cost,rows,weights)
                proofs[side]=dict(multipliers=weights,parts=parts,certified_bound=str(sign*bound),float_optimum_diagnostic=float(sign*fit.fun))
            raw=[F(proofs['lower']['certified_bound']),F(proofs['upper']['certified_bound'])]
            interval=[max(F(0),raw[0]),min(F(1),raw[1])];assert interval[0]<=interval[1]
            components[name]=dict(raw_dual_interval=list(map(str,raw)),interval=list(map(str,interval)),approximate=list(map(float,interval)),proofs=proofs)
            print('ENERGY FLIPS',length,name,*map(float,interval),flush=True)
        net=next(r for r in prior['rows'] if r['L']==length)['actual_C0_energy_contrast_interval']
        joint=polygon({k:v['interval'] for k,v in components.items()},net)
        answers.append(dict(L=length,components=components,prior_net_interval=net,joint_outer_polygon=joint))
    base.save(a.out/'RESULT.json',dict(rows=answers,source_hashes=hashes,cells=design['good_box_cells']+1,constraints=len(rows),
        exact_simplex_dual_certificates=12,same_threshold_coupling=True,actual_C0_conditional_model=True,
        no_new_matrices_or_samples_or_probability_integrals=True,not_causal_attribution=True,
        prior_density_CDF_spectral_moment_C0_certificates_inherited=True,independent_proof_review=False,
        no_certified_LP_optimality_claim=True,wall_seconds=time.monotonic()-start))

if __name__=='__main__':main()
