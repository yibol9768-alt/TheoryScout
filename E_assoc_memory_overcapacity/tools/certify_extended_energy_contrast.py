"""Signed actual-C0 energy contrast bounds with the extended density model.

Four floating LPs propose multipliers. Exact simplex residuals plus retained
R45 bounds certify the inclusions, with no LP-optimality claim.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse
import certify_extended_energy_full as full

base=full.base


def propose(root,out):
    import numpy as np
    from scipy.optimize import linprog
    d,rows,kernels,counts=full.model(root)
    scales=[max(F(1),abs(r['rhs']),max(abs(x) for x in r['coefficients'])) for r in rows]
    matrix=np.array([[float(x/s) for x in r['coefficients']] for r,s in zip(rows,scales)])
    rhs=np.array([float(r['rhs']/s) for r,s in zip(rows,scales)]);answers=[]
    for length,values in kernels.items():
        for side,index,sign in [('lower',0,1),('upper',1,-1)]:
            cost=[sign*v[index] for v in values]
            fit=linprog(np.array(list(map(float,cost))),A_ub=matrix,b_ub=rhs,A_eq=np.ones((1,len(cost))),b_eq=[1.],bounds=(0,None),method='highs')
            assert fit.success,fit.message
            multipliers=[(j,str(F.from_float(float(-v))/scales[j])) for j,v in enumerate(fit.ineqlin.marginals) if v<0]
            bound,parts=base.simplex_dual(cost,rows,multipliers)
            answers.append(dict(L=length,side=side,multipliers=multipliers,parts=parts,certified_bound=str(sign*bound),float_optimum_diagnostic=float(sign*fit.fun)))
            print('SIGNED ENERGY',length,side,float(sign*bound),flush=True)
    base.save(out,dict(rows=answers))


def certify(root,out):
    d,rows,kernels,counts=full.model(root);p=base.read(root/'revisions/r54_extended_contrast_proposal.json')
    old=base.read(root/'revisions/r45_two_sided_energy/TWO_SIDED.json');answers=[]
    assert len(p['rows'])==4
    for length,values in kernels.items():
        bounds=[]
        for side,index,sign in [('lower',0,1),('upper',1,-1)]:
            r=next(r for r in p['rows'] if r['L']==length and r['side']==side)
            bound,parts=base.simplex_dual([sign*v[index] for v in values],rows,r['multipliers'])
            assert str(sign*bound)==r['certified_bound'] and parts==r['parts']
            bounds.append(sign*bound)
        prior=list(map(F,next(r for r in old['rows'] if r['L']==length)['actual_C0_energy_contrast_interval']))
        final=[max(prior[0],bounds[0]),min(prior[1],bounds[1])]
        assert prior[0]<=final[0]<=final[1]<=prior[1]
        answers.append(dict(L=length,previous_R45_interval=list(map(str,prior)),fresh_dual_interval=list(map(str,bounds)),retained_actual_C0_energy_contrast_interval=list(map(str,final)),approximate=list(map(float,final))))
    certificate=base.read(root/'revisions/r54_extended_energy_full_certificate.json')
    for n,h in certificate['source_hashes'].items():assert base.sha(root/n)==h,n
    names=list(certificate['source_hashes'])+['tools/certify_extended_energy_contrast.py','revisions/r54_extended_energy_full_certificate.json','revisions/r54_extended_contrast_proposal.json']
    base.save(out,dict(source_hashes={n:base.sha(root/n) for n in names},rows=answers,constraints=84,
        four_exact_simplex_residual_duals=True,old_bounds_retained=True,LP_optimality_not_claimed=True,
        conditional_actual_C0_model=True,original_recall_probability_bounds_unchanged=True,
        not_causal_attribution_or_statistical_confidence_interval=True,new_matrices=0,new_samples=0,
        prior_density_CDF_spectral_moment_C0_inputs_inherited_except_new_scalar_density_enclosures=True,
        independent_proof_review=False))
    print('EXACT SIGNED ENERGY',[(r['L'],r['approximate']) for r in answers],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--propose',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.propose:propose(root,a.out)
    else:certify(root,a.out)

if __name__=='__main__':main()
