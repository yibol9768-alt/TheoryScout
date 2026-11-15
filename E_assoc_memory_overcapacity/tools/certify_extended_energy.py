"""Energy LP with extended512-bit density minorants and frozen R52 rows.

Floating optimization only proposes a zero witness or nonnegative duals for
indicator mass objectives. Exact replay certifies every stated bound.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse
import certify_integrated_energy_minorant as prior

base=prior.base


class Extended(prior.Integrated):
    def __init__(self,upper,old_lower,new_lower):
        super().__init__(upper,old_lower)
        self.pieces={r['L']:[(F(b['left']),F(b['right']),F(b['density_lower'])) for b in r['bins']] for r in new_lower['rows']}
        self.integral_prefix={}


def model(root):
    d,rows,old_r45,old_kernels,old_counts=prior.model(root)
    new_lower=base.read(root/'revisions/r54_extended_density.json')
    for n,h in new_lower['source_hashes'].items():assert base.sha(root/n)==h,n
    old_lower=base.read(root/'revisions/r45_sphere_density_lower_v2.json');upper=base.read(root/'revisions/r44_sphere_density.json')
    obj=Extended(upper,old_lower,new_lower)
    ds=list(map(F,d['D_grid']));vs=list(map(F,d['V_grid']));meanv=tuple(map(F,d['mean_V_interval']))
    boxes=[(a,b,c,e) for a,b in zip(ds,ds[1:]) for c,e in zip(vs,vs[1:])];negative=F(d['negative_threshold_upper'])
    new={};counts=[]
    for r in base.read(root/'revisions/r23_cdf_dense8/RESULT.json')['rows']:
        length=r['L'];vals=[obj.kernel(r,box,meanv,negative) for box in boxes]+[old_kernels[length][-1]]
        assert all(a[2:]==b[2:] and a[0]<=b[0]<=b[1]<=a[1] for a,b in zip(old_kernels[length],vals))
        new[length]=vals
        counts.append(dict(L=length,lower_strictly_tightened=sum(b[0]>a[0] for a,b in zip(old_kernels[length],vals)),upper_strictly_tightened=sum(b[1]<a[1] for a,b in zip(old_kernels[length],vals))))
    assert base.integer.BITS==160 and base.S==1<<160
    return d,rows,new,counts


def objectives(kernels):
    result=[]
    for length,values in kernels.items():
        env=[prior.previous.zero.flips.envelopes(v[0],v[1]) for v in values]
        for component in ['positive','negative','total']:
            cost=[e[component][0] for e in env];assert all(c>=0 for c in cost)
            result.append((length,component,cost))
    return result


def propose(root,out):
    import numpy as np
    from scipy.optimize import linprog
    d,rows,kernels,counts=model(root);costs=objectives(kernels);size=d['good_box_cells']+1
    allowed=[i for i in range(size) if all(c[i]==0 for _,_,c in costs)]
    scales=[max(F(1),abs(r['rhs']),max(abs(x) for x in r['coefficients'])) for r in rows]
    matrix=np.array([[float(r['coefficients'][i]/s) for i in allowed]+[1.] for r,s in zip(rows,scales)])
    rhs=np.array([float(r['rhs']/s) for r,s in zip(rows,scales)])
    fit=linprog(np.array([0.]*len(allowed)+[-1.]),A_ub=matrix,b_ub=rhs,A_eq=np.array([[1.]*len(allowed)+[0.]]),b_eq=[1.],bounds=[(0,None)]*len(allowed)+[(None,None)],method='highs')
    assert fit.success,fit.message
    diagnosis=dict(phase_I_scaled_margin=float(fit.x[-1]),common_zero_cells=len(allowed),not_an_infeasibility_certificate=True)
    if fit.x[-1]>0:
        mass=[[i,F.from_float(float(p))] for i,p in zip(allowed,fit.x[:-1]) if p>0]
        j=max(range(len(mass)),key=lambda j:mass[j][1]);mass[j][1]+=1-sum(p for i,p in mass)
        mass=[[i,str(p)] for i,p in mass]
        try:check=prior.previous.audit(d,rows,kernels,mass,84)
        except AssertionError:check=None
        if check:
            base.save(out,dict(mode='zero_witness',diagnostic=diagnosis,mass=mass,exact_minimum_slack=check['minimum_raw_slack']))
            print('EXACT ZERO CANDIDATE',check['positive_cells'],'cells',flush=True);return
    # Avoid tiny-objective solver tolerances: certify a mass lower bound on
    # cells with strictly positive cost, then multiply by their minimum cost.
    full=np.array([[float(x/s) for x in r['coefficients']] for r,s in zip(rows,scales)])
    answers=[]
    for length,component,cost in costs:
        indicator=[int(c>0) for c in cost]
        fit=linprog(np.array(indicator,dtype=float),A_ub=full,b_ub=rhs,A_eq=np.ones((1,size)),b_eq=[1.],bounds=(0,None),method='highs')
        assert fit.success,fit.message
        y=[(j,str(F.from_float(float(-v))/scales[j])) for j,v in enumerate(fit.ineqlin.marginals) if v<0]
        alpha,parts=base.simplex_dual(list(map(F,indicator)),rows,y)
        minimum=min(c for c in cost if c>0)
        bound=max(F(0),alpha)*minimum
        answers.append(dict(L=length,component=component,multipliers=y,indicator_mass_lower=str(alpha),minimum_positive_cost=str(minimum),certified_flip_lower=str(bound),dual_parts=parts,float_indicator_optimum_diagnostic=float(fit.fun)))
        print('INDICATOR',length,component,'mass >=',float(alpha),'flip >=',float(bound),flush=True)
    base.save(out,dict(mode='indicator_duals',diagnostic=diagnosis,rows=answers))


def sources(root):
    old=base.read(root/'revisions/r53_integrated_energy_certificate.json')
    receipt=base.read(root/'revisions/R53_HISTORY.json');manifest=base.read(root/receipt['path']/'manifest.json')
    assert base.sha(root/'revisions/r53_integrated_energy_certificate.json')==manifest['files']['revisions/r53_integrated_energy_certificate.json']
    for n,h in old['source_hashes'].items():assert base.sha(root/n)==h,n
    names=list(old['source_hashes'])+list(base.read(root/'revisions/r54_extended_density.json')['source_hashes'])+[
        'tools/certify_extended_energy.py','revisions/r53_integrated_energy_certificate.json','revisions/r54_extended_density.json','revisions/r54_extended_energy_proposal.json']
    return {n:base.sha(root/n) for n in names}


def certify(root,out):
    hashes=sources(root);d,rows,kernels,counts=model(root);p=base.read(root/'revisions/r54_extended_energy_proposal.json')
    if p['mode']=='zero_witness':
        check=prior.previous.audit(d,rows,kernels,p['mass'],84);assert check['minimum_raw_slack']==p['exact_minimum_slack']
        result=dict(mode=p['mode'],zero_witness=check,six_lower_LP_optima_exactly_zero=True)
    else:
        checked=[];costs=objectives(kernels)
        for r in p['rows']:
            cost=next(c for length,name,c in costs if length==r['L'] and name==r['component'])
            indicator=[F(int(c>0)) for c in cost];alpha,parts=base.simplex_dual(indicator,rows,r['multipliers']);minimum=min(c for c in cost if c>0)
            bound=max(F(0),alpha)*minimum
            assert str(alpha)==r['indicator_mass_lower'] and str(minimum)==r['minimum_positive_cost'] and parts==r['dual_parts'] and str(bound)==r['certified_flip_lower']
            checked.append(dict(L=r['L'],component=r['component'],indicator_mass_lower=str(alpha),minimum_positive_cost=str(minimum),certified_flip_lower=str(bound)))
        assert len(checked)==6
        result=dict(mode=p['mode'],rows=checked,positive_lower_bounds=sum(F(r['certified_flip_lower'])>0 for r in checked),LP_optimality_not_claimed=True)
    result.update(source_hashes=hashes,cells=d['good_box_cells']+1,inequalities=84,counts=counts,
        signed_intervals_nested=True,original_recall_coefficients_unchanged=True,high_precision_module_did_not_change_inherited160bit_globals=True,
        no_new_matrices_or_random_samples=True,new_scalar_probability_density_certificates_used=True,new_high_dimensional_probability_integrals=0,
        full_moment_nonidentifiability_or_actual_zero_flips_not_claimed=True,not_statistical_CI_or_causal_attribution=True,
        prior_C0_spectral_moment_CDF_coupling_certificates_inherited=True,independent_proof_review=False)
    base.save(out,result);print('EXACT EXTENDED ENERGY',result['mode'],counts,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--propose',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.propose:propose(root,a.out)
    else:certify(root,a.out)

if __name__=='__main__':main()
