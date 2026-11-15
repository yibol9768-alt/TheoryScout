"""Combine extended whole-span density minima and common-segment integrals.

The initial integrated-only R54 implementation/results stay frozen. Class
substitution is local to this sequential process and is restored in finally.
"""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_left,bisect_right
import argparse
import certify_extended_energy as initial

base=initial.base


class Full(initial.Extended):
    def __init__(self,upper,old_lower,new_lower):
        super().__init__(upper,old_lower,new_lower)
        first=new_lower['rows'][0]
        self.full_left=[F(b['left'])**2 for b in first['bins']]
        self.full_right=[F(b['right'])**2 for b in first['bins']]
        self.full_rows={r['L']:[F(b['density_lower']) for b in r['bins']] for r in new_lower['rows']}
        self.full_end=F(new_lower['grid_end'])**2

    def signed_minorant(self,length,box,meanv):
        low,high=super().signed_minorant(length,box,meanv)
        dl,dh,vl,vh=box;ml,mh=meanv
        xmin2=dl*dl/max(vh,mh);xmax2=dh*dh/min(vl,ml)
        start=bisect_left(self.full_right,xmin2);end=bisect_right(self.full_left,xmax2)
        values=self.full_rows[length][start:end]
        if xmax2>=self.full_end:values=values+[F(0)]
        assert values
        density=min(values);al,ah=self.inverse_interval(vl,vh,ml,mh)
        if al>=0:low=max(low,density*dl*al)
        if ah<=0:high=min(high,density*dl*ah)
        assert low<=high
        return low,high


def model(root):
    old_class=initial.Extended
    try:
        initial.Extended=Full
        return initial.model(root)
    finally:initial.Extended=old_class


def propose(root,out):
    old_class=initial.Extended
    try:
        initial.Extended=Full
        initial.propose(root,out)
    finally:initial.Extended=old_class


def certify(root,out):
    hashes=initial.sources(root)
    for name in ['tools/certify_extended_energy_full.py','revisions/r54_extended_energy_full_proposal.json']:
        hashes[name]=base.sha(root/name)
    d,rows,kernels,counts=model(root);p=base.read(root/'revisions/r54_extended_energy_full_proposal.json')
    if p['mode']=='zero_witness':
        check=initial.prior.previous.audit(d,rows,kernels,p['mass'],84)
        assert check['minimum_raw_slack']==p['exact_minimum_slack']
        result=dict(mode=p['mode'],zero_witness=check,six_lower_LP_optima_exactly_zero=True)
    else:
        checked=[];costs=initial.objectives(kernels)
        for r in p['rows']:
            cost=next(c for length,name,c in costs if length==r['L'] and name==r['component'])
            alpha,parts=base.simplex_dual([F(int(c>0)) for c in cost],rows,r['multipliers'])
            minimum=min(c for c in cost if c>0);bound=max(F(0),alpha)*minimum
            assert str(alpha)==r['indicator_mass_lower'] and str(minimum)==r['minimum_positive_cost'] and parts==r['dual_parts'] and str(bound)==r['certified_flip_lower']
            checked.append(dict(L=r['L'],component=r['component'],indicator_mass_lower=str(alpha),minimum_positive_cost=str(minimum),certified_flip_lower=str(bound)))
        assert len(checked)==6
        result=dict(mode=p['mode'],rows=checked,positive_lower_bounds=sum(F(r['certified_flip_lower'])>0 for r in checked),LP_optimality_not_claimed=True)
    result.update(source_hashes=hashes,counts=counts,cells=d['good_box_cells']+1,inequalities=84,
        extended_whole_span_and_integrated_common_segment_bounds_both_retained=True,
        all_signed_intervals_nested_against_R53=True,original_recall_coefficients_unchanged=True,
        inherited160bit_globals_unchanged=True,actual_zero_flips_or_full_moment_nonidentifiability_not_claimed=True,
        prior_C0_spectral_CDF_coupling_moment_certificates_inherited=True,
        new_scalar_density_probability_certificates_used=True,new_high_dimensional_probability_integrals=0,
        new_matrices=0,new_random_draws=0,independent_proof_review=False)
    base.save(out,result);print('EXACT FULL EXTENDED ENERGY',result['mode'],counts,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--propose',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.propose:propose(root,a.out)
    else:certify(root,a.out)

if __name__=='__main__':main()
