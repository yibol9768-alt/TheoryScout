"""Stdlib replay of the exact cell-mass zero witness and non-lift proof."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import argparse

import certify_energy_zero_witness as cert

def references():
    # Feasible nonnegative cost with exact zero primal/dual; a negative
    # probability or a mass on a positive-cost cell must be rejected.
    costs=[F(0),F(0),F(1),F(0)];p=[F(1,4),F(3,4),F(0),F(0)]
    assert sum(p)==1 and sum(a*b for a,b in zip(costs,p))==0 and min(costs)==0
    rejected=0
    for bad in [[F(-1,4),F(5,4),F(0),F(0)],[F(1,4),F(1,2),F(1,4),F(0)]]:
        assert min(bad)<0 or sum(a*b for a,b in zip(costs,bad))>0;rejected+=1
    variance=0;covariance=0
    for a,b in [(F(0),F(1)),(F(211,200),F(53,50)),(F(-2),F(3))]:
        for weights in [[1,1,1],[1,2,3],[0,1,1],[1,0,0]]:
            w=[F(x,sum(weights)) for x in weights];v=[a,(a+b)/2,b];mean=sum(x*p for x,p in zip(v,w))
            var=sum(p*(x-mean)**2 for x,p in zip(v,w))
            assert var<=(mean-a)*(b-mean)<=(b-a)**2/4;variance+=1
            for d in [[F(-1),F(0),F(1)],[F(2),F(2),F(2)],[F(3),F(-2),F(1,2)]]:
                md=sum(x*p for x,p in zip(d,w));vd=sum(p*(x-md)**2 for x,p in zip(d,w))
                cov=sum(p*(x-md)*(y-mean) for x,y,p in zip(d,v,w))
                assert cov**2<=vd*var<=vd*(b-a)**2/4;covariance+=1
    # A moment upper alone allows concentrated energy while a lower
    # variance can rule it out; the toy constraints are not the C0 law.
    assert F(0)<=F(1,10) and not F(0)>=F(1,20)
    return dict(zero_primal_and_zero_multiplier_dual_cases=1,invalid_mass_or_positive_cost_cases_rejected=rejected,
                interval_variance_reference_cases=variance,Cauchy_covariance_reference_cases=covariance,
                upper_moment_not_a_lower_moment_reference=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('certificate',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];saved=cert.base.read(a.certificate)
    for n,h in saved['source_hashes'].items():assert cert.base.sha(root/n)==h,n
    old_hashes=cert.sources(root)
    for n,h in old_hashes.items():assert saved['source_hashes'][n]==h
    actual=cert.audit(root,saved['mass_by_cell'])
    for k,v in actual.items():assert saved[k]==v,k
    raw=cert.base.read(root/'revisions/r51_zero_feasibility_diagnostic.json')
    assert next(r for r in raw['trials'] if r['label']=='frozen_no_tail')['atoms']==saved['mass_by_cell']
    cert.base.save(a.out,dict(source_hashes_verified=len(saved['source_hashes']),reference=references(),
        exact_mass_normalization_and_all82_inequalities_reconstructed=True,exact_six_zero_relaxation_minima_reconstructed=True,
        variance_and_covariance_nonlift_obstructions_reconstructed=True,strictly_positive_mass_cells=saved['positive_cell_masses'],
        energy_strip=saved['energy_strip'],minimum_raw_row_slack=saved['minimum_raw_row_slack'],
        any_lift_variance_V_upper=saved['any_lift_variance_V_upper'],true_C0_variance_V_lower=saved['true_C0_variance_V_lower'],
        full_moment_nonidentifiability_or_actual_zero_flips_claimed=False,optimizer_required=False,
        new_probability_integrals_recomputed=0,prior_density_CDF_spectral_moment_C0_certificates_inherited=True,
        independent_proof_review=False))
    print('PASS33 cell masses,82 exact slacks,six zero LP minima and variance/Cauchy non-lift certificates')

if __name__=='__main__':main()
