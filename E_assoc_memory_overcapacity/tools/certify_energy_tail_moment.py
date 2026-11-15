"""Tail-controlled lower second moment inside the frozen good window.

Uses a spherical fourth moment of W, the raw fourth moment of D, and the
inherited bad-event budget. No new matrix construction or probability integral.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import certify_energy_variance_cut as global_cut

zero=global_cut.zero
base=zero.base
I=base.integer.I


def compute(root):
    zero.sources(root)
    inp=base.read(root/'revisions/r28_conditional_moments_v2/INPUTS.json')
    moment=base.read(root/'revisions/r28_conditional_moments_v2/RESULT.json')['C0_moments']
    eig_name='revisions/r27_bad_matrices/W/actual_spectrum.json'
    err_name='revisions/r27_bad_matrices/W/ERRORS.json'
    for name in [eig_name,err_name]:assert base.sha(root/name)==inp['source_hashes'][name],name
    eig=base.read(root/eig_name);err=base.read(root/err_name)
    n=inp['n'];q=inp['q'];m=n-1;assert len(eig['eigenvalues'])==q<n
    design=base.read(root/'revisions/r45_two_sided_energy/DESIGN.json')
    delta=F(design['tail_mass_upper']);center=F(1057,1000)
    mw=base.restore(moment['mean_W']);radius=F(err['total'])+F(n,m)*F(inp['N_operator_perturbation'])
    # W=n u^T A u, A=n N/(n-1); the archived W spectrum is for Ahat.
    centered=[I(F(v))+I.raw(-I(radius).hi,I(radius).hi)-mw/n for v in eig['eigenvalues']]
    full_zero_eigenvalue=-mw/n
    u2=sum(v.square() for v in centered)+(n-q)*full_zero_eigenvalue.square()
    u4=sum(v.square().square() for v in centered)+(n-q)*full_zero_eigenvalue.square().square()
    fourth=I(F(n**3,(n+2)*(n+4)*(n+6)))*(12*u2.square()+48*u4)
    mu=base.restore(moment['mean_D']);vd=base.restore(moment['variance_D'])
    d4=(base.restore(moment['fourth_central_D'])+4*mu*base.restore(moment['third_central_D'])
        +6*mu.square()*vd+mu.square().square())
    assert fourth.lo>0 and d4.lo>0
    first=base.hi((I(base.hi(fourth))*I(delta)).sqrt().sqrt())
    shift=mw-center;shift_upper=max(abs(base.lo(shift)),abs(base.hi(shift)))
    second=shift_upper*base.hi(I(delta).sqrt())
    third=base.hi(d4.sqrt())/m
    tail_upper=(first+second+third)**2
    variance_lower=base.lo(base.restore(moment['variance_V']))
    good_lower=variance_lower-tail_upper;assert good_lower>0
    ds=list(map(F,design['D_grid']));vs=list(map(F,design['V_grid']))
    coefficients=[-max((a-center)**2,(b-center)**2) for _ in zip(ds,ds[1:]) for a,b in zip(vs,vs[1:])]+[F(0)]
    assert len(coefficients)==design['good_box_cells']+1
    return dict(center=str(center),n=n,q=q,ambient_zero_eigenvalues_included=n-q,
        A_eigenvalue_error_radius=str(radius),mean_W=mw.record(),centered_A_trace2=u2.record(),centered_A_trace4=u4.record(),
        W_central_fourth_moment=fourth.record(),D_raw_fourth_moment=d4.record(),tail_probability_upper=str(delta),
        restricted_L2_norm_upper_terms=list(map(str,[first,second,third])),tail_second_moment_upper=str(tail_upper),
        full_V_variance_lower=str(variance_lower),good_window_second_moment_lower=str(good_lower),
        label='tail_controlled_lower_V_second_moment',coefficients=list(map(str,coefficients)),rhs=str(-good_lower),
        outside_coefficient_zero_only_after_subtracting_proved_tail_moment_budget=True,
        W_and_D_dependence_allowed=True,new_probability_integrals=0,new_random_draws=0,
        prior_spectrum_coupling_moment_C0_inputs_inherited=True,independent_proof_review=False)


def references():
    # Exact finite sphere monomials independently enumerate small diagonal
    # quadratic fourth moments, including non-block ambient zero eigenvalues.
    from itertools import product
    from math import factorial,prod
    checked=0
    for values in [(F(1),F(2)),(F(0),F(1),F(3)),(F(0),F(0),F(2),F(4))]:
        n=len(values);mean=sum(values)/n;c=[v-mean for v in values]
        u2=sum(v*v for v in c);u4=sum(v**4 for v in c)
        formula=(12*u2*u2+48*u4)/prod(n+2*j for j in range(4))
        enum=F(0)
        for counts in product(range(5),repeat=n):
            if sum(counts)!=4:continue
            multi=F(factorial(4),prod(factorial(k) for k in counts))
            odd=prod(prod(range(1,2*k,2)) for k in counts)
            enum+=multi*prod(v**k for v,k in zip(c,counts))*F(odd,prod(n+2*j for j in range(4)))
        assert formula==enum;checked+=1
    tail_cases=0
    for vals,probs,bad in [([F(0),F(2)],[F(3,4),F(1,4)],[1]),([F(-1),F(0),F(4)],[F(1,3)]*3,[0,2]),([F(2)],[F(1)],[])]:
        p=sum(probs[j] for j in bad);m4=sum(w*x**4 for w,x in zip(probs,vals));restricted=sum(probs[j]*vals[j]**2 for j in bad)
        assert restricted**2<=m4*p;tail_cases+=1
    return dict(small_sphere_fourth_moments_enumerated=checked,restricted_second_moment_Holder_cases=tail_cases)


def sources(root):
    names=list(zero.sources(root))+['tools/certify_energy_tail_moment.py','tools/certify_energy_variance_cut.py',
        'tools/certify_energy_zero_witness.py','revisions/r51_energy_zero_witness.json',
        'revisions/r51_zero_feasibility_diagnostic.json','revisions/r51_variance_cut.json',
        'revisions/r27_bad_matrices/W/actual_spectrum.json','revisions/r27_bad_matrices/W/ERRORS.json']
    return {n:base.sha(root/n) for n in names}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--compare',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];v=compute(root);v['reference']=references();v['source_hashes']=sources(root)
    if a.compare:assert v==base.read(a.compare)
    base.save(a.out,v)
    print('EXACT TAIL MOMENT:bad <=',float(F(v['tail_second_moment_upper'])),'; good >=',float(F(v['good_window_second_moment_lower'])),flush=True)

if __name__=='__main__':main()
