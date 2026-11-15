"""Stdlib audit of paired arithmetic and the bounded-iid confidence radius.

The coverage theorem and conditional iid assumptions remain explicit inputs;
finite references do not independently prove that statistical theorem.
"""
from fractions import Fraction as F
from pathlib import Path
from itertools import product
import argparse

import certify_z4b_bounded_proxy as cert

def references():
    cases=0
    # Perfect positive and negative correlation and a constant observed
    # sample distinguish paired variance from diagonal-only variance.
    for mode in ['equal','opposite','constant']:
        counts=[]
        for j in range(1280):
            a=512 if mode=='constant' else 1024*(j%2)
            b=1024-a if mode=='opposite' else a
            counts.append([[a,0],[b,0],[0,0],[0,0]])
        vectors,means,cov=cert.moments(counts)
        c=[F(1),F(1) if mode=='opposite' else F(-1)]+[F(0)]*6
        result=cert.record(dict(label=mode,kind='reference',coefficients=c,constant=F(1,7)),vectors,means,cov)
        assert F(result['sample_variance'])==0
        if mode!='constant':assert F(result['paired_covariance_decomposition']['diagonal_only_variance'])>0
        assert int(result['radius']['upper_integer'])>0 # observed zero variance does not prove a zero population variance
        cases+=1
    support_cases=0
    for ka,kb in [(F(-2),F(-3)),(F(1,3),F(-2,5)),(F(0),F(1))]:
        c=[F(1),F(-1),F(0),F(0),-ka,kb,F(0),F(0)];constant=F(17,40)*(ka-kb)
        values=[constant+sum(a*x for a,x in zip(c,v)) for v in product([0,1],repeat=8)]
        assert min(values)==constant+sum(min(F(0),a) for a in c)
        assert max(values)==constant+sum(max(F(0),a) for a in c)
        assert max(values)-min(values)==2+abs(ka)+abs(kb)
        support_cases+=len(values)
    pairwise=0
    for values in [[F(0),F(1)],[F(0),F(1,4),F(3,4),F(1)],[F(2,3)]*4]:
        n=len(values);mean=sum(values)/n
        direct=sum((x-mean)**2 for x in values)/(n-1)
        pair=sum((values[j]-values[k])**2 for j in range(n) for k in range(j+1,n))/(n*(n-1))
        assert direct==pair;pairwise+=1
    assert 20*F(1,400)==F(1,20)
    return dict(correlated_and_constant_sample_cases=cases,exact_signed_support_corner_cases=support_cases,
                sample_variance_normalization_cases=pairwise,twenty_tail_union_budget_verified=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('certificate',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];saved=cert.read(a.certificate);data,hashes=cert.source_hashes(root)
    assert saved['source_hashes']==hashes
    actual=cert.compute(data);actual['source_hashes']=hashes;assert actual==saved
    assert cert.read(root/'z4b/formal/analysis/report.json')['decisions']==data['decisions']
    # Independent rational square test of every outward radical bound.
    for row in saved['records']:
        variance=F(row['sample_variance']);span=F(row['span']);u=F(int(row['radius']['upper_integer']),cert.S)
        linear=F(49,3*(saved['n']-1))*span
        assert u>=linear and (u-linear)**2>=F(14,saved['n'])*variance
        lower,upper=map(F,row['confidence_interval']);sl,su=map(F,row['support'])
        assert sl<=lower<=upper<=su
    cert.save(a.out,dict(references=references(),exact_eight_by_eight_covariance_reconstructed=True,
        exact_paired_variable_means_and_variances_reconstructed=10,confidence_radical_upper_bounds_independently_squared=10,
        finite_family_size=10,per_tail_failure='1/400',joint_failure_upper='1/20',exp7_exceeds800_exactly_verified=True,
        source_hashes_verified=len(hashes),registered_decisions_unchanged=True,
        proxy_lower_bounds_strictly_positive=[r['label'] for r in saved['records'] if r['kind']=='linear_proxy' and r['finite_sample_proxy_lower_strictly_positive']],
        one_pp_matching_gates_certified=[r['label'] for r in saved['records'] if r['kind']=='matching' and r['finite_sample_bound_certifies_one_pp_gate']],
        statistical_theorem='Maurer and Pontil 2009 Theorem4; iid bounded datasets conditional on independent locks/fixed coefficients',
        primary_source_pdf_sha256=hashes['revisions/r47_primary_source/maurer_pontil_2009_v1.pdf'],
        no_Student_t_approximation_used=True,actual_exact_matching_certified=False,independent_statistical_review=False))
    print('PASS exact paired covariance, ten bounded-iid intervals, family budget and references')

if __name__=='__main__':main()
