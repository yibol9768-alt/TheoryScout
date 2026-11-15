"""Exact scope checks: summability is not an iff, and reference-one matters."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json

def compute():
    records=[];identities=0
    for s in [F(1),F(5,4),F(3,2)]:
        cap=1/max(F(1),s);q=F(1);ref=F(1);p=F(0);total=F(0)
        for j in range(1,65):
            eta=cap if j==1 else cap/F(j*(j+1))
            assert 0<eta<=cap
            q*=1-eta*s;ref*=1-eta;p=p*(1-eta*s)+eta;total+=eta
            assert q==0 and p==1/s and total==cap*(F(3,2)-F(1,j+1))
            assert total<F(3,2)*cap
            k=1-ref;assert k>0 and min(F(1),1/s)<=p/k<=max(F(1),1/s)
            identities+=1
        records.append(dict(eigenvalue=str(s),first_step=str(cap),finite_total_step_budget=str(F(3,2)*cap),
                            residual_zero_from_step1=True,least_squares_reached_without_divergent_sum=True))
    s=F(1,2);eta=F(3,2);p=2*eta-eta*eta*s;k=2*eta-eta*eta
    assert eta<=1/s and 1-eta*s>=0 and p/k==F(5,2)>1/s
    assert eta>1/max(F(1),s)
    return dict(finite_sum_exact_hit_examples=records,rational_recurrence_and_ratio_checks=identities,
                actual_spectrum_nonovershoot_alone_insufficient=True,reference_one_cap_counterexample=dict(s=str(s),eta=str(eta),T=2,
                    filter_ratio=str(p/k),claimed_upper_without_reference_one=str(1/s)),
                divergence_is_sufficient_not_necessary_for_each_individual_schedule=True,
                random_training_datasets_generated=0,independent_proof_review=False,
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--compare',type=Path);a=p.parse_args()
    result=compute()
    if a.compare:assert json.loads(a.compare.read_text())==result
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('PASS finite-sum exact-hit and reference-one step-cap scope checks')

if __name__=='__main__':main()
