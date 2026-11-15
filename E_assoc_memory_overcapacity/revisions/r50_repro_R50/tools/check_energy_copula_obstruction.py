"""Exact dependence counterexample, generic laws rather than the C0 sphere law."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json


def construction(sign):
    assert sign in [-1,1]
    d=[F(63,20),F(77,20),F(91,20)]
    v=[F(16,25),F(1),F(36,25)]
    z=[[0,1,-1],[1,-1,0],[-1,0,1]]
    u=[1,-2,1];b=[11,-20,9]
    matrix=[[F(1,9)+F(z[i][j],15)+F(sign*u[i]*b[j],1800) for j in range(3)] for i in range(3)]
    assert all(p>0 for row in matrix for p in row)
    assert all(sum(row)==F(1,3) for row in matrix)
    assert all(sum(matrix[i][j] for i in range(3))==F(1,3) for j in range(3))
    assert sum(u)==sum(b)==0 and sum(x*y for x,y in zip(d,u))==sum(x*y for x,y in zip(v,b))==0
    law=[(d[i],v[j],F(9,10)*matrix[i][j]) for i in range(3) for j in range(3)]
    law.append((F(1407,400),F(121,100),F(1,10)))
    return matrix,law


def marginal(law,column):
    return {x:sum(p for d,v,p in law if (d,v)[column]==x) for x in sorted({(d,v)[column] for d,v,p in law})}


def conditional_means(law,column):
    return {x:sum(p*(v if column==0 else d) for d,v,p in law if (d,v)[column]==x)/mass
            for x,mass in marginal(law,column).items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    laws={};rows=[];classifications=0
    tlo=F(3499,1000);thi=F(3501,1000)
    for sign in [-1,1]:
        matrix,law=construction(sign);laws[sign]=law
        mean_d=sum(d*p for d,v,p in law);mean_v=sum(v*p for d,v,p in law)
        covariance=sum(d*v*p for d,v,p in law)-mean_d*mean_v
        assert mean_v==F(209,200) and covariance>0
        root={F(16,25):F(4,5),F(1):F(1),F(36,25):F(6,5),F(121,100):F(11,10)}
        mean_root=sum(root[v]*p for d,v,p in law);assert mean_root==F(101,100)
        assert mean_v>mean_root**2
        def probability(mode):
            nonlocal classifications
            total=F(0)
            for d,v,p in law:
                energy={'actual':v,'rms':mean_v,'unit':F(1)}[mode]
                # Strict endpoint comparisons certify every threshold in the
                # continuous interval, with no numerical square-root evaluation.
                success=d*d>thi*thi*energy
                failure=d*d<tlo*tlo*energy
                assert success!=failure
                total+=p*success;classifications+=1
            return total
        q=probability('actual');qr=probability('rms');q1=probability('unit')
        assert qr==F(3,5) and q1==F(7,10)
        assert q-qr==sign*F(29,2000) and q1-qr==F(1,10)
        rows.append(dict(sign=sign,conditional_base_matrix=[[str(p) for p in row] for row in matrix],
            mean_D=str(mean_d),mean_V=str(mean_v),mean_root_V=str(mean_root),covariance_D_V=str(covariance),
            actual_Q=str(q),fixed_RMS_Qr=str(qr),unit_energy_Q1=str(q1),actual_minus_RMS=str(q-qr),
            unit_minus_RMS=str(q1-qr),mean_margin=str(mean_d-(tlo+thi)/2*mean_root)))
    for col in [0,1]:
        assert marginal(laws[-1],col)==marginal(laws[1],col)
        assert conditional_means(laws[-1],col)==conditional_means(laws[1],col)
    for row in rows:
        assert {k:v for k,v in row.items() if k in ['mean_D','mean_V','mean_root_V','covariance_D_V','mean_margin']}=={k:v for k,v in rows[0].items() if k in ['mean_D','mean_V','mean_root_V','covariance_D_V','mean_margin']}
    output=dict(scope='Generic positive D,V laws; T independent uniform on the stated interval, not the spherical order-threshold or represented C0 law',
        threshold_interval=[str(tlo),str(thi)],rows=rows,strict_interval_event_classifications=classifications,
        marginal_laws_identical=True,both_conditional_mean_regressions_identical=True,
        positive_covariance_identical=True,all_univariate_moments_identical=True,
        Jensen_gap_identical=True,mean_margin_identical=True,
        positive_fixed_energy_contrast_identical=True,signed_random_energy_remainder_has_opposite_signs=True,
        actual_C0_recall_bounds_changed=False,independent_proof_review=False)
    with args.out.open('x') as stream:json.dump(output,stream,indent=2);stream.write('\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
