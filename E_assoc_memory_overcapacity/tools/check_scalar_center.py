"""Exact jet/midpoint references and stored scalar-center arithmetic audit."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import json
from certify_adam2_scalar_center import I,S,J,read,sha,save,restore,h2,density
import certify_weighted_chisquare as integer


def references():
    jet_checks=0;polynomial_checks=0;scalar_checks=0
    for a in [F(1,16),F(1,4),F(1),F(4)]:
        s={F(1,16):F(1,4),F(1,4):F(1,2),F(1):F(1),F(4):F(2)}[a]
        for b in [F(-2),F(0),F(1,3)]:
            for c in [F(-1,3),F(0),F(2)]:
                x=J(a,b,c)
                for got,want in [(x.reciprocal(),[1/a,-b/a**2,b*b/a**3-c/a**2]),
                                 (x.sqrt(),[s,b/(2*s),c/(2*s)-b*b/(8*s**3)]),
                                 (x*x,[a*a,2*a*b,b*b+2*a*c])]:
                    for value,exact in zip(got.v,want):assert value.contains(exact);jet_checks+=1
    for cells in [1,2,4,8,16]:
        step=F(1,cells)
        for power in range(9):
            approx=F(0);error=F(0)
            for k in range(cells):
                left=k*step;right=(k+1)*step;mid=(left+right)/2
                approx+=step*mid**power
                M=power*(power-1)*right**(power-2) if power>=2 else 0
                error+=M*step**3/24
            assert approx-error<=F(1,power+1)<=approx+error;polynomial_checks+=1
    for g in [F(0),F(1,16),F(1,4),F(1),F(3),F(12),F(100)]:
        x=I(g);u=I(F(3,10))*x/(x+1);r=u-x
        v=(I(F(99,100))*x.square()+r.square())/I(F(199,100))
        direct=u-I(F(3,10))*(r-I(F(9,10))*x)/I(F(19,10))/(v.sqrt()+1)
        closed=h2(x)
        assert max(direct.lo,closed.lo)<=min(direct.hi,closed.hi)
        assert closed.lo>=0 and closed.hi<=I(F(3,5)).hi
        scalar_checks+=1
    # Exact global h<=3/5 algebra for r in [7/10,1].
    for k in range(101):
        r=F(7,10)+F(3*k,1000)
        gap=F(19,10)**2*(F(99,100)+r*r)-F(199,100)*(F(9,10)+r)**2
        assert gap==(1-r)*(F(981,500)-F(81,50)*r)>=0;scalar_checks+=1
    return dict(exact_jet_coefficient_checks=jet_checks,exact_polynomial_midpoint_checks=polynomial_checks,
                scalar_recursion_and_bound_checks=scalar_checks,independent_proof_review=False)


def audit(directory):
    root=Path(__file__).resolve().parents[1];result=read(directory/'RESULT.json');quad=read(directory/'QUADRATURE.json')
    for name,digest in result['source_hashes'].items():assert sha(root/name)==digest
    prior=read(root/'revisions/r28_conditional_moments_v2/RESULT.json')
    assert result['C0_center']==prior['C0_moments']['mean_D']
    assert (result['d'],result['n'])==(prior['q']//2,prior['n'])
    assert quad['radius']==result['radius']==12
    assert quad['half_normal_multiplier']==(2/(2*integer.pi_interval()).sqrt()).record()
    assert quad['positive_tail_uppers']==[(I(F(3,5))*density(12)).record(),(I(F(9,300))*density(12)).record()]
    step=F(quad['step']);assert len(quad['cells'])==result['cells']
    sums=[I(0),I(0)];errors=[I(0),I(0)]
    for k,cell in enumerate(quad['cells']):
        assert F(cell['left'])==k*step and F(cell['right'])==(k+1)*step
        for j,name in enumerate(['a','q']):
            item=cell[name];second=restore(item['second_derivative']);M=max(abs(second.lo),abs(second.hi))
            error=I(F(M,S))*I(step**3/24);assert error.record()==item['error']
            sums[j]=sums[j]+restore(item['midpoint']);errors[j]=errors[j]+error
    assert [s.record() for s in sums]==quad['midpoint_sums']
    assert [s.record() for s in errors]==quad['quadrature_errors']
    moments=[];multiplier=restore(quad['half_normal_multiplier'])
    for total,error,tail in zip(sums,errors,quad['positive_tail_uppers']):
        moments.append(multiplier*I.raw(total.lo-error.hi,total.hi+error.hi+restore(tail).hi))
    a,q=moments;assert a.record()==result['a'] and q.record()==result['q']
    c=a/q.sqrt();assert c.record()==result['c']
    theory=I(result['d'])*c/I(result['n']).sqrt();assert theory.record()==result['theory_center']
    shift=restore(result['C0_center'])-theory;assert shift.record()==result['C0_minus_theory_center'] and shift.hi<0
    bound=(-shift)/(2*integer.pi_interval()).sqrt()
    assert str(F(bound.hi,S))==result['auxiliary_unit_Gaussian_center_effect_absolute_upper']
    return dict(cells_audited=result['cells'],source_hashes=len(result['source_hashes']),sign_verified=True,
                derived_center_and_proxy_bound_reconstructed=True,derivative_bounds_inherit_jet_implementation=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('certificate',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--coarse',type=Path);args=p.parse_args();out=dict(reference=references(),audit=audit(args.certificate))
    if args.coarse:
        a=read(args.certificate/'RESULT.json');b=read(args.coarse/'RESULT.json')
        for name in ['a','q','c','theory_center','C0_minus_theory_center']:
            x=restore(a[name]);y=restore(b[name]);assert y.lo<=x.lo<=x.hi<=y.hi,name
        out['fine_five_intervals_nested_in_coarse']=True
    save(args.out,out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
