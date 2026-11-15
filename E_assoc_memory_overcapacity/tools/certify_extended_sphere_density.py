"""512-bit scalar density minorants with asymmetric radius windows on[0,12].

Load an isolated copy of the frozen integer interval module; never change
the precision or globals of any inherited160-bit certificate module.
"""
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
import argparse
import importlib.util
import math

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_r54_integer',ROOT/'tools/certify_weighted_chisquare.py')
integer=importlib.util.module_from_spec(spec);spec.loader.exec_module(integer)
integer.BITS=512;integer.SCALE=1<<512
I=integer.I;S=integer.SCALE
def lo(x):return F(x.lo,S)
def hi(x):return F(x.hi,S)


def exp_positive(x):
    x=F(x);assert 0<=x<=20000
    y=x;power=0
    while y>F(1,8):y/=2;power+=1
    term=I(1);total=I(1)
    for k in range(1,81):term=term*I(y)/k;total=total+term
    next_term=term*I(y)/81;remainder=next_term/(1-I(y)/82)
    value=I.raw(total.lo,total.hi+remainder.hi)
    for _ in range(power):value=value.square()
    return value


def unit(x):return I.raw(max(0,x.lo),min(S,x.hi))


@lru_cache(None)
def normal_cdf(x,pi_low,pi_high):
    x=F(x);assert 0<=x<=16;pi=I.raw(pi_low,pi_high)
    if x>7:
        # Mills: phi(x)/(x+1/x) <= P(Z>x) <= phi(x)/x.
        phi=1/(exp_positive(x*x/2)*(2*pi).sqrt())
        tail_lower=phi/(I(x)+1/I(x));tail_upper=phi/I(x)
        return unit(I.raw((1-tail_upper).lo,(1-tail_lower).hi))
    term=I(x);total=I(0);z=I(x*x/2)
    for k in range(321):
        total=total+term if k%2==0 else total-term
        term=term*z*I(F(2*k+1,(k+1)*(2*k+3)))
    # Alternating even/odd Taylor polynomials bound the exponential before
    # integration, regardless of whether initial terms decrease.
    integrated=I.raw((total-term).lo,total.hi)
    return unit(I(F(1,2))+integrated/(2*pi).sqrt())


def gamma_radius(m,rho):
    assert m>=2 and m%2==0 and F(rho)>0
    shape=m//2;x=F(shape)*F(rho)**2;ex=exp_positive(x);ix=I(x)
    term=I(1);total=I(1)
    for k in range(1,shape):term=term*ix/k;total=total+term
    upper=unit(total/ex)
    return unit(1-upper),upper


def compute(root):
    import certify_sphere_threshold_cdf as old
    old_table=old.read(root/'revisions/r45_sphere_density_lower_v2.json')
    upper_table=old.read(root/'revisions/r44_sphere_density.json')
    m=old_table['m'];s=m-1;even=s-1;assert m==16594
    pi=integer.pi_interval();windows=[]
    design=[(F(17,20),F(5,4),30),(F(9,10),F(6,5),24),(F(19,20),F(11,10),16),(F(97,100),F(21,20),10)]
    for a,b,H in design:
        left=hi((I(F(s,even))*a*a).sqrt())
        right=lo(I(F(s*b*b-H*H,even)).sqrt());assert left<1<right
        lt=hi(gamma_radius(even,left)[0]);rt=hi(gamma_radius(even,right)[1])
        nt=hi(2/exp_positive(F(H*H,2)));mass=1-lt-rt-nt;assert 0<mass<=1
        windows.append(dict(a=str(a),b=str(b),normal_cutoff=H,left_radius=str(left),right_radius=str(right),
            left_tail_upper=str(lt),right_tail_upper=str(rt),normal_tail_upper=str(nt),mass_lower=str(mass)))
    h=m//2;normalizer=I(F(4**(h-1),math.comb(2*h-2,h-1)))/(pi*I(m).sqrt())
    grid=[F(j,32) for j in range(385)]
    xs=[(I(s*t*t)/(m-t*t)).sqrt() for t in grid];marginal=[]
    for t in grid:
        v=I(1-t*t/m);marginal.append(lo(normalizer*v**((m-4)//2)*v.sqrt()))
    rows=[]
    for length in [1,5]:
        k=length-1;bins=[];previous=next(r for r in old_table['rows'] if r['L']==length)
        up=next(r for r in upper_table['rows'] if r['L']==length)
        for j,(a,b) in enumerate(zip(grid,grid[1:])):
            candidates=[]
            for w in windows:
                xa=lo(xs[j]*F(w['a']));xb=hi(xs[j+1]*F(w['b']))
                first=normal_cdf(xa,pi.lo,pi.hi);second=normal_cdf(xb,pi.lo,pi.hi)
                event=lo(math.comb(s,k)*(1-second)**k*first**(s-k))
                mass=F(w['mass_lower']);candidates.append(max(F(0),(event-1+mass)/mass))
            fresh=m*marginal[j+1]*max(candidates)
            old_lower=F(previous['bins'][j]['density_lower']) if j<len(previous['bins']) else F(0)
            bound=max(fresh,old_lower)
            if j<len(up['bins']):assert bound<=F(up['bins'][j]['density_upper'])
            bins.append(dict(left=str(a),right=str(b),density_lower=str(bound),new_window=candidates.index(max(candidates)),strictly_above_old=bound>old_lower))
        rows.append(dict(L=length,bins=bins,tail_density_lower='0',positive_bins=sum(F(b['density_lower'])>0 for b in bins),strictly_tightened_bins=sum(b['strictly_above_old'] for b in bins)))
        print('EXTENDED DENSITY',length,'positive',rows[-1]['positive_bins'],'improved',rows[-1]['strictly_tightened_bins'],flush=True)
    return dict(m=m,residual_dimension=s,grid_step='1/32',grid_end='12',arithmetic_bits=512,
        windows=windows,rows=rows,normal_CDF_uses_Taylor_up_to7_and_Mills_above7=True,
        normal_cutoff_uses_two_sided_Chernoff=True,old_density_minorants_retained=True,
        inherited_160bit_module_globals_unchanged=True,new_matrices=0,new_samples=0,
        new_scalar_probability_enclosures=True,new_high_dimensional_probability_integrals=0,
        independent_proof_review=False)


def references():
    # Independent analytic Mills inequalities: integration by parts gives
    # q=phi/x-int_x^infty phi(t)/t^2 dt, with remainder<=q/x^2.
    # These finite checks verify the chosen interval branches/scales.
    pi=integer.pi_interval();checks=0
    import certify_sphere_threshold_cdf as old
    old_pi=old.integer.pi_interval()
    for x in [F(0),F(1,2),F(1),F(2),F(4),F(6),F(7)]:
        a=normal_cdf(x,pi.lo,pi.hi);b=old.normal_cdf(x,old_pi)
        assert lo(a)<=F(b.hi,old.S) and F(b.lo,old.S)<=hi(a);checks+=1
    for x in [F(15,2),F(8),F(12),F(16)]:
        a=normal_cdf(x,pi.lo,pi.hi);assert 0<=lo(a)<=hi(a)<=1
        phi=1/(exp_positive(x*x/2)*(2*pi).sqrt())
        assert lo(1-a)<=hi(phi/I(x)) and hi(1-a)>=lo(phi/(I(x)+1/I(x)));checks+=1
    # New exponential intervals satisfy multiplication and reciprocal checks.
    for x in [F(0),F(1,8),F(1),F(10),F(72),F(450)]:
        a=exp_positive(x);b=exp_positive(x/2).square()
        assert a.lo<=b.hi and b.lo<=a.hi and (a*(1/a)).contains(1)
    return dict(normal_CDF_branch_and_inherited_overlap_cases=checks,exponential_half_argument_and_reciprocal_cases=6)


def sources(root):
    import certify_sphere_threshold_cdf as old
    names=['tools/certify_extended_sphere_density.py','tools/certify_weighted_chisquare.py',
        'tools/certify_sphere_threshold_cdf.py','revisions/r45_sphere_density_lower_v2.json','revisions/r44_sphere_density.json']
    receipt=old.read(root/'revisions/R53_HISTORY.json');manifest=old.read(root/receipt['path']/'manifest.json')
    for n in names[1:]:assert old.sha(root/n)==manifest['files'][n],n
    return {n:old.sha(root/n) for n in names}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--compare',type=Path);a=p.parse_args()
    import certify_sphere_threshold_cdf as old
    hashes=sources(ROOT);v=compute(ROOT);v['reference']=references();v['source_hashes']=hashes
    if a.compare:assert old.read(a.compare)==v
    old.save(a.out,v)

if __name__=='__main__':main()
