"""Outward density upper bounds for actual sphere order statistics (stdlib).

Uses a conditional-coordinate identity and Gaussian radius/count inclusions.
No threshold samples, optimization, or floating-point probability evaluation.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import math

import certify_sphere_threshold_cdf as sphere

I=sphere.I
S=sphere.S

def low(x):return F(x.lo,S)
def high(x):return F(x.hi,S)

def radius_windows(s,pi):
    assert s%2==1 and s>25
    even=s-1;normal_tail=2*(1-low(sphere.normal_cdf(F(5),pi)))
    rows=[]
    for epsilon in map(F,['1/200','3/400','1/100','3/200']):
        a,b=1-epsilon,1+epsilon
        left=high((I(F(s,even))*a*a).sqrt())
        right=low(I((s*b*b-25)/even).sqrt())
        left_tail=high(sphere.gamma_radius(even,left)[0])
        right_tail=high(sphere.gamma_radius(even,right)[1])+normal_tail
        mass=1-left_tail-right_tail
        assert 0<mass<=1
        rows.append(dict(a=str(a),b=str(b),mass_lower=str(mass),
                         left_radius=str(left),right_radius=str(right),
                         left_tail_upper=str(left_tail),right_tail_upper=str(right_tail)))
    return rows

def compute(m=16594):
    assert m>=6 and m%2==0
    pi=sphere.integer.pi_interval();s=m-1;h=m//2
    normalizer=I(F(4**(h-1),math.comb(2*h-2,h-1)))/(pi*I(m).sqrt())
    windows=radius_windows(s,pi)
    ts=[F(j,32) for j in range(193)]
    xs=[(I(s*t*t)/(m-t*t)).sqrt() for t in ts]
    marginal=[]
    for t in ts:
        base=I(1-t*t/m)
        marginal.append(high(normalizer*base**((m-4)//2)*base.sqrt()))
    # Fixed design: [0,6] in steps 1/32, then monotone marginal tail.
    rows=[]
    for length in [1,5]:
        bins=[]
        for j,(a,b) in enumerate(zip(ts,ts[1:])):
            candidates=[]
            for window in windows:
                xa=low(xs[j]*F(window['a']));xb=high(xs[j+1]*F(window['b']))
                assert 0<=xa<=xb<=7
                ghigh=high(sphere.gaussian_order_cdf(s,length,xb,pi))
                glow=low(sphere.gaussian_order_cdf(s,length-1,xa,pi)) if length>1 else F(0)
                conditional=min(F(1),max(F(0),(ghigh-glow)/F(window['mass_lower'])))
                candidates.append(conditional)
            bound=m*marginal[j]*min(candidates)
            bins.append(dict(left=str(a),right=str(b),density_upper=str(bound),
                             radius_window=candidates.index(min(candidates))))
        tail=m*marginal[-1]
        rows.append(dict(L=length,bins=bins,tail_from='6',tail_density_upper=str(tail),
                         global_density_upper=str(max([tail]+[F(x['density_upper']) for x in bins]))))
    return dict(m=m,residual_dimension=s,grid_step='1/32',grid_end='6',normal_split_square=25,
                normalizer=normalizer.record(),radius_windows=windows,rows=rows,
                arithmetic_bits=sphere.integer.BITS,threshold_samples_used=False,
                negative_threshold_treatment='Inherited binomial sign bound, not this density table',
                independent_proof_review=False)

def references():
    counts=0
    # Finite direction and independent radius laws, atoms included.  Verify
    # inclusion and the nested-event CDF difference before averaging.
    for vector in product(map(F,[-2,0,1,3]),repeat=3):
        for x in [F(1,2),F(1),F(2)]:
            for a,b in [(F(1,2),F(3,2)),(F(1),F(2))]:
                for radius in map(F,['1/4','1/2','1','3/2','2','3']):
                    direction=sum(v>x for v in vector)
                    small=sum(radius*v>x*b for v in vector)
                    large=sum(radius*v>x*a for v in vector)
                    assert small<=large
                    for k in range(3):
                        if a<=radius<=b and direction==k:assert small<=k<=large
                        assert int(small<=k)-int(large<=k-1)==int(small<=k<=large)
                        counts+=1
    # Half-integer Gamma factorial normalizer satisfies its exact recurrence.
    ratios=[F(4**j,math.comb(2*j,j)) for j in range(1,12)]
    for j in range(1,11):assert ratios[j]/ratios[j-1]==F(2*j+2,2*j+1)
    return dict(radius_count_event_cases=counts,normalizer_recurrences=10)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    result=compute();result['references']=references()
    names=['tools/certify_sphere_order_density.py','tools/certify_sphere_threshold_cdf.py','tools/certify_weighted_chisquare.py',
           'revisions/r28_conditional_moments_v2/INPUTS.json','revisions/r21_sphere_cdf/RESULT.json']
    assert sphere.read(root/names[-2])['n']-1==result['m']==sphere.read(root/names[-1])['m']
    result['source_hashes']={n:sphere.sha(root/n) for n in names}
    sphere.save(a.out,result)
    print('DENSITY',[(r['L'],float(F(r['global_density_upper']))) for r in result['rows']],flush=True)

if __name__=='__main__':main()
