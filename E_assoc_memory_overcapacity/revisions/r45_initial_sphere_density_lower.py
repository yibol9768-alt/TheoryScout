"""Positive sphere-order density minorants from Gaussian radius/count bands."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import math

import certify_sphere_order_density as upper

I=upper.I
low=upper.low
high=upper.high

def compute(previous):
    m=previous['m'];s=m-1;h=m//2;pi=upper.sphere.integer.pi_interval()
    normalizer=I(F(4**(h-1),math.comb(2*h-2,h-1)))/(pi*I(m).sqrt())
    windows=previous['radius_windows']
    assert windows==upper.radius_windows(s,pi)
    rows=[]
    for old in previous['rows']:
        length=old['L'];bins=[]
        for cell in old['bins']:
            a,b=F(cell['left']),F(cell['right'])
            xl=(I(s*a*a)/(m-a*a)).sqrt();xh=(I(s*b*b)/(m-b*b)).sqrt()
            candidates=[]
            for window in windows:
                xa=low(xl*F(window['a']));xb=high(xh*F(window['b']))
                # Count at low threshold <= k and count at high >= k imply
                # exactly k throughout.  The CDF difference is only a union
                # lower bound here: these two complement events are NOT nested.
                first=low(upper.sphere.gaussian_order_cdf(s,length,xa,pi))
                second=high(upper.sphere.gaussian_order_cdf(s,length-1,xb,pi)) if length>1 else F(0)
                mass=F(window['mass_lower']);delta=1-mass
                candidates.append(max(F(0),(first-second-delta)/mass))
            marginal_base=I(1-b*b/m)
            marginal=low(normalizer*marginal_base**((m-4)//2)*marginal_base.sqrt())
            bound=m*marginal*max(candidates)
            assert 0<=bound<=F(cell['density_upper'])
            bins.append(dict(left=cell['left'],right=cell['right'],density_lower=str(bound),
                             radius_window=candidates.index(max(candidates))))
        rows.append(dict(L=length,bins=bins,tail_density_lower='0',
                         positive_bins=sum(F(c['density_lower'])>0 for c in bins),
                         largest_bin_minorant=str(max(F(c['density_lower']) for c in bins))))
    return dict(m=m,residual_dimension=s,grid_step=previous['grid_step'],grid_end=previous['grid_end'],rows=rows,
                arithmetic_bits=upper.sphere.integer.BITS,probability_samples_used=False,independent_proof_review=False)

def references():
    cases=0
    for vector in product(map(F,[-2,0,1,3]),repeat=3):
        for x in [F(1,2),F(1),F(2)]:
            for a,b in [(F(1,2),F(3,2)),(F(1),F(2))]:
                for radius in map(F,['1/4','1/2','1','3/2','2','3']):
                    target=sum(v>x for v in vector)
                    small=sum(radius*v>x*b for v in vector);large=sum(radius*v>x*a for v in vector)
                    for k in range(3):
                        event=large<=k and small>=k
                        assert int(large<=k)-int(small<=k-1)<=int(event)
                        if event and a<=radius<=b:assert target==k
                        cases+=1
    mixture_cases=0
    # Radius-window probability can be greater than its certified lower
    # bound. The proposed minorant must remain valid for every such mass.
    for p,mass in product([F(0),F(1,5),F(3,4),F(1)],[F(1,2),F(4,5),F(1)]):
        max_event=p*mass+1-mass
        for lower_mass in [mass/2,mass]:
            for lower_event in [F(0),max_event/2,max_event]:
                bound=max(F(0),(lower_event-(1-lower_mass))/lower_mass)
                assert bound<=p;mixture_cases+=1
    return dict(lower_count_event_cases=cases,radius_probability_lower_bound_cases=mixture_cases)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];previous=upper.sphere.read(root/'revisions/r44_sphere_density.json')
    result=compute(previous);result['references']=references()
    names=['revisions/r44_sphere_density.json','tools/certify_sphere_density_lower.py','tools/certify_sphere_order_density.py',
           'tools/certify_sphere_threshold_cdf.py','tools/certify_weighted_chisquare.py']
    for n,digest in previous['source_hashes'].items():assert upper.sphere.sha(root/n)==digest
    result['source_hashes']={n:upper.sphere.sha(root/n) for n in names}
    upper.sphere.save(a.out,result)
    print('LOWER DENSITY',[(r['L'],r['positive_bins'],float(F(r['largest_bin_minorant']))) for r in result['rows']],flush=True)

if __name__=='__main__':main()
