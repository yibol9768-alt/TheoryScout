"""Exact finite references for sphere count brackets and signed CDF envelopes.

No Gaussian numerical integration or represented-C0 probability claim.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import json
import math

def checks():
    values=list(map(F,[-2,0,1,3]));weights=list(map(F,['1/10','2/10','3/10','4/10']))
    multinomial=0
    for size in range(1,5):
        for a,b in [(F(1,2),F(1)),(F(1),F(2)),(F(1,2),F(3))]:
            plo=sum(w for v,w in zip(values,weights) if v<=a);phi=sum(w for v,w in zip(values,weights) if v>b)
            for k in range(size+1):
                total=sum(math.prod(weights[j] for j in choice) for choice in product(range(4),repeat=size)
                          if sum(values[j]>a for j in choice)==sum(values[j]>b for j in choice)==k)
                assert total==math.comb(size,k)*phi**k*plo**(size-k);multinomial+=1
    radii=list(map(F,['1/4','1/2','1','3/2','2','3']));brackets=0
    for t in [F(1,2),F(1),F(2)]:
        for a,b in [(F(1,2),F(3,2)),(F(1),F(2))]:
            mass=F(sum(a<=r<=b for r in radii),len(radii))
            for k in range(4):
                target=F(0);band=F(0);wide=F(0)
                for choice in product(range(4),repeat=3):
                    vector=[values[j] for j in choice];w=math.prod(weights[j] for j in choice)
                    target+=w*int(sum(v>t for v in vector)==k)
                    for r in radii:
                        lo=sum(r*v>t*a for v in vector);hi=sum(r*v>t*b for v in vector)
                        band+=w/F(len(radii))*int(lo==hi==k)
                        wide+=w/F(len(radii))*(int(hi<=k)-int(lo<=k-1))
                for certified_mass in [mass/2,mass]:
                    assert max(F(0),(band-1+certified_mass)/certified_mass)<=target<=min(F(1),wide/certified_mass)
                    brackets+=1
    segments=[(F(j),F(j+1),w) for j,w in enumerate([F(1,4),F(1,2),F(1,4)])]
    def cdf(x):return sum(w*max(F(0),min(F(1),x-a)) for a,b,w in segments)
    envelopes=0;positive=0;negative=0
    for dl,dh in [(F(0),F(1,2)),(F(1,2),F(1)),(F(1),F(2)),(F(2),F(4)),(F(4),F(10))]:
        for rl,rh in [(F(1,2),F(3,4)),(F(1),F(3,2)),(F(3,2),F(2))]:
            for ml,mh in [(F(3,4),F(5,4)),(F(1),F(3,2))]:
                left=dl/max(rh,mh);right=dh/min(rl,ml)
                densities=[w for a,b,w in segments if left<=b and a<=right]
                if right>=3:densities.append(F(0))
                lower_density=min(densities);upper_density=max(densities)
                al=1/rh-1/ml;ah=1/rl-1/mh
                lower=lower_density*dl*al if al>=0 else -upper_density*dh*max(abs(al),abs(ah))
                upper=lower_density*dl*ah if ah<=0 else upper_density*dh*max(abs(al),abs(ah))
                positive+=int(lower>0);negative+=int(upper<0)
                for d,r,mu in product([dl,(dl+dh)/2,dh],[rl,(rl+rh)/2,rh],[ml,(ml+mh)/2,mh]):
                    assert lower<=cdf(d/r)-cdf(d/mu)<=upper;envelopes+=1
    assert positive>0 and negative>0
    # Half-integer Gamma normalization, exact factorial ratio recurrence.
    ratios=[F(4**j,math.comb(2*j,j)) for j in range(1,12)]
    for j in range(1,11):assert ratios[j]/ratios[j-1]==F(2*j+2,2*j+1)
    return dict(multinomial_empty_band_cases=multinomial,finite_independent_radius_direction_brackets=brackets,
                signed_local_density_envelope_cases=envelopes,strict_positive_cells=positive,strict_negative_cells=negative,
                half_integer_normalizer_recurrences=10,actual_C0_probabilities_computed=False,independent_proof_review=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=checks()
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('PASS order-density count, normalization and signed energy references')

if __name__=='__main__':main()
