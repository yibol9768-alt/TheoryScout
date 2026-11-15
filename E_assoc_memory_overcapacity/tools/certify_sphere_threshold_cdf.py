"""Deterministic sphere order-threshold CDF enclosures via Gaussian radii.

Reads certified R20 point LOWER bounds, never the sampled threshold bank.
Integer interval arithmetic only; gamma shape m/2 must be an integer.
The final recall bound is conditional on R13's exact represented C0 model.
"""
from fractions import Fraction as F
from pathlib import Path
from itertools import product
import argparse
import hashlib
import json
import math
import time

import certify_weighted_chisquare as integer

I=integer.I
S=integer.SCALE


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
def unit(value):
    return I.raw(max(0,value.lo),min(S,value.hi))


def exp_positive(x):
    """Positive Taylor series with geometric remainder, then exact squaring."""
    x=F(x);assert 0<=x<=10000
    reductions=0;y=x
    while y>F(1,8):y/=2;reductions+=1
    term=I(1);total=I(1)
    for k in range(1,41):term=term*I(y)/k;total=total+term
    next_term=term*I(y)/41
    remainder=next_term/(1-I(y)/42)
    enclosure=I.raw(total.lo,total.hi+remainder.hi)
    for _ in range(reductions):enclosure=enclosure.square()
    return enclosure


def normal_cdf(x,pi):
    """Odd/even integrated Taylor bounds for exp(-u^2/2), 0<=x<=7."""
    x=F(x);assert 0<=x<=7
    term=I(x);total=I(0);z=I(x*x/2)
    for k in range(161):
        total=total+term if k%2==0 else total-term
        term=term*z*I(F(2*k+1,(k+1)*(2*k+3)))
    # k=160 is even; the next (negative) term gives the odd lower sum.
    integrated=I.raw((total-term).lo,total.hi)
    return unit(I(F(1,2))+integrated/(2*pi).sqrt())


def gamma_radius(m,rho):
    """rho=||G||/sqrt(m); integer-shape gamma CDF by finite positive sum."""
    assert m>=2 and m%2==0 and F(rho)>0
    shape=m//2;x=F(shape)*F(rho)**2;ex=exp_positive(x);ix=I(x)
    term=I(1);total=I(1)
    for k in range(1,shape):term=term*ix/k;total=total+term
    upper=unit(total/ex);lower=unit(1-upper)
    return lower,upper


def gaussian_order_cdf(m,length,x,pi):
    phi=normal_cdf(x,pi);tail=1-phi
    return unit(sum(math.comb(m,k)*tail**k*phi**(m-k) for k in range(length)))


def threshold_cdf(m,length,t,pi,radii):
    lower=F(0);upper=F(1);best_low=None;best_high=None
    for rho,delta in radii['lower']:
        # Independence of direction and radius refines the union bound.
        g=gaussian_order_cdf(m,length,t*rho,pi)
        d=F(delta.hi,S);assert d<1
        bound=max(F(0),F(((g-I(d))/(1-I(d))).lo,S))
        if bound>lower:lower=bound;best_low=str(rho)
    for rho,delta in radii['upper']:
        g=gaussian_order_cdf(m,length,t*rho,pi)
        d=F(delta.hi,S);assert d<1
        bound=min(F(1),F((g/(1-I(d))).hi,S))
        if bound<upper:upper=bound;best_high=str(rho)
    assert lower<=upper
    return dict(t=str(t),lower=str(lower),upper=str(upper),
                approximate=[float(lower),float(upper)],lower_radius=best_low,upper_radius=best_high)


def event_checks():
    checks=0
    for m in [2,3]:
        for vector in product(map(F,[-3,-1,0,1,3]),repeat=m):
            norm2=sum(v*v for v in vector)/m
            if not norm2:continue
            for length in range(1,m+1):
                z=sorted(vector,reverse=True)[length-1]
                for t in [F(1,2),F(1),F(2)]:
                    success=z<=0 or z*z<=t*t*norm2
                    for rho in [F(1,2),F(9,10)]:
                        if z<=rho*t and norm2>=rho*rho:assert success
                        checks+=1
                    for rho in [F(11,10),F(2)]:
                        if success and norm2<=rho*rho:assert z<=rho*t
                        checks+=1
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic()
    receipt=read(root/'revisions/R20_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
    protected=['revisions/r20_threshold_family/RESULT.json','revisions/r20_threshold_family/DESIGN.json',
               'tools/certify_weighted_chisquare.py']
    for p in protected:assert sha(root/p)==manifest['files'][p]
    points=read(root/protected[0]);design=read(root/protected[1]);m=design['n']-1
    assert m==16594 and points['fixed_node_lower_numerical_inclusion']
    pi=integer.pi_interval();radii={'lower':[],'upper':[]};radius_records=[]
    # Deterministic rational grid, not a statistical design or new samples.
    for sign,label in [(-1,'lower'),(1,'upper')]:
        for k in range(1,51):
            rho=1+sign*F(k,1000);lo,hi=gamma_radius(m,rho);delta=lo if sign<0 else hi
            radii[label].append((rho,delta))
            radius_records.append(dict(radius=str(rho),tail=label,probability=delta.record()))
        print('radius',label,'complete',time.monotonic()-start,flush=True)
    save(args.out/'RADII.json',dict(m=m,method='Integer-shape gamma finite sum / enclosed exp',records=radius_records))
    certified=points['certified_nodes'];ts=sorted({F(v['t']) for v in certified});assert len(ts)==18
    cdf_rows=[];recall=[]
    # p_C0(t) is common to both lists. Use every certified node for each list.
    heights=[max(F(v['target_probability_lower']) for v in certified if F(v['t'])==t) for t in ts]
    for j in range(len(ts)-2,-1,-1):heights[j]=max(heights[j],heights[j+1])
    heights.append(F(0))
    for length in [1,5]:
        nodes=[]
        for index,t in enumerate(ts):
            row=threshold_cdf(m,length,t,pi,radii);row['index']=index;nodes.append(row)
            save(args.out/f'L{length}_cdf{index:02d}.json',row)
        lows=list(map(lambda r:F(r['lower']),nodes));highs=list(map(lambda r:F(r['upper']),nodes))
        for j in range(1,len(ts)):lows[j]=max(lows[j],lows[j-1])
        for j in range(len(ts)-2,-1,-1):highs[j]=min(highs[j],highs[j+1])
        assert all(lo<=hi for lo,hi in zip(lows,highs))
        bound=sum((heights[j]-heights[j+1])*lows[j] for j in range(len(ts)))
        contributions=[dict(t=str(ts[j]),jump=str(heights[j]-heights[j+1]),cdf_lower=str(lows[j]),
                            contribution=str((heights[j]-heights[j+1])*lows[j])) for j in range(len(ts))]
        recall.append(dict(L=length,conditional_recall_lower=str(bound),approximate=float(bound),
                           contributions=contributions))
        cdf_rows.append(dict(L=length,nodes=nodes,monotone_lower=list(map(str,lows)),monotone_upper=list(map(str,highs))))
        print('L',length,'deterministic conditional recall LOWER',float(bound),flush=True)
    # Circle references: at t=1, F_{2,1}=1/2 and F_{2,2}=1.
    toy_radii={'lower':[],'upper':[]}
    for k in range(1,10):
        rho=F(k,10);toy_radii['lower'].append((rho,gamma_radius(2,rho)[0]))
    for k in range(11,31):
        rho=F(k,10);toy_radii['upper'].append((rho,gamma_radius(2,rho)[1]))
    circle=[]
    for length,target in [(1,F(1,2)),(2,F(1))]:
        v=threshold_cdf(2,length,F(1),pi,toy_radii);assert F(v['lower'])<=target<=F(v['upper']);circle.append(v)
    output=dict(m=m,certified_C0_point_nodes=len(ts),cdf_enclosures=cdf_rows,conditional_recall_lower=recall,
                exact_event_inclusion_checks=event_checks(),circle_reference_checks=circle,
                reads_threshold_samples=False,new_samples=0,statistical_failure_budget_used=None,
                threshold_distribution_method='Gaussian order statistics + independent exact gamma radius CDF',
                scope='Deterministic LOWER bound for full random-threshold conditional recall at exact represented C0; inherits R9-R13 numeric assumptions',
                independent_review=False,no_C0_recall_upper=True,no_unrounded_Gaussian_input_certificate=True,
                source_hashes={p:sha(root/p) for p in protected},wall_seconds=time.monotonic()-start)
    save(args.out/'RESULT.json',output)
    print(json.dumps(dict(lower=[dict(L=r['L'],lower=r['approximate']) for r in recall],
                         event_checks=output['exact_event_inclusion_checks'],wall_seconds=output['wall_seconds']),indent=2))


if __name__=='__main__':main()
