"""Finite monotone refinements of R21's radial mixture CDF enclosures.

Only inherited deterministic CDF/radius intervals and point bounds are read.
An upper bound for the known lower-staircase expectation is NOT an upper
bound for the target recall. No new random data or quadratic integrals.
"""
from fractions import Fraction as F
from pathlib import Path
from bisect import bisect_left,bisect_right
from itertools import product
import argparse
import hashlib
import json
import time

import certify_sphere_threshold_cdf as sphere

S=sphere.S


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def floor_unit(x):return F(max(0,min(S,x.numerator*S//x.denominator)),S)
def ceil_unit(x):return F(max(0,min(S,-((-x.numerator*S)//x.denominator))),S)


def mixture_lower(k,H,gamma_upper,upper):
    """F(t_k) lower using H(x) and upper CDF at nodes beyond k."""
    extended=upper+[F(1)];g=gamma_upper[k]
    assert 0<=g<1
    tail=sum((extended[j+1]-extended[j])*gamma_upper[j] for j in range(k+1,len(upper)))
    return floor_unit((H-g*extended[k+1]-tail)/(1-g))


def mixture_upper(k,H,gamma_lower,lower):
    """F(t_k) upper using H(x) and lower CDF at nodes before k."""
    g=gamma_lower[k];assert 0<g<=1
    previous=lower[k-1] if k else F(0)
    subtotal=sum((lower[j]-(lower[j-1] if j else 0))*gamma_lower[j] for j in range(k))
    return ceil_unit((H-subtotal+previous*g)/g)


class RadialCDF:
    def __init__(self,records):
        points=[]
        for r in records:
            p=r['probability'];lo=F(int(p['lower_integer']),S);hi=F(int(p['upper_integer']),S)
            if r['tail']=='upper':lo,hi=1-hi,1-lo
            points.append((F(r['radius']),lo,hi))
        points.sort();self.x=[p[0] for p in points];self.lo=[p[1] for p in points];self.hi=[p[2] for p in points]
        for j in range(1,len(points)):self.lo[j]=max(self.lo[j],self.lo[j-1])
        for j in range(len(points)-2,-1,-1):self.hi[j]=min(self.hi[j],self.hi[j+1])
        assert all(a<=b for a,b in zip(self.lo,self.hi))
    def __call__(self,x):
        left=bisect_right(self.x,x)-1;right=bisect_left(self.x,x)
        return (self.lo[left] if left>=0 else F(0),self.hi[right] if right<len(self.x) else F(1))


def toy_checks():
    """Independent finite product laws, including negative T and atoms."""
    ts=[F(1,2),F(1),F(2),F(3)];qpoints=[F(1,2),F(1),F(3,2)]
    tpoints=list(map(F,[-1,0]))+ts
    laws=[v for v in product(range(3),repeat=len(tpoints)) if sum(v)==2]
    qlaws=[v for v in product(range(4),repeat=3) if sum(v)==3];checks=0
    for weights in laws:
        true=[sum(F(w,2) for t,w in zip(tpoints,weights) if t<=node) for node in ts]
        lo=[v/2 for v in true];hi=[(1+v)/2 for v in true]
        for qw in qlaws:
            for k,t in enumerate(ts):
                for factor in [F(3,4),F(5,4)]:
                    x=t*factor
                    H=sum(F(w,2)*F(q,3) for value,w in zip(tpoints,weights) for radius,q in zip(qpoints,qw) if value*radius<=x)
                    G=[sum(F(q,3) for radius,q in zip(qpoints,qw) if radius<=x/node) for node in ts]
                    if G[k]<1:assert mixture_lower(k,H,G,hi)<=true[k];checks+=1
                    if G[k]>0:assert mixture_upper(k,H,G,lo)>=true[k];checks+=1
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sweeps',type=int,default=8);args=p.parse_args();assert 1<=args.sweeps<=32
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic()
    hist=read(root/'revisions/R21_HISTORY.json');manifest=read(root/hist['path']/'manifest.json')
    names=['revisions/r21_sphere_cdf/RESULT.json','revisions/r21_sphere_cdf/RADII.json',
           'revisions/r20_threshold_family/RESULT.json','tools/certify_sphere_threshold_cdf.py','tools/certify_weighted_chisquare.py']
    for name in names:assert sha(root/name)==manifest['files'][name],name
    original=read(root/names[0]);radial=read(root/names[1]);points=read(root/names[2])['certified_nodes'];m=original['m']
    grid=RadialCDF(radial['records']);pi=sphere.integer.pi_interval();rows=[]
    for source in original['cdf_enclosures']:
        length=source['L'];ts=[F(v['t']) for v in source['nodes']]
        lo=list(map(F,source['monotone_lower']));hi=list(map(F,source['monotone_upper']))
        heights=[max(F(v['target_probability_lower']) for v in points if F(v['t'])==t) for t in ts]
        for j in range(len(ts)-2,-1,-1):heights[j]=max(heights[j],heights[j+1])
        jumps=[heights[j]-(heights[j+1] if j+1<len(ts) else 0) for j in range(len(ts))]
        def summary(index,low,high):
            bottom=sum(w*f for w,f in zip(jumps,low));top=sum(w*f for w,f in zip(jumps,high))
            return dict(sweep=index,staircase_expectation_lower=str(bottom),staircase_expectation_upper=str(top),
                        lower_approximate=float(bottom),upper_approximate=float(top),width_approximate=float(top-bottom),
                        lower_CDF=list(map(str,low)),upper_CDF=list(map(str,high)))
        sweeps=[summary(0,lo,hi)];candidates=[]
        for k,t in enumerate(ts):
            sides={}
            for sign,label in [(-1,'lower'),(1,'upper')]:
                entries=[]
                for step in range(1,51):
                    x=t*(1+sign*F(step,1000));H=sphere.gaussian_order_cdf(m,length,x,pi)
                    G=[grid(x/node) for node in ts]
                    entries.append((F(H.lo if sign<0 else H.hi,S),[v[1 if sign<0 else 0] for v in G]))
                sides[label]=entries
            candidates.append(sides)
        for sweep in range(1,args.sweeps+1):
            newlo=lo.copy();newhi=hi.copy()
            # Jacobi sweep: every new bound uses the previous enclosure only.
            for k in range(len(ts)):
                for H,G in candidates[k]['lower']:newlo[k]=max(newlo[k],mixture_lower(k,H,G,hi))
                for H,G in candidates[k]['upper']:newhi[k]=min(newhi[k],mixture_upper(k,H,G,lo))
            for k in range(1,len(ts)):newlo[k]=max(newlo[k],newlo[k-1])
            for k in range(len(ts)-2,-1,-1):newhi[k]=min(newhi[k],newhi[k+1])
            assert all(a<=b for a,b in zip(newlo,newhi))
            lo,hi=newlo,newhi;sweeps.append(summary(sweep,lo,hi))
        rows.append(dict(L=length,t=list(map(str,ts)),jumps=list(map(str,jumps)),sweeps=sweeps,
                         upper_is_only_for_known_lower_staircase=True))
        print('L',length,'initial',sweeps[0]['lower_approximate'],sweeps[0]['upper_approximate'],
              'refined',sweeps[-1]['lower_approximate'],sweeps[-1]['upper_approximate'],flush=True)
    tests=toy_checks()
    result=dict(sweeps=args.sweeps,rows=rows,exact_discrete_product_checks=tests,new_samples=0,new_point_integrals=0,
                new_radial_integrals=0,uses_threshold_bank=False,independent_review=False,
                source_hashes={name:sha(root/name) for name in names},wall_seconds=time.monotonic()-start,
                scope='Certified finite monotone CDF refinement; upper endpoint only bounds the supplied lower staircase expectation, not target recall')
    save(args.out/'RESULT.json',result);print('Exact product checks',tests,'wall',result['wall_seconds'],flush=True)


if __name__=='__main__':main()
