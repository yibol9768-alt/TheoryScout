"""Certify Adam2 scalar efficiency and fixed-C0 center shift.

Composite midpoint integration with interval second derivatives, exact
Gaussian tail bounds, rational parameters and 160-bit outward arithmetic.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse
import hashlib
import json
import time

import certify_weighted_chisquare as integer
from certify_sphere_threshold_cdf import exp_positive

I=integer.I
S=integer.SCALE


class J:
    """Interval enclosures of f, f', f''/2 at every point of a cell."""
    def __init__(self,x=0,d=0,d2=0):self.v=(I(x),I(d),I(d2))
    @classmethod
    def values(cls,v):return cls(*v)
    @staticmethod
    def cast(x):return x if isinstance(x,J) else J(x)
    def __add__(self,other):
        b=J.cast(other);return J.values([a+c for a,c in zip(self.v,b.v)])
    __radd__=__add__
    def __neg__(self):return J.values([-a for a in self.v])
    def __sub__(self,b):return self+-J.cast(b)
    def __rsub__(self,b):return J.cast(b)+-self
    def __mul__(self,other):
        b=J.cast(other);return J.values([sum(self.v[j]*b.v[k-j] for j in range(k+1)) for k in range(3)])
    __rmul__=__mul__
    def reciprocal(self):
        a,b,c=self.v;return J(1/a,-b/a.square(),b.square()/(a**3)-c/a.square())
    def __truediv__(self,b):return self*J.cast(b).reciprocal()
    def __rtruediv__(self,b):return J.cast(b)*self.reciprocal()
    def sqrt(self):
        a,b,c=self.v;s=a.sqrt();return J(s,b/(2*s),c/(2*s)-b.square()/(8*s**3))


def h2(x):
    # Exact rational alpha=3/10, beta1=9/10, beta2=99/100, e=1.
    r=1-F(3,10)/(x+1)
    s=((F(99,100)+r*r)/F(199,100)).sqrt()
    return F(3,10)*x/(x+1)+F(3,10)*x*(F(9,10)+r)/F(19,10)/(1+x*s)


def density(x):return 1/exp_positive(F(x)**2/2)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def restore(r):
    assert r['denominator_power2']==160
    return I.raw(int(r['lower_integer']),int(r['upper_integer']))


def integrate(cells,radius):
    step=F(radius,cells);sums=[I(0),I(0)];errors=[I(0),I(0)];records=[]
    for k in range(cells):
        left=k*step;right=(k+1)*step;mid=(left+right)/2
        X=J(I.raw(I(left).lo,I(right).hi),1);H=h2(X)
        point=I(mid);hp=h2(point);dm=density(mid)
        ex=I.raw(density(right).lo,density(left).hi)
        cell={}
        for index,(func,value) in enumerate([(X*H,point*hp),(H*H,hp.square())]):
            a,b,c=func.v
            second=ex*(2*c-2*X.v[0]*b+(X.v[0].square()-1)*a)
            bound=max(abs(second.lo),abs(second.hi))
            error=I(F(bound,S))*I(step**3/24)
            sums[index]=sums[index]+I(step)*value*dm
            errors[index]=errors[index]+error
            cell['a' if index==0 else 'q']=dict(midpoint=(I(step)*value*dm).record(),second_derivative=second.record(),error=error.record())
        records.append(dict(left=str(left),right=str(right),**cell))
    # 0 <= h2 <= 3/5 on [0,infinity); Mills: int_R^inf exp(-g²/2) <= exp(-R²/2)/R.
    tails=[I(F(3,5))*density(radius),I(F(9,25*radius))*density(radius)]
    multiplier=2/(2*integer.pi_interval()).sqrt();out=[]
    for value,error,tail in zip(sums,errors,tails):
        out.append(multiplier*I.raw(value.lo-error.hi,value.hi+error.hi+tail.hi))
    return out,dict(cells=records,step=str(step),radius=radius,
                   midpoint_sums=[v.record() for v in sums],quadrature_errors=[v.record() for v in errors],
                   positive_tail_uppers=[v.record() for v in tails],half_normal_multiplier=multiplier.record())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cells',type=int,default=1024);args=p.parse_args();assert 8<=args.cells<=32768
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic()
    receipt=read(root/'revisions/R28_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
    names=['revisions/r28_conditional_moments_v2/RESULT.json','tools/certify_weighted_chisquare.py']
    for name in names:assert sha(root/name)==manifest['files'][name]
    old=read(root/'revisions/R21_HISTORY.json');old_manifest=read(root/old['path']/'manifest.json')
    assert sha(root/'tools/certify_sphere_threshold_cdf.py')==old_manifest['files']['tools/certify_sphere_threshold_cdf.py']
    prior=read(root/names[0]);n=prior['n'];d=prior['q']//2;assert (d,n)==(512,16595)
    (a,q),details=integrate(args.cells,12);assert q.lo>0
    efficiency=a/q.sqrt();center=I(d)*efficiency/I(n).sqrt();mu=restore(prior['C0_moments']['mean_D']);shift=mu-center
    gaussian_proxy_bound=(-shift)/(2*integer.pi_interval()).sqrt() if shift.hi<0 else None
    archived=read(root/'ac2_checks/full/AC2_design.json')['methods']['Adam2']
    result=dict(a=a.record(),q=q.record(),c=efficiency.record(),theory_center=center.record(),C0_center=mu.record(),
                C0_minus_theory_center=shift.record(),C0_center_strictly_below_theory=shift.hi<0,
                auxiliary_unit_Gaussian_center_effect_absolute_upper=(str(F(gaussian_proxy_bound.hi,S)) if gaussian_proxy_bound else None),
                Gaussian_proxy_bound_not_actual_recall_bound=True,
                archived_float_c=str(F(archived['c'])),archived_float_c_contained=efficiency.contains(F(archived['c'])),
                archived_float_a_contained=a.contains(F(archived['a'])),archived_float_q_contained=q.contains(F(archived['q'])),
                parameters=dict(alpha='3/10',beta1='9/10',beta2='99/100',stabilizer='1',steps=2),
                d=d,n=n,cells=args.cells,radius=12,source_hashes={name:sha(root/name) for name in names+['ac2_checks/full/AC2_design.json','tools/certify_sphere_threshold_cdf.py']},
                new_samples=0,statistical_confidence_interval=False,independent_review=False,
                population_bias_not_estimated=True,recall_difference_not_certified=True,
                scope='Exact rational scalar Adam2 c_T compared with fixed represented C0 conditional center; inherits R28 moment assumptions',wall_seconds=time.monotonic()-start)
    save(args.out/'QUADRATURE.json',details);save(args.out/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ['a','q','c','theory_center','C0_minus_theory_center','wall_seconds']},indent=2))


if __name__=='__main__':main()
