"""Exact polynomial sphere references, mixed traces and perturbation tests."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json

from certify_conditional_moments import I,S,moments,exact_basic_traces


def mm(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def tr(a):return sum(a[j][j] for j in range(len(a)))
def power(a,k):
    r=[[F(i==j) for j in range(len(a))] for i in range(len(a))]
    for _ in range(k):r=mm(r,a)
    return r
def traces(b):
    q=len(b);k=[[(b[i][j]+b[j][i])/2 for j in range(q)] for i in range(q)];nt=mm(list(map(list,zip(*b))),b)
    return dict(mu=tr(k),t2=tr(power(k,2)),t3=tr(power(k,3)),t4=tr(power(k,4)),trN=tr(nt),trN2=tr(power(nt,2)),
                trNK=tr(mm(nt,k)),trNK2=tr(mm(nt,power(k,2)))),k,nt
def padd(a,b,scale=F(1)):
    out=a.copy()
    for e,v in b.items():out[e]=out.get(e,F(0))+scale*v
    return {e:v for e,v in out.items() if v}
def pmul(a,b):
    out={}
    for e,x in a.items():
        for f,y in b.items():
            g=tuple(i+j for i,j in zip(e,f));out[g]=out.get(g,F(0))+x*y
    return out
def ppow(a,k):
    out={(0,0):F(1)}
    for _ in range(k):out=pmul(out,a)
    return out
def pscale(a,c):return {e:v*c for e,v in a.items()}
def quadratic(a,scale):
    out={}
    for i in range(2):
        for j in range(2):
            e=[0,0];e[i]+=1;e[j]+=1;e=tuple(e);out[e]=out.get(e,F(0))+scale*a[i][j]
    return out
def integrate(poly,n):
    total=F(0)
    for exponent,c in poly.items():
        if any(v%2 for v in exponent):continue
        degree=sum(exponent)//2;num=den=1
        for e in exponent:
            for k in range(1,e,2):num*=k
        for k in range(degree):den*=n+2*k
        total+=c*F(num,den)
    return total


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    matrices=[[[F(1,2),F(-1,4)],[F(3,4),F(1,8)]],[[F(1,4),F(1,2)],[F(-3,4),F(1,4)]],
              [[F(0),F(1)],[F(-1),F(0)]],[[F(1,2),F(0)],[F(0),F(1,2)]]]
    moment_checks=mixed_checks=perturb_checks=basic_checks=0
    for b in matrices:
        t,k,nt=traces(b);flat=[float(v) for row in b for v in row];mu,nn,t2=exact_basic_traces(flat,2)
        assert (mu,nn,t2)==(t['mu'],t['trN'],t['t2']);basic_checks+=3
        for n in [2,3,5]:
            m=n-1;D=quadratic(k,F(n));W=quadratic(nt,F(n*n,m));V=padd(W,ppow(D,2),-F(1,m))
            md=integrate(D,n);mv=integrate(V,n);mw=integrate(W,n)
            centeredD=padd(D,{(0,0):md},-1);centeredV=padd(V,{(0,0):mv},-1);centeredW=padd(W,{(0,0):mw},-1)
            exact=dict(mean_D=md,variance_D=integrate(ppow(centeredD,2),n),third_central_D=integrate(ppow(centeredD,3),n),
                fourth_central_D=integrate(ppow(centeredD,4),n),mean_V=mv,variance_V=integrate(ppow(centeredV,2),n),
                mean_W=mw,variance_W=integrate(ppow(centeredW,2),n),covariance_DV=integrate(pmul(centeredD,centeredV),n))
            enclosed=moments({key:I(value) for key,value in t.items()},n,include_derived=False)
            for key,value in exact.items():assert enclosed[key].contains(value),(n,key,value,enclosed[key].record());moment_checks+=1
            pairs=[]
            for alpha in [F(1,3),F(2)]:
                anchor=F(3,2);A=4*alpha*m+8*alpha*alpha*anchor;B=-4*alpha*alpha*n
                h=[[A*k[i][j]+B*nt[i][j] for j in range(2)] for i in range(2)]
                c=(tr(power(h,2))-A*A*t['t2']-B*B*t['trN2'])/(2*A*B);assert c==t['trNK'];mixed_checks+=1
                rhs=(tr(power(h,3))-A**3*t['t3']-B**3*tr(power(nt,3)))/(3*A*B);pairs.append((A,B,rhs))
            A1,B1,r1=pairs[0];A2,B2,r2=pairs[1];det=A1*B2-A2*B1
            assert (r1*B2-r2*B1)/det==t['trNK2'];mixed_checks+=1
            assert (A1*r2-A2*r1)/det==tr(mm(k,power(nt,2)));mixed_checks+=1
        # Rank-one perturbations have exact operator error eta=epsilon.
        khat=max(sum(abs(v) for v in row) for row in k);nhat=max(sum(abs(v) for v in row) for row in nt);beta=max(F(1),nhat)
        for eta in [F(1,16),F(1,1024)]:
            for i in range(2):
                for j in range(2):
                    new=[row.copy() for row in b];new[i][j]+=eta;actual,_,_=traces(new)
                    kap=khat+eta;nu=eta*(2*beta+eta);rho=nhat+nu
                    deltaF=I(2).sqrt()*I(eta)
                    deltaN=min(2*nu,F((2*I(t['trN']).sqrt()*deltaF+deltaF.square()).hi,S))
                    deltaT2=min(4*kap*eta,F((2*I(t['t2']).sqrt()*deltaF+deltaF.square()).hi,S))
                    errors=dict(mu=2*eta,t2=deltaT2,t3=6*kap**2*eta,t4=8*kap**3*eta,trN=deltaN,
                        trN2=4*rho*nu,trNK=2*(nu*kap+nhat*eta),trNK2=2*(nu*kap**2+2*nhat*kap*eta))
                    for key,bound in errors.items():assert abs(actual[key]-t[key])<=bound,(key,actual[key],t[key],bound);perturb_checks+=1
    result=dict(exact_basic_trace_checks=basic_checks,exact_sphere_polynomial_moment_checks=moment_checks,
        exact_mixed_trace_reconstructions=mixed_checks,exact_rank_one_perturbation_checks=perturb_checks,
        ambient_dimensions=[2,3,5],noncommuting_and_degenerate_cases=True,checks_not_independent_mathematical_review=True)
    with args.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
