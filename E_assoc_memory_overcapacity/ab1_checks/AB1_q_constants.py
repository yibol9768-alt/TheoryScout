#!/usr/bin/env python3
"""Independent Q constant calculator. Certified q lower integral; no AA1 imports.
Five positional parameters T,e,kappa,beta1,beta2; alpha=.6,rho=.1 defaults.
All output numbers are outward Decimal interval endpoints (48 digits).
"""
import argparse, hashlib, json, time
from pathlib import Path
from fractions import Fraction as F
import AB1_decimal_backend as iv
I=iv.I;D=iv.D
ROOT=Path(__file__).resolve().parents[1]
def sq(x):
    return I.pair(D(0) if x.l<=0<=x.h else min(iv.DOWN.multiply(x.l,x.l),iv.DOWN.multiply(x.h,x.h)),
                  max(iv.UP.multiply(x.l,x.l),iv.UP.multiply(x.h,x.h)))
def power(x,r):
    r=F(r)
    if r.denominator==1:
        out=I(1)
        for _ in range(abs(r.numerator)): out=out*x
        return out if r>=0 else 1/out
    assert x.l>0
    # exp(r*log(x)); Decimal ln correctly rounded, padded by adjacent values.
    lo=iv.NEAR.next_minus(iv.NEAR.ln(x.l));hi=iv.NEAR.next_plus(iv.NEAR.ln(x.h))
    return (I.pair(lo,hi)*r).exp()
def ln(x):
    return I.pair(iv.NEAR.next_minus(iv.NEAR.ln(I(x).l)),iv.NEAR.next_plus(iv.NEAR.ln(I(x).h)))
def density(z):return iv.NORMAL*(-sq(z)/2).exp()
def cdf_positive(z):
    # Integral [0,z] using composite midpoint Taylor with a uniform |phi''|<=1.
    # Absolute remainder <= z*h^2/24. Outward arithmetic, not a float CDF.
    n=2048;h=I(z)/n;total=I(0)
    for k in range(n):total=total+density(h*F(2*k+1,2))/2*h
    err=I(z)*sq(h)/24
    return I(F(1,2))+total+I.pair(-err.h,err.h)
def quantile(p):
    l,h=F(0),F(12)
    assert I(p).l>D('.5') and cdf_positive(h).l>I(p).h, 'quantile outside certified [0,12] bracket'
    # Enough for constants; keep both endpoints when the CDF enclosure straddles.
    for _ in range(24):
        mid=(l+h)/2;v=cdf_positive(mid)
        if v.h<I(p).l:l=mid
        elif v.l>I(p).h:h=mid
        else:break
    return I(l,h)
def q_enclosure(T,alpha,e,b1,b2,K,method,cells):
    edges={F(12*k,cells) for k in range(cells+1)}
    if method=='sign':edges.update(k*alpha for k in range(T+1) if 0<=k*alpha<=12)
    edges=sorted(edges);total=I(0)
    for l,h in zip(edges,edges[1:]):
        z=I(l,h);state=I(0);m=I(0);v=I(0)
        for t in range(1,T+1):
            residual=state-z
            if method=='adam':
                m=I(b1)*m+(1-b1)*residual;v=I(b2)*v+(1-b2)*sq(residual)
                state=state-alpha*(m/(1-b1**t))/( (v/(1-b2**t)).sqrt()+e )
            else:
                sign=I(1) if residual.l>0 else I(-1) if residual.h<0 else I(-1,1)
                state=state-alpha*sign
        total=total+sq(state)*density(z)*(h-l)
    tail=K*K*density(I(12))/12
    return I.pair(max(D(0),total.l),iv.UP.add(total.h,tail.h))
def calc(args):
    T=args.T;alpha=F(args.alpha);e=F(args.e);kap=F(args.kappa);b1=F(args.beta1);b2=F(args.beta2);rho=F(args.rho)
    assert T>=1 and alpha>0 and e>0 and kap>0 and 0<=b1<1 and 0<=b2<1 and 0<rho<F(1,2)
    method=args.method;r=F(1,6) if method=='sign' else F(1,2);alphaI=I(alpha)
    params=dict(T=T,e=str(e),kappa=str(kap),beta1=str(b1),beta2=str(b2),alpha=str(alpha),rho=str(rho),eta=args.eta,method=method)
    c={};steps=[]
    if method=='gd':
        eta=F(args.eta);k=1-(1-eta)**T;assert eta>0 and k>0
        K=I(k);va=2*sq(K);vq=2*sq(sq(K));q=sq(K)
        cg=I(eta**2*sum(j*(1+3*eta/2)**(j-1) for j in range(1,T)));f=2*(2*cg+k)
        c.update(k=K,C_g=cg)
    else:
        if method=='adam':
            lam=I(1+2*alpha/e);c0=I(1-b2).sqrt();H=[I(0)];As=[I(0)]
            for t in range(1,T+1):
                ht=I(e)*(power(lam,t)-1)/(2*c0);jt=power(lam,t-1)/c0;lt=(1+jt)/e
                at=As[-1]+alpha*lt*sum((sq(2*As[j-1]+H[j-1]+2) for j in range(1,t+1)),I(0)).sqrt()
                H.append(ht);As.append(at);steps.append(dict(t=t,H=ht.encode(),J=jt.encode(),L=lt.encode(),A=at.encode()))
            K=H[-1];f=As[-1];c.update(lambda_adam=lam,c0_adam=c0)
        else:
            K=I(T*alpha);A=I(2*T*T)/(2*iv.PI).sqrt();B=I(sum((t*alpha+2)**2 for t in range(T)))
            f=2*T*alpha*(A*power(2*B/A,F(1,3))+B*power(A/(2*B),F(2,3))).sqrt();c.update(A_sign=A,B_sign=B)
        va=sq(K);vq=sq(sq(K));q=q_enclosure(T,alpha,e,b1,b2,K,method,args.q_cells)
    assert q.l>0,'cannot certify q>0; increase --q-cells or supply a simpler fixed parameter case'
    # Conservative monotone substitution q_low<=q; K,v_a,v_q,f retain upper endpoints.
    ql=I(q.l);K=I(K.h);va=I(va.h);vq=I(vq.h);f=I(f.h)
    z=quantile(1-rho);s=min(rho/2,(F(1,2)-rho)/2);zouter=quantile(1-rho+s)
    Drho=1/(density(zouter)/2);Hrho=5*I(2).sqrt()*Drho+4*I(2).sqrt()*z
    Z=z+1;b=(ql/2).sqrt();M=3/I(kap).sqrt();D4=8*power(M,4)+480
    # min/max of enclosed constants must include both endpoint extrema.
    def vmax(xs):return I.pair(max(x.l for x in xs),max(x.h for x in xs))
    def vmin(xs):return I.pair(min(x.l for x in xs),min(x.h for x in xs))
    rstar=vmin([1/(64*Z),1/(4*(52*Z).sqrt())]);KR=12*(1+Z)+3*(52*Z).sqrt();Kn=12*(1+Z)+3*(2*Z*(sq(M)+2)).sqrt()
    Abar=2*ln(9)+ln(2)+r;Dbar=4*(Abar/kap).sqrt()+4*Abar/kap;Vbar=2*K*(2*Abar).sqrt();Qbar=ln(4)+r
    Nbase=vmax([I(128),sq(32*Z),8*Z*(sq(M)+2),8*(Z*D4).sqrt()]);Fbar=f*power(Dbar,2*r)
    dims={'integer_min':I(2),'n_gt_2d':I(3/kap),'scatter':sq(8*Dbar),'stable_rank':sq(4*Vbar/(b*rstar)),
          'q_moment':power(2*vq.sqrt()/ql,1/(1-r/2)),'n_base':(Nbase/kap).sqrt(),
          'quantile_legal':25*Qbar/(kap*s*s),'radius_legal':16*Qbar/kap,
          'threshold_close':Qbar*sq(Hrho)/(I(kap)*sq(vmin([z/2,I(1)])))}
    coeff={'event_probabilities':I(5),'gaussian_frobenius':I(F(2,9)),'trajectory_cutoff':4*Fbar/(b*rstar),
           'center_trajectory':4*Fbar/(b*I(kap).sqrt()),'center_scatter':2*Dbar/I(kap).sqrt(),
           'center_a_moment':va.sqrt()/(b*I(kap).sqrt()),'center_q_moment':vq.sqrt()/(sq(b)*I(kap).sqrt()),
           'scatter_rank':12*(1+Z)*Dbar,'stable_rank_transfer':2*KR/b*(Vbar+Fbar),
           'sphere_n_half':Kn/I(kap).sqrt(),'sphere_n':12*(Z*D4).sqrt()/kap,'threshold':Hrho*(Qbar/kap).sqrt()}
    C=sum(coeff.values(),I(0));dstar=vmax(list(dims.values()));Chat=C+1/(2*power(I(kap),F(3,2)))
    c.update(K=K,v_a=va,v_q=vq,q_integral_enclosure=q,q_lower_used=ql,f=f,r=I(r),z=z,s_rho=I(s),D_rho=Drho,H_rho=Hrho,
             b_star=b,Z_star=Z,M=M,D4=D4,r_star=rstar,K_R=KR,K_n=Kn,Abar=Abar,Dbar=Dbar,Vbar=Vbar,Qbar=Qbar,Nbase=Nbase,Fbar=Fbar,C_star=C,C_star_fixed=Chat,d_star=dstar)
    upper_int=lambda x:int(x.h.to_integral_value(rounding=iv.ROUND_CEILING))
    usable=upper_int(vmax([dstar,power(C,1/r)]))
    return dict(parameters=params,constants={k:v.encode() for k,v in c.items()},adam_steps=steps,
                dimension_terms={k:v.encode() for k,v in dims.items()},coefficient_terms={k:v.encode() for k,v in coeff.items()},
                sufficient_dimension=upper_int(dstar),sufficient_dimension_for_error_below_one=usable+1,
                q_method='exact GD' if method=='gd' else 'outward interval lower/upper Darboux integral on [0,12], K^2 Mills tail',
                note='q_lower substitution makes constants conservative; all constants are intervals, use upper dimension/coefficient endpoints')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,typ in [('T',int),('e',str),('kappa',str),('beta1',str),('beta2',str)]:p.add_argument(name,type=typ)
    p.add_argument('--method',choices=['adam','gd','sign'],default='adam');p.add_argument('--alpha',default='.6');p.add_argument('--rho',default='.1');p.add_argument('--eta',default='.5');p.add_argument('--q-cells',type=int,default=4096);p.add_argument('--output')
    args=p.parse_args();start=time.monotonic();iv.resources();result=calc(args)
    result.update(status='PASS',seconds=time.monotonic()-start,resources=iv.resources(),paper_Q_sha256=hashlib.sha256((ROOT/'paper/appendix/AA_quant_rank.tex').read_bytes()).hexdigest())
    text=json.dumps(result,indent=2)
    if args.output:
        out=(ROOT/args.output).resolve();assert out.is_relative_to(ROOT/'ab1_checks')
        with out.open('x') as f:f.write(text+'\n')
    print(text)
if __name__=='__main__':main()
