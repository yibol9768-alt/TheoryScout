"""Independent 90-digit diagnostic of constants, exact rational margins, and applicability.
mpmath is a diagnostic cross-check; the calculator's interval output is the certificate.
"""
import argparse, json, time
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
import AB1_decimal_backend as backend
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'ab1_checks';mp.mp.dps=90
M=lambda x:mp.mpf(str(x))
def load(name):return json.loads((BASE/name).read_text())
def atom(x):return mp.mpf(F(x).numerator)/F(x).denominator
def phi(z):return mp.exp(-z*z/2)/mp.sqrt(2*mp.pi)
def quant(p):return mp.sqrt(2)*mp.erfinv(2*p-1)
def qtrue(par):
    T=par['T'];alpha=atom(par['alpha']);e=atom(par['e']);b1=atom(par['beta1']);b2=atom(par['beta2'])
    if par['method']=='gd':return (1-(1-atom(par['eta']))**T)**2
    def response(z):
        h=m=v=mp.mpf(0)
        for t in range(1,T+1):
            g=h-z
            if par['method']=='adam':
                m=b1*m+(1-b1)*g;v=b2*v+(1-b2)*g*g
                h-=alpha*(m/(1-b1**t))/(mp.sqrt(v/(1-b2**t))+e)
            else:h-=alpha*mp.sign(g)
        return h
    edges=sorted(set([mp.mpf(0),mp.mpf(1),mp.mpf(4),mp.mpf(12)]+[alpha*k for k in range(T+1)]))
    edges.append(mp.inf)
    return mp.quad(lambda z:2*phi(z)*response(z)**2,edges)
def budget(data,d):
    c={k:M(v[1]) for k,v in data['constants'].items()};par=data['parameters'];kap=atom(par['kappa']);rho=atom(par['rho']);r=atom(data['constants']['r'][0]);n=int(mp.ceil(kap*d*d));m=n-1
    # Use lower q,b,rstar and upper other constants in the explicitly monotone formula.
    q=M(data['constants']['q_lower_used'][0]);b0=M(data['constants']['b_star'][0]);rs=M(data['constants']['r_star'][0])
    xi=mp.mpf(d)**(-r);xc=2*d*mp.log(9)+mp.log(2/xi);delta=4*mp.sqrt(xc/n)+4*xc/n
    ua=mp.sqrt(c['v_a'])/(d*mp.sqrt(xi));uq=mp.sqrt(c['v_q'])/(d*mp.sqrt(xi));b=mp.sqrt(q-uq);vd=2*c['K']*mp.sqrt(2*xc)/d
    x=mp.log(4/xi);theta=mp.ceil(rho*n)/m;omega=mp.sqrt(x/(2*m));zm=quant(1-theta-omega)/mp.sqrt(1+2*mp.sqrt(x/m)+2*x/m);zp=quant(1-theta+omega)/mp.sqrt(1-2*mp.sqrt(x/m));z=quant(1-rho);h=max(abs(zm-z),abs(zp-z))
    Z=c['Z_star'];Mbig=c['M'];D4=c['D4'];N=max(128,(32*Z)**2,8*Z*(Mbig*Mbig+2),8*mp.sqrt(Z*D4))
    checks=dict(scatter=delta<=mp.mpf(1)/8,q_moment=uq<=q/2,stable_rank=vd<=b*rs/4,n_base=n>=N,n_gt_2d=n>2*d,
                threshold=zm>0 and zp<=Z and h<=min(z/2,1),b_star=b>=b0)
    assert all(checks.values()),checks
    L=c['f']*delta**(2*r);kapd=mp.mpf(n)/(d*d)
    E=5*xi+mp.mpf(2)/(9*d*d)+h+4*L/(b*rs)+(4*L/b+2*delta+ua/b+uq/(b*b))/mp.sqrt(kapd)+12*(1+Z)*delta+2*c['K_R']/b*(vd+L)+c['K_n']/mp.sqrt(n)+12*mp.sqrt(Z*D4)/n
    upper=c['C_star']*mp.mpf(d)**(-r)
    assert E<=upper,(E,upper)
    return dict(d=str(d),checks=checks,finite_budget=mp.nstr(E,25),rate_budget=mp.nstr(upper,25))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output');args=parser.parse_args()
    start=time.monotonic();backend.resources();results=[]
    for method in ['gd','sign','adam']:
        file=f'AB1_q_{method}_example.json';data=load(file);q=qtrue(data['parameters']);ql,qh=map(M,data['constants']['q_integral_enclosure'])
        assert ql<=q<=qh,(method,ql,q,qh)
        terms=data['coefficient_terms'];Clo,Chi=map(M,data['constants']['C_star'])
        assert sum(M(v[0]) for v in terms.values())<=Chi and sum(M(v[1]) for v in terms.values())>=Clo
        rows=[budget(data,int(data['sufficient_dimension'])),budget(data,int(data['sufficient_dimension_for_error_below_one']))]
        results.append(dict(method=method,q_independent=mp.nstr(q,35),q_interval_contains=True,budgets=rows))
        print(method,'PASS',rows,flush=True)
    # Exact margin audits; do not turn random simulations into proofs.
    assert 8*F(9,8)**4<13
    assert 12*F(41,64)**2+48*F(41,64)**2<60
    assert F(1,4)+F(1,16)+F(1,4)<1
    assert mp.sqrt(3)/2-mp.sqrt(2)/16-mp.mpf(1)/8>mp.mpf('.5')
    result=dict(status='PASS',methods=results,seconds=time.monotonic()-start,resources=backend.resources())
    if args.output:
        out=(ROOT/args.output).resolve();assert out.is_relative_to(BASE)
        with out.open('x') as f:json.dump(result,f,indent=2)
    else:print(json.dumps(result,indent=2))
if __name__=='__main__':main()
