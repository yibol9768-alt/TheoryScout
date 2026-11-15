"""Check raw integral envelopes and all root signs using mpmath.iv.

No generator imports. Ordinary quadrature values below are diagnostics only.
"""
import argparse
from fractions import Fraction
import hashlib,json,re
from pathlib import Path
from mpmath import iv
from scipy.integrate import quad
import math
iv.dps=60
BASE=Path(__file__).resolve().parents[1]
def decode(x):
    lo=float.fromhex(x['lower_hex']);hi=float.fromhex(x['upper_hex'])
    assert lo==x['lower'] and hi==x['upper'] and lo<=hi
    return iv.mpf([lo,hi])
def overlap(x,y):assert x.a<=y.b and y.a<=x.b,(str(x),str(y))
def positive(x):assert x.a>0,str(x)
def main():
    p=argparse.ArgumentParser();p.add_argument('--certificate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cert=json.loads(a.certificate.read_text())
    for name,h in cert['source_hashes'].items():assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==h,name
    assert cert['grid']==4096
    integral={(r['e_numerator'],r['e_denominator']):r for r in cert['raw_efficiency_integrals']}
    root_count=0
    printed=re.findall(r'(16|32|64|128|256) & ([012]) & .*?\$\[([0-9.]+),([0-9.]+)\]\$ & \$\[', (BASE/'statements/polar_variance.tex').read_text())
    table={(int(g),int(t)):(Fraction(lo),Fraction(hi)) for g,t,lo,hi in printed}
    for row in cert['rows']:
        gamma=row['gamma'];mu=decode(row['mu']);k=decode(row['k']);nu=decode(row['nu'])
        overlap(nu,1+k/gamma-mu*mu/(gamma*gamma))
        batches=[r for r in cert['raw_kernel_batches'] if r['gamma']==gamma]
        assert len(batches)==128 and batches[0]['start']==0 and batches[-1]['stop']==4096
        assert all(x['stop']==y['start'] for x,y in zip(batches,batches[1:]))
        total=sum((decode(r['integral']) for r in batches),iv.mpf(0))
        overlap(k,4*total/gamma/(iv.pi*iv.pi));positive(k)
        # A second integral expression for the mean, used only as a float check.
        mean=4/math.pi*quad(lambda u:(1-u*u)*math.sqrt(1-u*u/gamma),0,1,epsabs=1e-13)[0]
        assert row['mu']['lower']<=mean<=row['mu']['upper']
        for crossing in row['crossings']:
            tau=crossing['tau']
            target=mu*iv.sqrt((1+tau*tau)/(nu+tau*tau-2*mu*mu/gamma))
            for key in ['e_certificate','analytic_lower_certificate','analytic_upper_certificate']:
                root=crossing[key];lo=root['lower_numerator'];hi=root['upper_numerator'];den=root['denominator'];assert lo<hi
                saved_target=decode(root['target'])
                if key=='e_certificate':overlap(saved_target,target)
                for n,sgn in [(lo,-1),(hi,1)]:
                    raw=integral[n,den];c=decode(raw['A'])/iv.sqrt(decode(raw['Q']))
                    positive(saved_target-c if sgn<0 else c-saved_target)
                if key=='e_certificate':
                    assert (gamma,tau) in table
                    lower,upper=table[gamma,tau]
                    assert Fraction(lo,den)>=lower and Fraction(hi,den)<=upper
                root_count+=1
    result=dict(status='PASS',gamma_count=5,MSR_crossings=15,root_brackets_including_variance_envelope=root_count,
                strict_endpoint_inequalities=2*root_count,paper_crossing_table_enclosed=True,
                raw_integral_envelopes_checked=True,arithmetic='mpmath.iv 60 decimal digits',
                diagnostic_quadrature_is_not_a_proof=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
