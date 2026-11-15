"""Certify represented-C0 score/energy moments from exact traces and eta.

No new matrices, probability integrals or random data. Two anchored spectra
recover mixed traces without assuming K and N commute.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse
import hashlib
import json
import time

import certify_weighted_chisquare as integer
from check_anchor_certificate import construction_arrays

I=integer.I
S=integer.SCALE


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def intersect(a,b):return I.raw(max(a.lo,b.lo),min(a.hi,b.hi))
def pad(value,radius):
    value=I(value);r=I(radius).hi;return I.raw(value.lo-r,value.hi+r)
def upper(value):return F(value.hi,S)


def exact_basic_traces(b,q):
    exponent=max(v.as_integer_ratio()[1].bit_length()-1 for v in b);scale=1<<exponent
    values=[]
    for v in b:
        p,d=v.as_integer_ratio();values.append(p << (exponent-(d.bit_length()-1)))
    mu=F(sum(values[i*q+i] for i in range(q)),scale)
    nn=F(sum(v*v for v in values),scale*scale)
    diag=sum(values[i*q+i]**2 for i in range(q))
    off=sum((values[i*q+j]+values[j*q+i])**2 for i in range(q) for j in range(i+1,q))
    t2=F(2*diag+off,2*scale*scale)
    return mu,nn,t2


def spectral_traces(directory,factor=F(1),powers=4):
    spectrum=read(directory/'actual_spectrum.json');epsilon=F(read(directory/'ERRORS.json')['total']);r=I(epsilon).hi
    eigen=[(I(F(v))+I.raw(-r,r))*I(factor) for v in spectrum['eigenvalues']]
    traces=[sum(v**k for v in eigen) for k in range(1,powers+1)]
    norm=F(max(max(abs(v.lo),abs(v.hi)) for v in eigen),S)
    return traces,norm


def moments(t,n):
    m=n-1;mu=t['mu'];t2=t['t2'];t3=t['t3'];t4=t['t4'];a=t['trN'];b=t['trN2'];c=t['trNK'];e=t['trNK2']
    u2=t2-mu.square()/n
    u3=t3-3*mu*t2/n+2*mu**3/(n*n)
    u4=t4-4*mu*t3/n+6*mu.square()*t2/(n*n)-3*mu**4/(n**3)
    varD=I(F(2*n,n+2))*u2
    third=I(F(8*n*n,(n+2)*(n+4)))*u3
    fourth=I(F(n**3,(n+2)*(n+4)*(n+6)))*(12*u2.square()+48*u4)
    d2=intersect(mu.square()+varD,I(F(n,n+2))*(mu.square()+2*t2))
    meanW=I(F(n,m))*a;meanV=meanW-d2/m
    varW=I(F(n*n,m*m)*F(2*n,n+2))*(b-a.square()/n)
    varD2=fourth+4*mu*third+4*mu.square()*varD-varD.square()
    covWD2=I(F(4*n*n,m*(n+2)*(n+4)))*(I(n)*(mu*c+2*e)-a*(mu.square()+2*t2))
    varV=varW+varD2/(m*m)-2*covWD2/m
    covDV=I(F(2*n*n,m*(n+2)))*(c-mu*a/n)-(third+2*mu*varD)/m
    # A separate raw-moment arrangement checks the dimension factors.
    qN2=(a.square()+2*b)/(n*(n+2))
    qNK2=(a*(mu.square()+2*t2)+4*mu*c+8*e)/(n*(n+2)*(n+4))
    qK4=(mu**4+12*mu.square()*t2+12*t2.square()+32*mu*t3+48*t4)/(n*(n+2)*(n+4)*(n+6))
    varVraw=I(F(n**4,m*m))*(qN2-2*qNK2+qK4)-meanV.square()
    qKN=(mu*a+2*c)/(n*(n+2));qK3=(mu**3+6*mu*t2+8*t3)/(n*(n+2)*(n+4))
    covDVraw=I(F(n**3,m))*(qKN-qK3)-mu*meanV
    varV=intersect(varV,varVraw);covDV=intersect(covDV,covDVraw)
    assert varD.lo>0 and varV.lo>0
    return dict(mean_D=mu,variance_D=varD,third_central_D=third,fourth_central_D=fourth,
                mean_W=meanW,mean_V=meanV,variance_W=varW,variance_V=varV,covariance_DV=covDV,
                correlation_DV=covDV/(varD*varV).sqrt(),RMS_wrong_energy=meanV.sqrt())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic();protected=[]
    groups=[('R13',['revisions/r11_anchor_certificate_v2/actual_construction.bin','revisions/r11_anchor_certificate_v2/SCOPE.json',
                   'revisions/r11_anchor_certificate_v2/ERRORS.json','revisions/r11_anchor_certificate_v2/actual_spectrum.json',
                   'revisions/r13_scatter_checks/CHECKS.json','tools/certify_weighted_chisquare.py']),
            ('R25',['revisions/r25_second_anchor/SCOPE.json','revisions/r25_second_anchor/ERRORS.json','revisions/r25_second_anchor/actual_spectrum.json']),
            ('R27',['revisions/r27_bad_matrices/SCOPE.json','revisions/r27_bad_matrices/D/ERRORS.json',
                    'revisions/r27_bad_matrices/D/actual_spectrum.json','revisions/r27_bad_matrices/W/ERRORS.json','revisions/r27_bad_matrices/W/actual_spectrum.json'])]
    for revision,names in groups:
        receipt=read(root/f'revisions/{revision}_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
        for name in names:assert sha(root/name)==manifest['files'][name];protected.append(name)
    q,b,_,_=construction_arrays(root/'revisions/r11_anchor_certificate_v2/actual_construction.bin')
    scope=read(root/'revisions/r11_anchor_certificate_v2/SCOPE.json');n=scope['n'];m=n-1;assert q==scope['block_dimension']
    mu,nn,t2=exact_basic_traces(b,q)
    K,khat=spectral_traces(root/'revisions/r27_bad_matrices/D')
    N,nhat=spectral_traces(root/'revisions/r27_bad_matrices/W',F(m,n),3)
    assert K[0].contains(mu) and K[1].contains(t2) and N[0].contains(nn)
    assert read(root/'revisions/r27_bad_matrices/SCOPE.json')['source_B_sha256']==sha(root/'revisions/r11_anchor_certificate_v2/actual_construction.bin')
    families=[]
    for directory in [root/'revisions/r11_anchor_certificate_v2',root/'revisions/r25_second_anchor']:
        sc=read(directory/'SCOPE.json');assert (sc['n'],sc['block_dimension'],sc['center_anchor'])==(n,q,scope['center_anchor'])
        alpha=F(sc['alpha']);anchor=F(sc['center_anchor']);A=4*alpha*m+8*alpha*alpha*anchor;B=-4*alpha*alpha*n
        H,_=spectral_traces(directory,powers=3);assert H[0].contains(A*mu+B*nn)
        mixed=(H[1]-I(A*A*t2)-I(B*B)*N[1])/I(2*A*B)
        rhs=(H[2]-I(A**3)*K[2]-I(B**3)*N[2])/I(3*A*B)
        families.append((alpha,A,B,mixed,rhs))
    alpha1,A1,B1,c1,r1=families[0];alpha2,A2,B2,c2,r2=families[1]
    determinant=A1*B2-A2*B1;assert determinant==16*n*m*alpha1*alpha2*(alpha1-alpha2)!=0
    mixed=intersect(c1,c2);e=(r1*I(B2)-r2*I(B1))/I(determinant);f=(I(A1)*r2-I(A2)*r1)/I(determinant)
    hat=dict(mu=I(mu),t2=I(t2),t3=K[2],t4=K[3],trN=I(nn),trN2=N[1],trNK=mixed,trNK2=e)
    eta=F(read(root/'revisions/r13_scatter_checks/CHECKS.json')['eta_upper'])
    beta=upper(I(nhat).sqrt());kappa=khat+eta;nu=upper(I(eta)*(2*I(beta)+I(eta)));rho=nhat+nu
    deltaF=I(q).sqrt()*I(eta)
    deltaNtrace=min(q*nu,upper(2*I(nn).sqrt()*deltaF+deltaF.square()))
    deltaT2=min(2*q*kappa*eta,upper(2*I(t2).sqrt()*deltaF+deltaF.square()))
    errors=dict(mu=q*eta,t2=deltaT2,t3=3*q*kappa*kappa*eta,t4=4*q*kappa**3*eta,
        trN=deltaNtrace,trN2=2*q*rho*nu,trNK=q*(nu*kappa+nhat*eta),
        trNK2=q*(nu*kappa*kappa+2*nhat*kappa*eta))
    exactC0={name:pad(value,errors[name]) for name,value in hat.items()}
    represented_moments=moments(hat,n);C0_moments=moments(exactC0,n)
    anchor_error=exactC0['mu']-I(F(scope['center_anchor']))
    source=dict(source_hashes={name:sha(root/name) for name in protected},n=n,q=q,eta=str(eta),
        common_K_norm=str(kappa),Bhat_operator_norm_upper=str(beta),Nhat_operator_norm_upper=str(nhat),
        N_operator_perturbation=str(nu),trace_perturbation_bounds={k:str(v) for k,v in errors.items()})
    save(args.out/'INPUTS.json',source)
    save(args.out/'TRACES.json',dict(exact_Bhat_basic=dict(mu=str(mu),trN=str(nn),t2=str(t2)),
        Bhat={k:v.record() for k,v in hat.items()},C0={k:v.record() for k,v in exactC0.items()},
        mixed_auxiliary_trKN2=f.record(),mixed_trace_determinant=str(determinant),no_commutation_assumption=True))
    result=dict(n=n,q=q,Bhat_moments={k:v.record() for k,v in represented_moments.items()},
        C0_moments={k:v.record() for k,v in C0_moments.items()},center_anchor_error=anchor_error.record(),
        mean_V_strictly_above_one=C0_moments['mean_V'].lo>S,covariance_DV_strictly_positive=C0_moments['covariance_DV'].lo>0,
        source_hashes=source['source_hashes'],new_samples=0,new_matrices=0,new_probability_integrals=0,
        ambient_dimension_not_block_dimension=True,raw_and_stable_moment_forms_intersected=True,
        independent_review=False,does_not_determine_recall=True,wall_seconds=time.monotonic()-start,
        scope='Exact represented C0 ideal Adam2 conditional score/energy moments; inherits R13 eta and spectral numeric assumptions')
    save(args.out/'RESULT.json',result)
    print(json.dumps(dict(C0={k:v.record()['approximate'] for k,v in C0_moments.items()},
                         anchor_error=anchor_error.record()['approximate'],wall_seconds=result['wall_seconds']),indent=2))


if __name__=='__main__':main()
