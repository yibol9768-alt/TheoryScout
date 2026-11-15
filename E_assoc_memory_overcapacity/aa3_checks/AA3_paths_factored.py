"""Remove the COMMON positive p_j^2 before interval integration.
This avoids interval dependency from a factor that cannot affect the sign.
"""
import AA3_paths as paths
from AA3_certify import *
KIND=1

def integrands(z0,n,a,e,b1,b2):
    C=lambda x:Jet.const(x,n)
    one=C(O);z=Jet([z0,O]+[Z]*(n-1));v=z+C(e)
    p1=div(O,add(O,b1));p2=div(O,add(O,b2));t=C(a)/v
    L=(one-t.scale(mul(rat(2),p2))+t.square().scale(p2)).root()
    den=z*L+C(e);num=one-t.scale(p1)
    Pj=one/v+num/den
    La=(t-one).scale(p2)/(L*v)
    Pa=-one.scale(p1)/(v*den)-num*z*La/den.square()
    w=z*Pj;wa=z*Pa
    if KIND==1:vj=z/(v*den)
    else:vj=-(z.square()*num*(one.scale(rat(2))/v-C(a)/v.square()))/(L*den.square()).scale(rat(2))
    rho=gaussian(z0,n)
    return [v*rho for v in [z*w,w.square(),z*wa,(w*wa).scale(rat(2)),z*vj,w*vj]]

def factored_state(a,e,b1,b2,r,cells=64,order=6):
    A,Q,Aa,Qa,C,T=integrate(lambda z,n:integrands(z,n,a,e,b1,b2),cells,order,6)
    s=sub(O,r);B=mul(a,A);V=mul(sq(a),Q)
    Ba=add(A,mul(a,Aa));Va=add(mul(mul(rat(2),a),Q),mul(sq(a),Qa))
    q=add(O,mul(mul(rat(2),s),sub(V,B)));new=add(r,mul(s,B))
    lam=sub(mul(Ba,add(q,new)),mul(new,Va))
    j=sub(mul(C,add(mul(rat(2),Q),mul(a,Qa))),mul(mul(rat(2),add(A,mul(a,Aa))),T))
    return dict(c1=div(new,sqrt(q)),lam=lam,j1=j,j2=j)

if __name__=='__main__':
    resources();started=time.monotonic();paths.state=factored_state
    sources=['aa3_checks/AA3_paths_factored.py','aa3_checks/AA3_paths.py','aa3_checks/AA3_certify.py','aa3_checks/AA3_interval_core.py','paper/appendix/L_multistep.tex','Z3_ADAM_HISTORY.md']
    hashes=lambda:{s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources}
    before=hashes();rows=[]
    for e in ['1/2','2']:
        for kind in [1,2]:
            KIND=kind;row=paths.path(e,kind)
            row['stored_J_normalization']='J_beta/(a*p_j^2), continuously extended at a=0'
            rows.append(row)
            print('COMPLETE',e,kind,'cells',len(row['cells']),'seconds',time.monotonic()-started,flush=True)
    assert before==hashes()
    result=dict(status='PASS',paths=rows,source_hashes=before,hashes_unchanged=True,resources=resources(),seconds=time.monotonic()-started)
    with (OUT/'AA3_paths_factored_certificate.json').open('x') as f:json.dump(result,f,indent=2)
    print('PASS',result['seconds'],result['resources'],flush=True)
