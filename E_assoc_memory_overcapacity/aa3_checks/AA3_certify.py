"""160-bit outward integer intervals; new continuous-target certificates.

No floating number decides a proof branch. Core copied verbatim from z3.
"""
from AA3_interval_core import *
import argparse

OUT=Path(__file__).resolve().parent

def F(x):
    x=Fraction(str(x));return rat(x.numerator,x.denominator)
def enc(x):return dict(lo=str(x[0]),hi=str(x[1]),power=P,display=[x[0]/S,x[1]/S])
def box(l,h):return (F(l)[0],F(h)[1])
def resources():
    c=Path('/sys/fs/cgroup')/next(s.split(':',2)[2].strip().lstrip('/') for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
    ans={s:(c/s).read_text().strip() for s in ['memory.max','memory.swap.max','memory.peak']}
    ans['threads']=len(list(Path('/proc/self/task').iterdir()))
    ans['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert ans['memory.max']=='3221225472' and ans['memory.swap.max']=='0' and ans['threads']<=2
    return ans

def gaussian(z0,n):
    c=[density(z0)]
    for k in range(1,n+1):c.append(neg(div(add(mul(z0,c[k-1]),c[k-2] if k>=2 else Z),rat(k))))
    return Jet(c)

def jets(z0,n,a,e,b1,b2):
    C=lambda x:Jet.const(x,n)
    one=C(O);z=Jet([z0,O]+[Z]*(n-1));v=z+C(e)
    p1=div(O,add(O,b1));p2=div(O,add(O,b2));t=C(a)/v
    L=(one-t.scale(mul(rat(2),p2))+t.square().scale(p2)).root()
    den=z*L+C(e);num=one-t.scale(p1)
    Pj=one/v+num/den
    La=(t-one).scale(p2)/(L*v)
    Pa=-one.scale(p1)/(v*den)-num*z*La/den.square()
    w=z*Pj;wa=z*(Pj+Pa.scale(a))
    # Correct w_a below: w=d/a=z*P, hence w_a=z*P_a (not d_a).
    wa=z*Pa
    v1=z.scale(sq(p1))/(v*den)
    v2=-(z.square()*num*(one.scale(rat(2))/v-C(a)/v.square())).scale(sq(p2))/(L*den.square()).scale(rat(2))
    rho=gaussian(z0,n)
    return [y*rho for y in [z*w,w.square(),z*wa,(w*wa).scale(rat(2)),z*v1,w*v1,z*v2,w*v2]]

def integrate(fun,cells=64,order=6,count=8):
    out=[Z]*count
    for i in range(cells):
        mid=rat(6*(i*i+(i+1)**2),cells*cells)
        whole=(rat(12*i*i,cells*cells)[0],rat(12*(i+1)**2,cells*cells)[1])
        h=rat(6*(2*i+1),cells*cells);powers=[O]
        for _ in range(order+1):powers.append(mul(powers[-1],h))
        center=fun(mid,order-1);full=fun(whole,order)
        for j in range(count):
            val=sum_iv([mul(center[j].c[k],div(mul(rat(2),powers[k+1]),rat(k+1))) for k in range(0,order,2)])
            err=mul((abmax(full[j].c[order]),)*2,div(mul(rat(2),powers[order+1]),rat(order+1)))
            out[j]=add(out[j],add(val,(-err[1],err[1])))
    # e>=.5, a/e<=.65, beta_j>=0: envelopes in AA3_ADAM_PREDICTIONS.md.
    tail=rat(1,1<<60)
    return [add(v,(-tail[1],tail[1])) for v in out]

def state(a,e,b1,b2,r,cells=64,order=6):
    vals=integrate(lambda z,n:jets(z,n,a,e,b1,b2),cells,order)
    A,Q,Aa,Qa,C1,T1,C2,T2=vals;s=sub(O,r)
    B=mul(a,A);V=mul(sq(a),Q)
    Ba=add(A,mul(a,Aa));Va=add(mul(mul(rat(2),a),Q),mul(sq(a),Qa))
    q=add(O,mul(mul(rat(2),s),sub(V,B)))
    new=add(r,mul(s,B));old=sub(O,mul(s,B))
    lam=sub(mul(Ba,add(q,new)),mul(new,Va))
    # J_beta/a, continuously extended at a=0.
    js=[sub(mul(C,add(mul(rat(2),Q),mul(a,Qa))),mul(mul(rat(2),add(A,mul(a,Aa))),T)) for C,T in [(C1,T1),(C2,T2)]]
    return dict(c0=div(old,sqrt(q)),c1=div(new,sqrt(q)),q=q,lam=lam,j1=js[0],j2=js[1],integrals=vals)

def continuous(e_text,amax_text,target='3/10'):
    e=F(e_text);r=F('1/4');b1=F('2/5');b2=F('7/10');amax=Fraction(amax_text)
    stack=[(Fraction(0),amax)];rows=[];checks=0
    while stack:
        lo,hi=stack.pop();st=state(box(lo,hi),e,b1,b2,r)
        checks+=1
        good=st['lam'][0]>0 and st['j1'][1]<0 and (st['j2'][0]>0 if e_text=='1/2' else st['j2'][1]<0)
        if good:
            rows.append(dict(a=[str(lo),str(hi)],lam=enc(st['lam']),j1=enc(st['j1']),j2=enc(st['j2'])))
        else:
            assert hi-lo>Fraction(1,2**18),(e_text,str(lo),str(hi),{k:enc(st[k]) for k in ['lam','j1','j2']})
            mid=(lo+hi)/2;stack.extend([(mid,hi),(lo,mid)])
        if checks%25==0:print('continuous',e_text,'checks',checks,'accepted',len(rows),'remaining',len(stack),flush=True)
    rows.sort(key=lambda x:Fraction(x['a'][0]));assert rows[0]['a'][0]=='0' and Fraction(rows[-1]['a'][1])==amax
    assert all(Fraction(x['a'][1])==Fraction(y['a'][0]) for x,y in zip(rows,rows[1:]))
    end=state(F(amax),e,b1,b2,r,128,10)
    assert end['c1'][0]>F(Fraction(1,4)+Fraction(target))[1]
    return dict(e=e_text,r='1/4',beta1='2/5',beta2='7/10',amax=str(amax),delta_certified=target,
                c1_at_amax=enc(end['c1']),cells=rows,checks=checks)

def points(numeric):
    rows=[]
    for e in ['1/2','2']:
        for dt in ['1/100','1/10','1/4','3/10']:
            group=[]
            for b1,b2 in [('0','99/100'),('9/10','0'),('9/10','99/100')]:
                hint=next(x for x in numeric['rows'] if x['e']==float(Fraction(e)) and x['delta']==float(Fraction(dt)) and x['beta1']==float(Fraction(b1)) and x['beta2']==float(Fraction(b2)))
                # Float is ONLY a hint. Both exact rational endpoints are re-integrated.
                center=Fraction(round(hint['a']*10**14),10**14);rad=Fraction(2,10**12)
                lo,hi=center-rad,center+rad;u=F(Fraction(1,4)+Fraction(dt))
                ends=[state(F(a),F(e),F(b1),F(b2),F('1/4'),128,10) for a in [lo,hi]]
                assert ends[0]['c1'][1]<u[0] and ends[1]['c1'][0]>u[1]
                # First crossing proved independently over [0,hi].
                mono=[]
                for j in range(16):
                    st=state(box(hi*j/16,hi*(j+1)/16),F(e),F(b1),F(b2),F('1/4'),48,4)
                    assert st['lam'][0]>0
                    mono.append(dict(a=[str(hi*j/16),str(hi*(j+1)/16)],lam=enc(st['lam'])))
                st=state(box(lo,hi),F(e),F(b1),F(b2),F('1/4'),128,10)
                group.append(dict(beta1=b1,beta2=b2,root=[str(lo),str(hi)],c0=enc(st['c0']),
                                  endpoint_c1=[enc(x['c1']) for x in ends],monotonic=mono))
                print('point',e,dt,b1,b2,enc(st['c0'])['display'],flush=True)
            ca,cb,cc=[(int(x['c0']['lo']),int(x['c0']['hi'])) for x in group]
            gap1=sub(cc,ca);gap2=sub(cc,cb)
            assert gap1[1]<0 and (gap2[0]>0 if e=='1/2' else gap2[1]<0)
            rows.append(dict(e=e,delta=dt,rows=group,gap1=enc(gap1),gap2=enc(gap2)))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['test','continuous','points']);args=p.parse_args()
    resources();started=time.monotonic()
    sources=['ICML_REVIEW_SIM6.md','Z3_ADAM_HISTORY.md','PROOF_REVIEW17.md','Z2B_BR_PROOF.md','paper/appendix/L_multistep.tex','paper/appendix/K_ft_rank.tex','aa3_checks/AA3_interval_core.py','aa3_checks/AA3_certify.py','aa3_checks/AA3_numeric.py']
    hashes=lambda:{s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources}
    before=hashes()
    if args.mode=='test':
        for e in ['1/2','2']:
            st=state(box(0,Fraction(e)/100),F(e),F('2/5'),F('7/10'),F('1/4'))
            print(e,{k:enc(st[k])['display'] for k in ['lam','j1','j2']},flush=True)
        print('seconds',time.monotonic()-started,flush=True);exit()
    if args.mode=='continuous':result=dict(continuous=[continuous('1/2','29/100'),continuous('2','53/100')])
    else:result=dict(points=points(json.loads((OUT/'AA3_numeric.json').read_text())))
    assert before==hashes()
    result.update(status='PASS',source_hashes=before,hashes_unchanged=True,resources=resources(),seconds=time.monotonic()-started)
    with (OUT/('AA3_'+args.mode+'_certificate.json')).open('x') as f:json.dump(result,f,indent=2)
    print('PASS',args.mode,result['seconds'],result['resources'],flush=True)
