"""Exact rational root brackets, T=2 Sign, and a history ablation."""
from certify import *

def sign_stats(a):
    # Alternating Taylor series for the half-normal mass from 0 to a.
    acc=Z;power=a;factor=O
    for k in range(22):
        term=div(mul(power,factor),rat(2*k+1))
        acc=sub(acc,term) if k%2 else add(acc,term)
        if k==20:upper=acc[1]
        power=mul(power,sq(a));factor=div(factor,rat(2*(k+1)))
    mass=mul(DEN,(acc[0],upper));p=sub(O,mass)
    B=mul(mul(rat(2),a),density(a));V=mul(mul(rat(4),sq(a)),p)
    q=add(sub(O,mul(rat(2),B)),mul(rat(2),V))
    return dict(B=B,V=V,q=q,H=sub(sq(B),mul(rat(4,25),q)),c0=div(sub(O,B),sqrt(q)),c1=div(B,sqrt(q)))

def frozen_jets(e,z0,n):
    z=Jet([z0,O]+[Z]*(n-1));one=Jet.const(O,n)
    Pj=one.scale(rat(2))/(z+Jet.const(e,n))
    dc=[density(z0)]
    for k in range(1,n+1):dc.append(neg(div(add(mul(z0,dc[k-1]),dc[k-2] if k>=2 else Z),rat(k))))
    weight=z.square()*Jet(dc)
    return [Pj*weight,Pj.square()*weight]

def frozen(e,cells=192,order=10):
    sums=[Z,Z];errs=[Z,Z]
    for i in range(cells):
        c=rat(6*(i*i+(i+1)**2),cells*cells)
        cell=hull(rat(12*i*i,cells*cells),rat(12*(i+1)**2,cells*cells));rad=rat(6*(2*i+1),cells*cells)
        pw=[O]
        for _ in range(order+1):pw.append(mul(pw[-1],rad))
        mid=frozen_jets(e,c,order-1);full=frozen_jets(e,cell,order)
        for j in range(2):
            val=sum_iv([mul(mid[j].c[k],div(mul(rat(2),pw[k+1]),rat(k+1))) for k in range(0,order,2)])
            err=mul((abmax(full[j].c[order]),)*2,div(mul(rat(2),pw[order+1]),rat(order+1)))
            sums[j]=add(sums[j],add(val,(-err[1],err[1])));errs[j]=add(errs[j],err)
    A,Q=[add(x,(-TAIL[1],TAIL[1])) for x in sums];u=rat(2,5)
    disc=sub(mul(add(O,sq(u)),sq(A)),mul(mul(rat(2),sq(u)),Q))
    assert sub(sq(A),mul(mul(rat(2),sq(u)),Q))[0]>0
    a=div(u,add(mul(u,A),sqrt(disc)));B=mul(a,A)
    c0=mul(u,sub(div(O,B),O))
    return dict(A=A,Q=Q,a=a,c0=c0),errs

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='y2_checks/points.json');args=parser.parse_args()
    start=time.monotonic();out=[]
    for es,lo,hi in [('1/10','0.223391402','0.223391423'),('1/2','0.295298152','0.295298174'),('1','0.380601690','0.380601711')]:
        e=F(es);a=hull(F(lo),F(hi));ends=[]
        for x,sg in [(lo,-1),(hi,1)]:
            vals,err=integ(F(x),e,192,10);s=stats(F(x),vals)
            assert s['H'][1]<0 if sg<0 else s['H'][0]>0,('root endpoint',es,x,show(s['H']))
            ends.append(dict(a=x,H=show(s['H']),integrals=[show(v) for v in vals],remainders=[show(v) for v in err]))
        vals,err=integ(a,e,192,10);s=stats(a,vals)
        c0=div(sub(O,s['B']),sqrt(s['q']));c1=div(s['B'],sqrt(s['q']))
        fr,fre=frozen(e);gap=sub(c0,fr['c0']);assert gap[0]>0
        out.append(dict(e=es,a=[lo,hi],endpoints=ends,integrals=[show(v) for v in vals],remainders=[show(v) for v in err],c0=show(c0),c1=show(c1),frozen_direction={k:show(v) for k,v in fr.items()},actual_minus_frozen=show(gap),frozen_remainders=[show(v) for v in fre]))
        print('Adam',es,'root',lo,hi,'c0',show(c0)['display_only'],'history gain',show(gap)['display_only'],flush=True)
    sl,sh='0.204939708','0.204939729';ends=[]
    for x,sg in [(sl,-1),(sh,1)]:
        st=sign_stats(F(x));assert st['H'][1]<0 if sg<0 else st['H'][0]>0
        ends.append(dict(a=x,bounds={k:show(v) for k,v in st.items()}))
    ss=sign_stats(hull(F(sl),F(sh)))
    gd=sqrt(rat(21,25));gap_sign=sub((int(out[0]['c0']['lo']),int(out[0]['c0']['hi'])),ss['c0']);assert gap_sign[0]>0
    result=dict(status='PASS',u='2/5',r='0',beta1='9/10',beta2='99/100',points=out,sign=dict(a=[sl,sh],endpoints=ends,bounds={k:show(v) for k,v in ss.items()}),GD_c0=show(gd),adam_point1_minus_sign=show(gap_sign),cells=192,order=10,power=P,tail=show(TAIL),source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['y2_checks/interval_core.py','y2_checks/certify.py',__file__]},seconds=time.monotonic()-start,resources=resources())
    with open(args.output,'x') as f:json.dump(result,f,indent=2)
    print('PASS','Sign c0',show(ss['c0'])['display_only'],'Adam(.1)-Sign',show(gap_sign)['display_only'],'seconds',result['seconds'],'resources',result['resources'],flush=True)
