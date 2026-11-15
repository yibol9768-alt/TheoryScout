"""Independent exact coverage/decision audit plus direct-recursion quadrature checks.
Does not call the certificate generators or trust their display_only fields.
"""
from certify import *
from scipy.integrate import quad

def read(p):
    with open(p) as f:return json.load(f)
def dec(v):
    assert v['power']==160
    x=(int(v['lo']),int(v['hi']));assert x[0]<=x[1];return x

def cover1(rows,lo,hi):
    rows=sorted((Fraction(a),Fraction(b)) for a,b in rows)
    last=Fraction(lo)
    for a,b in rows:assert a==last and b>a;last=b
    assert last==Fraction(hi)

def primitive_check():
    samples=[(Fraction(i,7),Fraction(j,7)) for i in range(-3,4) for j in range(i,4)]
    def en(x):return hull(F(str(x[0])),F(str(x[1])))
    def has(iv,x):return Fraction(iv[0],S)<=x<=Fraction(iv[1],S)
    for x in samples:
        a=en(x)
        for xx in x:assert has(sq(a),xx*xx)
        if x[0]<=0<=x[1]:assert has(sq(a),Fraction(0))
        if a[0]>=0:
            y=sqrt(a);assert y[0]**2<=a[0]*S and y[1]**2>=a[1]*S
        for y in samples:
            b=en(y)
            for xx in x:
                for yy in y:
                    assert has(add(a,b),xx+yy) and has(mul(a,b),xx*yy)
                    if y[0]>0:assert has(div(a,b),xx/yy)
    assert DEN[0]>F('3/4')[1] and DEN[1]<S
    assert Fraction(3,2)*(1-Fraction(441,20000))*(1-Fraction(441,10000))>Fraction(7,5)
    assert Fraction(7,5)*Fraction(29,50)-Fraction(441,625)==Fraction(133,1250)
    # Cauchy direction bound for the regular beta pair; tail domination constants.
    b1,b2=Fraction(9,10),Fraction(99,100)
    c2=((b1/(1+b1))**2/(b2/(1+b2))+(1/(1+b1))**2/(1/(1+b2)))
    assert c2<4
    assert all(Fraction(c,1)*(Fraction(1,2)**72)<Fraction(1,2)**60 for c in [3,9,30,180,1080])
    return dict(primitive_interval_pairs=len(samples)**2,direction_cauchy_squared=str(c2))

def sources(obj):
    for p,h in obj['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h

def root_audit(row,betas):
    lo,hi=map(Fraction,row['a']);assert lo<hi
    for k,ep in enumerate(row['endpoints']):
        assert Fraction(ep['a'])==[lo,hi][k]
        vs=[dec(v) for v in ep['integrals']];s=stats(F(ep['a']),vs)
        assert s['H']==dec(ep['H'])
        assert s['H'][1]<0 if k==0 else s['H'][0]>0
    vals=[dec(v) for v in row['integrals']];a=hull(F(str(lo)),F(str(hi)));s=stats(a,vals)
    cc=div(sub(O,s['B']),sqrt(s['q']));assert cc==dec(row['c0'])
    # Independent scalar recurrence, not the generator's explicit closed form.
    am=float((lo+hi)/2);em=float(Fraction(row['e']));b1,b2=map(lambda x:float(Fraction(x)),betas)
    def direct(z):
        d=m=v=0.
        for t in [1,2]:
            g=d-z;m=b1*m+(1-b1)*g;v=b2*v+(1-b2)*g*g
            d-=am*(m/(1-b1**t))/(math.sqrt(v/(1-b2**t))+em)
        return d/am
    rho=lambda z:math.sqrt(2/math.pi)*math.exp(-z*z/2)
    qa=quad(lambda z:z*direct(z)*rho(z),0,12,epsabs=1e-12,epsrel=1e-12)[0]
    qq=quad(lambda z:direct(z)**2*rho(z),0,12,epsabs=1e-12,epsrel=1e-12)[0]
    for value,iv in [(qa,vals[0]),(qq,vals[1])]:assert iv[0]/S<value<iv[1]/S
    return {'e':row['e'],'beta':list(betas),'A_direct_float':qa,'Q_direct_float':qq}

if __name__=='__main__':
    start=time.monotonic();primitives=primitive_check();reg=read('y2_checks/region.json');points=read('y2_checks/points.json');hist=read('y2_checks/history.json')
    for obj in [reg,points,hist]:assert obj['status']=='PASS';sources(obj)
    assert reg['inputs_before']==reg['inputs_after']
    for p,h in reg['inputs_before'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    leaves=reg['leaves'];rects=[]
    for row in leaves:
        al,ar=map(Fraction,row['a']);el,er=map(Fraction,row['e']);assert 0<=al<ar<=Fraction(1,2) and Fraction(1,10)<=el<er<=1
        rects.append((al,ar,el,er));a=hull(F(str(al)),F(str(ar)));vals=[dec(v) for v in row['integrals']];s=stats(a,vals)
        for k in ['N','J']:assert s[k]==dec(row['bounds'][k]) and s[k][0]>0
    aa=sorted(set([r[0] for r in rects]+[r[1] for r in rects]));assert aa[0]==0 and aa[-1]==Fraction(1,2)
    for l,r in zip(aa[:-1],aa[1:]):
        mid=(l+r)/2;cover1([(el,er) for al,ar,el,er in rects if al<mid<ar],'1/10','1')
    cover1([r['e'] for r in reg['edge']],'1/10','1')
    for row in reg['edge']:
        s=stats(F('1/2'),[dec(v) for v in row['integrals']]);H5=sub(sq(s['B']),mul(rat(1,4),s['q']))
        assert s['B']==dec(row['B']) and s['B'][0]>0 and H5==dec(row['H5']) and H5[0]>0
    checks=[root_audit(row,['9/10','99/100']) for row in points['points']]
    for row in hist['models']:
        cover1([d['a'] for d in row['derivative_slabs']],'0','2/5')
        for d in row['derivative_slabs']:
            a=hull(F(d['a'][0]),F(d['a'][1]));s=stats(a,[dec(v) for v in d['integrals']]);assert s['N']==dec(d['N']) and s['N'][0]>0
        checks.append(root_audit(row,row['beta']))
    from certify_points import sign_stats
    for k,ep in enumerate(points['sign']['endpoints']):
        s=sign_stats(F(ep['a']));assert {k:show(v) for k,v in s.items()}==ep['bounds'];assert s['H'][1]<0 if k==0 else s['H'][0]>0
    ss=sign_stats(hull(F(points['sign']['a'][0]),F(points['sign']['a'][1])))
    assert ss['c0']==dec(points['sign']['bounds']['c0'])
    assert sub(dec(points['points'][0]['c0']),ss['c0'])==dec(points['adam_point1_minus_sign']) and dec(points['adam_point1_minus_sign'])[0]>0
    for row in points['points']:
        assert sub(dec(row['c0']),dec(row['frozen_direction']['c0']))==dec(row['actual_minus_frozen']) and dec(row['actual_minus_frozen'])[0]>0
    actual=dec(points['points'][2]['c0']);stateless,momonly,rmsonly=[dec(r['c0']) for r in hist['models']]
    expected=dict(adding_momentum_with_rms=sub(rmsonly,actual),adding_momentum_without_rms=sub(stateless,momonly),adding_rms_with_momentum=sub(actual,momonly),adding_rms_without_momentum=sub(rmsonly,stateless),stateless_minus_actual=sub(stateless,actual))
    for k,v in expected.items():assert v==dec(hist['gaps'][k]) and v[0]>0
    result=dict(status='PASS',exact_rectangle_cover=True,rectangle_count=len(leaves),parameter_strip_count=len(aa)-1,all_stored_decisions_recomputed=True,inputs_and_sources_unchanged=True,primitive_checks=primitives,direct_recurrence_numerical_checks=checks,numerical_checks_are_not_proofs=True,seconds=time.monotonic()-start,resources=resources(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with open('y2_checks/verification.json','x') as f:json.dump(result,f,indent=2)
    print('PASS',result,flush=True)
