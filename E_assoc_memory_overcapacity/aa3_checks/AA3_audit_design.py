"""Deterministic audit, power arithmetic, and a VALID one-step BR control.
Does not generate finite-dimensional training data or run a preregistered experiment.
"""
from AA3_numeric import *
from fractions import Fraction as F
import hashlib

def read(v):return (F(int(v['lo']),2**v['power']),F(int(v['hi']),2**v['power']))
def ivsub(a,b):return a[0]-b[1],a[1]-b[0]
def cover1(rows,key,hi):
    ranges=sorted((F(x[key][0]),F(x[key][1])) for x in rows)
    assert ranges[0][0]==0 and ranges[-1][1]==hi
    assert all(a<b for a,b in ranges)
    assert all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))

def saved_audit():
    result={}
    for name in ['continuous','points','paths']:
        file=OUT/('AA3_'+name+'_certificate.json')
        if name=='paths' and (OUT/'AA3_paths_factored_certificate.json').exists():
            file=OUT/'AA3_paths_factored_certificate.json'
        if name=='paths' and (OUT/'AA3_centered_paths_certificate.json').exists():
            file=OUT/'AA3_centered_paths_certificate.json'
        if not file.exists():continue
        cert=json.loads(file.read_text());assert cert['status']=='PASS'
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cert['source_hashes'].items())
        if name=='continuous':
            for row in cert['continuous']:
                cover1(row['cells'],'a',F(row['amax']))
                for c in row['cells']:
                    assert read(c['lam'])[0]>0 and read(c['j1'])[1]<0
                    assert read(c['j2'])[0]>0 if row['e']=='1/2' else read(c['j2'])[1]<0
                assert read(row['c1_at_amax'])[0]>F('11/20')
            result[name]=[dict(e=r['e'],cells=len(r['cells'])) for r in cert['continuous']]
        elif name=='points':
            for row in cert['points']:
                u=F('1/4')+F(row['delta'])
                for arm in row['rows']:
                    lo,hi=map(F,arm['root']);assert 0<lo<hi
                    cover1(arm['monotonic'],'a',hi)
                    assert all(read(c['lam'])[0]>0 for c in arm['monotonic'])
                    assert read(arm['endpoint_c1'][0])[1]<u<read(arm['endpoint_c1'][1])[0]
                A,B,C=[read(a['c0']) for a in row['rows']]
                for gap,calc in [(row['gap1'],ivsub(C,A)),(row['gap2'],ivsub(C,B))]:
                    assert read(gap)[0]<=calc[0]<=calc[1]<=read(gap)[1]
                assert read(row['gap1'])[1]<0
                assert read(row['gap2'])[0]>0 if row['e']=='1/2' else read(row['gap2'])[1]<0
            result[name]=len(cert['points'])
        else:
            for row in cert['paths']:
                amax,bmax=F(row['amax']),F(row['beta_max']);rects=row['cells']
                bounds=sorted({F(x) for c in rects for x in c['a']})
                assert bounds[0]==0 and bounds[-1]==amax
                for al,ah in zip(bounds,bounds[1:]):
                    mid=(al+ah)/2
                    cover1([c for c in rects if F(c['a'][0])<=mid<=F(c['a'][1])],'beta',bmax)
                for c in rects:
                    assert 0<=F(c['a'][0])<F(c['a'][1])<=amax and 0<=F(c['beta'][0])<F(c['beta'][1])<=bmax
                    assert read(c['lam'])[0]>0
                    positive=row['kind']==2 and row['e']=='1/2'
                    assert read(c['J_over_a'])[0]>0 if positive else read(c['J_over_a'])[1]<0
                cover1(row['endpoint_cover'],'beta',bmax)
                assert all(read(c['c1'])[0]>F('11/20') for c in row['endpoint_cover'])
            result[name]=[dict(e=r['e'],kind=r['kind'],cells=len(r['cells']),endpoint_cells=len(r['endpoint_cover'])) for r in cert['paths']]
    near=OUT/'AA3_near_threshold_certificate.json'
    if near.exists():
        cert=json.loads(near.read_text());assert cert['status']=='PASS'
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cert['source_hashes'].items())
        for row in cert['cases']:
            u=F('1/4')+F(row['delta'])
            for arm in row['rows']:
                lo,hi=map(F,arm['root']);assert 0<lo<hi
                cover1(arm['monotonic'],'a',hi)
                assert all(read(c['lam'])[0]>0 for c in arm['monotonic'])
                assert read(arm['endpoints'][0])[1]<u<read(arm['endpoints'][1])[0]
            gap=ivsub(read(row['rows'][1]['c0']),read(row['rows'][0]['c0']))
            assert read(row['gap'])[0]<=gap[0]<=gap[1]<=read(row['gap'])[1]
            assert gap[0]>0 if row['positive'] else gap[1]<0
        result['near_threshold']=len(cert['cases'])
    return result

def raw(z,a,e,b1,b2):
    d=np.zeros_like(z);m=d.copy();v=d.copy();dd=np.zeros((3,len(z)));dm=dd.copy();dv=dd.copy();one=dd.copy();one[0]=1
    for t in (1,2):
        g=d-z;dg=dd
        nm=b1*m+(1-b1)*g;ndm=b1*dm+(1-b1)*dg;ndm[1]+=m-g
        nv=b2*v+(1-b2)*g*g;ndv=b2*dv+(1-b2)*2*g*dg;ndv[2]+=v-g*g
        m,v,dm,dv=nm,nv,ndm,ndv
        mh=m/(1-b1**t);dmh=dm/(1-b1**t);dmh[1]+=m*t*b1**(t-1)/(1-b1**t)**2
        vh=v/(1-b2**t);dvh=dv/(1-b2**t);dvh[2]+=v*t*b2**(t-1)/(1-b2**t)**2
        den=np.sqrt(vh)+e;dden=dvh/(2*np.sqrt(vh))
        dd=dd-one*mh/den-a*(dmh/den-mh*dden/den**2);d=d-a*mh/den
    return d,dd

def raw_audit():
    errors=[]
    for e in [.5,2.]:
        for a in [e*.001,e*.1,e*.3,e*.6]:
            for b1,b2 in [(0,.99),(.9,0),(.9,.99),(.4,.7)]:
                direct,dt=raw(z,a,e,b1,b2);formula=endpoint(a,e,b1,b2)
                errors.append(float(np.max(abs(direct-formula[0]))))
                errors.append(float(np.max(abs(dt-np.array(formula[1:])))))
    assert max(errors)<1e-11
    return dict(cases=32,max_endpoint_or_derivative_error=max(errors))

def one_step(e):
    r=.25;u=.55;kappa=.5;rho=.1;sigma=np.sqrt(2*(1-r));zz=ndtri(1-rho)
    h=z/(z+e);beta=(weight@(z*h))/sigma;Q=weight@(h*h)
    dbeta=(weight@((z**3-3*z)*h))/(2*sigma**3)
    dQ=(weight@((z*z-1)*h*h))/(2*sigma*sigma)
    def values(alpha):
        q=1-alpha*sigma*sigma*beta+alpha*alpha*Q
        num=np.array([1-alpha*(1-r)*beta,r+alpha*(1-r)*beta])
        return num/np.sqrt(q)
    alpha=brentq(lambda a:values(a)[1]-u,0,2)
    c=values(alpha);q=1-alpha*sigma*sigma*beta+alpha*alpha*Q
    ell=(1+2*alpha*(beta*(r-3)+(r-1)*(3-2*r)*dbeta)+alpha*alpha*(2*sigma*sigma*beta*beta+(3-2*r)*dQ))/(kappa*q)
    qp=-sigma*sigma*beta+2*alpha*Q
    dc=np.array([-(1-r)*beta,(1-r)*beta])/np.sqrt(q)-c*qp/(2*q)
    rows=[]
    for k,(r0,r1) in enumerate([(1,r),(r,1)]):
        mu=c[k]/np.sqrt(kappa);m=alpha*((r1-r0)*(3-2*r)*dbeta-r0*beta)/(kappa**1.5*np.sqrt(q))
        t=zz-mu;phi=np.exp(-t*t/2)/np.sqrt(2*np.pi)
        BR=phi*(m-mu*(ell+mu*mu)/2-mu/2-mu*mu*t/2)
        rows.append(dict(task=k,c=c[k],Rinf=ndtr(-t),BR=BR,Ralpha=phi*dc[k]/np.sqrt(kappa)))
    corrected=[]
    for d in [1024,2048]:
        shift=-rows[1]['BR']/(d*rows[1]['Ralpha'])
        corrected.append(dict(d=d,alpha_shift=shift,alpha_corrected=alpha+shift,
                              old_matched_prediction=rows[0]['Rinf']+(rows[0]['BR']-rows[0]['Ralpha']/rows[1]['Ralpha']*rows[1]['BR'])/d))
    return dict(e=e,alpha=alpha,rows=rows,finite_matched=corrected)

if __name__=='__main__':
    import sys
    resources();started=time.monotonic()
    result=dict(saved=saved_audit(),raw_recursion=raw_audit(),one_step_controls=[one_step(e) for e in [.5,2.]])
    num=json.loads((OUT/'AA3_numeric.json').read_text())
    table=[]
    for e in [.5,2.]:
        arms=[match(.25,.3,e,*b) for b in [(0,0),(0,.99),(.9,0),(.9,.99)]]
        for arm in arms:arm['old_Rinf']=ndtr(arm['c0']/np.sqrt(.5)-ndtri(.9))
        contrast=[x for x in num['contrasts'] if x['e']==e and x['delta']==.3]
        for x in contrast:
            x=dict(x);x['sd_budget_trace']=2*abs(x['gap']);x['iid_sd_d2048']=x['iid_influence_sd']/2048
            x['inflation_allowed']=x['sd_budget_trace']/x['iid_sd_d2048'];x['normal_power_B128_budget']=float(ndtr(np.sqrt(128)/2-ndtri(.9875)))
            x['rank_sd_budget']=.0002;x['normal_rank_power_B1024']=float(ndtr(abs(x['recall_gap'])*np.sqrt(1024)/.0002-ndtri(.9875)))
            table.append(x)
        result.setdefault('arms',[]).extend(arms)
    result.update(design_table=table,resources=resources(),seconds=time.monotonic()-started)
    suffix=sys.argv[1] if len(sys.argv)>1 else 'design'
    with (OUT/('AA3_'+suffix+'.json')).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2),flush=True)
