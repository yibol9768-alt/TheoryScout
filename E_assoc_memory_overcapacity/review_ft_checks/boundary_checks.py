"""Boundary counterexamples and moment checks, independent of author code."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ[k]=='2'
import json, math, time, resource
from pathlib import Path
from decimal import Decimal, localcontext
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

OUT=Path(__file__).resolve().parent
START=time.monotonic()
rg=np.random.default_rng(741523)

def msr(u,x,y):
    d,n=x.shape
    ux=u@x
    diag=np.sum(y*ux,axis=0)
    wm=(y.sum(axis=1)@ux-diag)/(n-1)
    total=np.sum((ux@ux.T)*(y@y.T))
    vv=(total-diag@diag)/(n*(n-1))-np.mean(wm*wm)
    return float(np.mean(diag-wm)/np.sqrt(vv))

def excess(u,s,opt):
    v=u-opt
    return float(np.mean((v@s)*v))

def make_data(d,ry):
    n=d*d
    x,y,z=[rg.normal(size=(d,n))/math.sqrt(d) for _ in range(3)]
    y1=ry*y+math.sqrt(1-ry*ry)*z
    s=d/n*(x@x.T)
    z0,z1=[d/math.sqrt(n)*(a@x.T) for a in (y,y1)]
    opt=np.linalg.solve(s,z1.T).T
    return x,y,y1,s,z0,z1,opt

negative=[]
shrink=[]
for d in (32,64,128):
    for seed in range(4):
        x,y,y1,s,z0,z1,opt=make_data(d,-.99)
        base=excess(z0,s,opt)
        grad=z0@s-z1
        row=dict(d=d,seed=seed,baseline_loss=base)
        for name,direction in [('GD',grad),('Adam_e10',grad/(np.abs(grad)+10*math.sqrt(3.98)))]:
            aa=float(np.mean((direction@s)*direction))
            bb=float(np.mean((direction@s)*(z0-opt)))
            disc=bb*bb-aa*(1-.0625)*base
            assert disc>0
            alpha=(1-.0625)*base/(bb+math.sqrt(disc))
            u=z0-alpha*direction
            relative=excess(u,s,opt)/base
            assert abs(relative-.0625)<1e-12
            row[name]=dict(alpha=alpha,relative_loss=relative,old_msr=msr(u,x,y))
        row['Adam_minus_GD']=row['Adam_e10']['old_msr']-row['GD']['old_msr']
        negative.append(row)
        x,y,y1,s,z0,z1,opt=make_data(d,0.)
        base=excess(z0,s,opt)
        aa=float(np.mean((z0@s)*z0))
        bb=float(np.mean(z0*z1))
        cc=float(np.mean(opt*z1))
        scale=(bb+math.sqrt(bb*bb+aa*(.75*base-cc)))/aa
        u=scale*z0
        row=dict(d=d,seed=seed,scale=scale,relative_loss=excess(u,s,opt)/base,
                 old_msr_before=msr(z0,x,y),old_msr_after=msr(u,x,y),
                 new_msr_before=msr(z0,x,y1),new_msr_after=msr(u,x,y1))
        assert abs(row['relative_loss']-.75)<1e-12
        assert abs(row['old_msr_after']-row['old_msr_before'])<1e-12
        assert abs(row['new_msr_after']-row['new_msr_before'])<1e-12
        shrink.append(row)

def normalized_delta(t,name,b1,b2):
    u=m=v=0.
    for it in range(1,5):
        g=u-t
        if name=='GD': u-=.08*g
        elif name=='Sign': u-=.08*np.sign(g)
        else:
            m=b1*m+(1-b1)*g
            v=b2*v+(1-b2)*g*g
            eps=1. if b1==b2==0. else .1
            u-=.08*(m/(1-b1**it))/(math.sqrt(v/(1-b2**it))+eps)
    return u

raw=json.loads((OUT/'check_results.json').read_text())['real_tasks']
moments={}
density=lambda z: math.sqrt(2/math.pi)*math.exp(-z*z/2)
phi=lambda z: math.exp(-z*z/2)/math.sqrt(2*math.pi)
for name,b1,b2 in [('GD',.9,.99),('Sign',.9,.99),('Adam',.9,.99),('Adam',.9,0.),('Adam',0.,0.)]:
    if name=='GD':
        bn=1-.92**4; vn=bn*bn
    elif name=='Sign':
        edges=[0.,.08,.16,.24,.32,math.inf]
        bn=vn=0.
        for a,b in zip(edges[:-1],edges[1:]):
            point=(a+b)/2 if math.isfinite(b) else a+1
            delta=normalized_delta(point,name,b1,b2)
            bn+=2*delta*(phi(a)-phi(b))
            vn+=2*delta*delta*(ndtr(b)-ndtr(a))
    else:
        bn=quad(lambda z:z*normalized_delta(z,name,b1,b2)*density(z),0,math.inf,epsabs=1e-11)[0]
        vn=quad(lambda z:normalized_delta(z,name,b1,b2)**2*density(z),0,math.inf,epsabs=1e-11)[0]
    moments[(name,b1,b2)]=(bn,vn)
summary=[]
for d in (16,32,64,128):
    for key,(bn,vn) in moments.items():
        rr=[r for r in raw if r['d']==d and (r['method'],r['beta1'],r['beta2'])==key]
        errors={k:[] for k in ('q','cross','loss','old_margin','new_margin','old_var','new_var','old_msr','new_msr')}
        for r in rr:
            sig2=2*(1-r['rx']*r['ry'])
            b,v=sig2*bn,sig2*vn
            q=1-b+v
            a0,a1=1-b/2,r['rx']*r['ry']+b/2
            pred=dict(q=q,cross=-b/2,loss=sig2-2*b+v,old_margin=a0,new_margin=a1,
                      old_var=q,new_var=q,old_msr=a0/math.sqrt(q),new_msr=a1/math.sqrt(q))
            observed=dict(q=r['q'],cross=r['cross'],loss=r['empirical_loss'],
                          old_margin=r['old']['margin'],new_margin=r['new']['margin'],
                          old_var=r['old']['variance'],new_var=r['new']['variance'],
                          old_msr=r['old']['msr'],new_msr=r['new']['msr'])
            for k in errors: errors[k].append(observed[k]-pred[k])
        summary.append(dict(d=d,key=list(key),records=len(rr),
            mean_abs_errors={k:float(np.mean(np.abs(v))) for k,v in errors.items()},
            max_trace_error=max(abs(r[task]['total_trace_error']) for r in rr for task in ('old','new'))))

fixed_accuracy=[]
with localcontext() as ctx:
    ctx.prec=70
    for t in (1,4):
        denom=Decimal(32*t)
        pts=[Decimal(0)]
        target=Decimal(3)/4
        fn=lambda a:(1-a)**(2*t)-target
        for j in range(1,33):
            a=Decimal(3)/5*Decimal(j)/denom
            pts.append(a)
            if fn(pts[-2])>=0 and fn(a)<=0:
                lo,hi=pts[-2:]
                break
        for _ in range(24):
            mid=(lo+hi)/2
            pts.append(mid)
            if fn(mid)>0: lo=mid
            else: hi=mid
        best=min(pts,key=lambda a:abs(fn(a)))
        fixed_accuracy.append(dict(T=t,selected_decimal=str(best),
                                   loss_error_decimal=str(fn(best))))

result=dict(negative_mean_real_tasks=negative,shrinkage_matching=shrink,
            normalized_moments={str(k):v for k,v in moments.items()},moment_summary=summary,
            fixed_accuracy_decimal=fixed_accuracy,
            resources=dict(seconds=time.monotonic()-START,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
with (OUT/'boundary_results.json').open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
print('negative-mean exact empirical matching:')
for d in (32,64,128):
    rr=[r for r in negative if r['d']==d]
    print(d,'Adam-GD differences',[r['Adam_minus_GD'] for r in rr])
print('decimal fixed-accuracy:',json.dumps(fixed_accuracy,indent=2))
print('resources:',result['resources'])
