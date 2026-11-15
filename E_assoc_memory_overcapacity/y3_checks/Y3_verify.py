"""Independent deterministic checks, no MR datasets or retrieval experiments."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    assert os.environ.get(k)=='2'
import csv,json,math,time,resource
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
P=Path(__file__).resolve().parent
start=time.process_time()
with (P/'Y3_coefficients.csv').open() as f: rows=list(csv.DictReader(f))
errs=[]
for method in ['Sign','Adam.1','Adam1','Adam2']:
    for r in [.5,0.,-.5]:
        row=next(a for a in rows if a['method']==method and float(a['r'])==r)
        v=float(row['sigma'])**2;e={'Adam.1':.1,'Adam1':1.,'Adam2':2.}.get(method)
        def response(w):
            def h(g):return 1. if method=='Sign' else math.sqrt(w)*g/(math.sqrt(w)*g+e*math.sqrt(v))
            def ev(fun):return quad(lambda g:math.sqrt(2/math.pi)*math.exp(-g*g/2)*fun(g),0,np.inf,epsabs=1e-12,epsrel=1e-12)[0]
            return np.array([ev(lambda g:g*h(g))/math.sqrt(w),ev(lambda g:h(g)**2)])
        step=v*1e-4
        derivative=(response(v-2*step)-8*response(v-step)+8*response(v+step)-response(v+2*step))/(12*step)
        expected=np.array([float(row['beta_prime']),float(row['Q_prime'])])
        errs.append(float(np.max(abs(derivative-expected))))
assert max(errs)<2e-8

# Six-dimensional fixed scatter factors, embedded into S^{n-1}.
# Tensor Gauss-Hermite degree 9 integrates all degree <=8 polynomials exactly.
n,d,q=43,3,1.7
base=np.arange(1,37,dtype=float).reshape(6,6)
root=np.eye(6)+.03*np.sin(base+base.T)
lx,ly=root[:3],root[3:]
u=np.array([[.9,.2,-.3],[.1,.7,.4],[-.2,.5,.8]])
a=ly.T@u@lx;m=(a+a.T)/2;c=a.T@a
nodes,weights=hermgauss(5)
ix=np.indices((5,)*6).reshape(6,-1).T
g=math.sqrt(2)*nodes[ix]
w=np.prod(weights[ix]/math.sqrt(math.pi),axis=1)
qm=np.einsum('bi,ij,bj->b',g,m,g)
qc=np.einsum('bi,ij,bj->b',g,c,g)
tr=lambda z:float(np.trace(z))
tm,tc=tr(m),tr(c)
raw={'M':tm,'C':tc,'M2':tm**2+2*tr(m@m),
     'MC':tm*tc+2*tr(m@c),
     'M3':tm**3+6*tm*tr(m@m)+8*tr(m@m@m),
     'C2':tc**2+2*tr(c@c),
     'M2C':tc*(tm*tm+2*tr(m@m))+4*tm*tr(m@c)+8*tr(m@m@c),
     'M4':tm**4+12*tm*tm*tr(m@m)+12*tr(m@m)**2+32*tm*tr(m@m@m)+48*tr(m@m@m@m)}
poly={'M':qm,'C':qc,'M2':qm**2,'MC':qm*qc,'M3':qm**3,'C2':qc**2,'M2C':qm**2*qc,'M4':qm**4}
gauss_err={key:abs(float(w@poly[key])-value) for key,value in raw.items()}
assert max(gauss_err.values())<1e-8
degrees={'M':1,'C':1,'M2':2,'MC':2,'M3':3,'C2':2,'M2C':3,'M4':4}
s={key:value/math.prod(n+2*j for j in range(degrees[key])) for key,value in raw.items()}
fac=n/(d*math.sqrt(q))
moments=dict(D=fac*s['M'],D2=fac**2*s['M2'],
             V=fac**2/(n-1)*(s['C']-s['M2']),
             DV=fac**3/(n-1)*(s['MC']-s['M3']),
             V2=fac**4/(n-1)**2*(s['C2']-2*s['M2C']+s['M4']))
assert abs(moments['D']-tr(a)/(d*math.sqrt(q)))<1e-12
ev=n/(d*d*q*(n-1))*(tr(c)-(tr(a)**2+2*tr(m@m))/(n+2))
assert abs(ev-moments['V'])<1e-12
vard=2*n/(d*d*q*(n+2))*(tr(m@m)-tr(a)**2/n)
assert abs(vard-(moments['D2']-moments['D']**2))<1e-12

# Independently insert X2 shared-key GD formula (18), not the Y3 closure expression.
gd_err=[]
for row in rows:
    if row['method']!='GD':continue
    mu,mm,ell=[float(row[k]) for k in ['mu','m','ell']]
    cc=mu*mu;t=1.2815515655446004-mu
    gg=mm-mu*ell/2-mu/4
    br=math.exp(-t*t/2)/math.sqrt(2*math.pi)*(gg-mu/4-mu*cc/2-t*cc/2)
    gd_err.append(abs(br-float(row['B_recall'])))
assert max(gd_err)<1e-12

cg=Path('/proc/self/cgroup').read_text().strip().split('::')[-1]
cp=Path('/sys/fs/cgroup')/cg.lstrip('/')
result=dict(checks='deterministic, no random MR data, no preregistered experiment run',
            derivative_max_error=max(errs),gaussian_cubature_errors=gauss_err,
            finite_sphere_exact_joint_moments=moments,gd_x2_max_error=max(gd_err),
            cpu_seconds=time.process_time()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            memory_max=(cp/'memory.max').read_text().strip(),memory_swap_max=(cp/'memory.swap.max').read_text().strip())
assert result['memory_max']=='3221225472' and result['memory_swap_max']=='0'
with (P/'Y3_verification.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
