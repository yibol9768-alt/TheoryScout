"""Independent broadcasting/reconstruction audit; smoke seeds only, no new fitting."""
import json
from pathlib import Path
import time
import numpy as np
from run import OUT,ROOT,B,ETA,EPS,digest,dump,frozen,direction,tangent,layout,grad,advance,probes,popG,samples,utc
START=time.monotonic(); frozen()
meta=json.loads((OUT/'formal/metadata.json').read_text())
assert meta['complete'] and digest(OUT/'run.py')==meta['code_sha256']
mo=dict(np.load(OUT/'formal/model_p2_n32.npz'))
lay=layout(2); k=len(lay['kind']); n=32
rng=np.random.default_rng(3991)
w=rng.standard_normal((k,n)); m=.03*rng.standard_normal((k,n)); v=.1+abs(rng.standard_normal((k,n)))
amp=np.full(k,float(mo['a0'])); t=7
# Same independent smoke batches, explicitly one at a time.
r1=np.random.default_rng(3991); r2=np.random.default_rng(3991)
metrics,mu=probes(w,m,v,amp,t,lay,mo,r1,count=16)
ds=[]; dsignals=[]
q=direction(w)/np.sqrt(n); G=popG(w,amp,mo); un=G/np.linalg.norm(G,axis=-1,keepdims=True)
# samples() consumes normals for all x then y. Match a single vectorized block.
xx,yy=samples(r2,mo,16*B); xx=xx.reshape(16,B,n); yy=yy.reshape(16,B)
for x,y in zip(xx,yy):
    g,ga=grad(w,amp,x,y)
    nw,_,_,_,_,_=advance(w,m,v,amp,g,ga,t+1,lay)
    d=tangent(direction(nw)/np.sqrt(n)-q,q); ds.append(d); dsignals.append(-np.sum(d*un,axis=-1))
ds=np.array(ds); means=ds.mean(0); variances=np.sum(np.var(ds,axis=0,ddof=1),axis=-1)
mu_error=float(np.max(abs(mu-means))); var_error=float(np.max(abs(metrics[:,1]-variances)))
d_error=float(np.max(abs(metrics[:,0]-np.mean(dsignals,axis=0))))
assert max(mu_error,var_error,d_error)<1e-12
# Separate projected model finite difference in coefficient, amplitude and w coordinates.
x,y=xx[0],yy[0]; g,ga=grad(w,amp,x,y); h=1e-5
ww=w[:1].copy(); aa=amp[:1].copy()
def loss(w,a): return .5*np.mean((a[0]*(direction(w)[0]@x.T)-y)**2)
fd=[]
for j in range(n):
    dw=np.zeros_like(ww); dw[0,j]=h
    fd.append((loss(ww+dw,aa)-loss(ww-dw,aa))/(2*h))
ge=float(np.linalg.norm(np.array(fd)-g[0])/np.linalg.norm(g[0]))
ae=float(abs((loss(ww,aa+h)-loss(ww,aa-h))/(2*h)-ga[0])/max(abs(ga[0]),1e-15))
assert ge<1e-6 and ae<1e-6
count=0; reconstruction=0.; proj=0.
for task in meta['tasks']:
    f=np.load(OUT/'formal'/task['file']); count+=f['risks'].size
    err=np.max(abs(f['risks']-f['modes'].sum(-1)))
    reconstruction=max(reconstruction,float(err))
    pidx=np.flatnonzero(f['kind']==1)
    proj=max(proj,float(np.max(abs(f['monitor'][:,pidx,6]))))
    assert np.isfinite(f['monitor']).all() and np.isfinite(f['modes']).all()
    if f['probes'].size:
        assert np.isfinite(f['probes'][:,:,[0,1,3,4,5,6,7]]).all()
        assert np.isfinite(f['probe_mu']).all()
    f.close()
assert count==69480 and reconstruction<1e-12 and proj<1e-12
result=dict(passed=True,utc=utc(),probe_mu_error=mu_error,probe_variance_error=var_error,
    probe_signal_error=d_error,projected_gradient_error=ge,amplitude_gradient_error=ae,
    modal_risk_reconstruction_error=reconstruction,projected_applied_cross_term_max=proj,
    endpoint_count=count,prereg_sha256=frozen(),wall_seconds=time.monotonic()-START)
dump(OUT/'audit.json',result); print(json.dumps(result,indent=2))
