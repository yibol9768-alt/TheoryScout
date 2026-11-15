"""Finite-step query-conditional LMS moments; no parameters fitted to data."""
import numpy as np
from scipy.special import ndtr, roots_genlaguerre, gammaln
from numpy.polynomial.hermite import hermgauss

DIMS = (64, 128, 256)
LOADS = (2, 4, 8)
ALPHAS = (0., 1., 1.5)
FAMILIES = ('bounded', 'raw')
READOUTS = ('tau4', 'tau8', 'tau16', 'full')
BETAS = (.025, .05, .1, .2, .4, .8, 1.2)
METHODS = ('hebb',) + tuple(f'delta_{b:g}' for b in BETAS)
TEST_SEEDS = tuple(range(5001, 5017))
FORECAST_SEEDS = tuple(range(61001, 61017))
SMOKE_SEED = 94031
ORACLE_DIMS = (64, 256)
ORACLE_METHODS = ('hebb', 'delta_0.4', 'delta_1.2')
ORACLE_DRAWS = 128
RHO_GRID = np.linspace(-1., 1., 2049)

def spectrum(d, a, family):
    x = np.arange(1, d+1, dtype=float)
    v = (x/d + .05)**(-a) if family == 'bounded' else x**(-a)
    return v/v.mean()

def query_ids(n):
    return np.floor((np.arange(min(n,256))+.5)*n/min(n,256)).astype(int)

def bins(n, ids): return np.minimum(3, 4*ids//n)

def value_dim(n, d, readout):
    return d if readout == 'full' else int(np.ceil(int(readout[3:])*np.log(n)))

def seed_rng(seed, d, load, a, family, stream):
    return np.random.default_rng(np.random.SeedSequence(
        [seed, d, load, int(2*a), FAMILIES.index(family), stream]))

def key_array(seed,d,load,a,family):
    return seed_rng(seed,d,load,a,family,0).standard_normal((d*load,d))*np.sqrt(spectrum(d,a,family)/d)

class Kernel:
    """One method/cell: diagonal eigensystem plus off-diagonal powers.

    Peak tensor is <=256*256*256 float64 (~128MiB), discarded per method.
    All times are integer steps; no continuum ODE and no fitted correction.
    """
    def __init__(self,d,load,a,family,method,ids=None):
        self.d=d; self.n=d*load; self.ids=query_ids(self.n) if ids is None else np.asarray(ids)
        self.delta=spectrum(d,a,family)/d; self.method=method
        if method=='hebb': return
        b=self.beta=float(method.split('_')[1]); delta=self.delta
        t=self.ids; s=self.n-1-self.ids
        F=np.diag(1-2*b*delta+2*b*b*delta**2)+b*b*np.outer(delta,delta)
        ev,P=np.linalg.eigh(F)
        assert np.all(ev>0)
        self.ev,self.P=ev,P
        self.power=np.exp(s[:,None]*np.log(ev))
        forcing=P.T@(b*b*delta)
        def trajectory(steps):
            steps=np.asarray(steps)
            den=ev-1
            series=np.empty((len(steps),d))
            nz=np.abs(den)>1e-13
            series[:,nz]=np.expm1(steps[:,None]*np.log(ev[nz]))/den[nz]
            series[:,~nz]=steps[:,None]
            return (series*forcing)@P.T
        self.rt,self.rs=trajectory(t),trajectory(s)
        self.Fdelta=(self.power*(P.T@delta))@P.T
        self.Fr=((self.rt@P)*self.power)@P.T
        self.diagF=self.power@(P*P).T
        gamma=1-b*(delta[:,None]+delta[None,:])+2*b*b*np.outer(delta,delta)
        self.G=np.power(gamma[None,:,:],s[:,None,None])
        self.G[:,np.arange(d),np.arange(d)]=0.
        self.Bdelta=self.Fdelta+self.G@delta
        self.Jpower=(1-b*delta)[None,:]**s[:,None]

    def conditional(self,Z):
        X=Z*Z; delta=self.delta
        if self.method=='hebb':
            m=X.sum(axis=1)
            return m,m*m+(self.n-1)*(X@delta),np.zeros(len(X))
        b=self.beta; Y=X@self.P; Yr=(X*self.rt)@self.P
        GX=np.einsum('qij,qj->qi',self.G,X,optimize=True)
        Bq=np.sum(Y*Y*self.power,axis=1)+np.sum(X*GX,axis=1)
        Br=np.sum(Y*Yr*self.power,axis=1)+np.sum(X*self.rt*GX,axis=1)
        m=b*np.sum(X*self.Jpower,axis=1)
        Q=np.sum(X*(self.Fr+self.rs),axis=1)-2*b*Br+b*b*(1+np.sum(X*self.rt,axis=1))*Bq
        vm=b*b*Bq-m*m
        assert np.all(Q>=m*m-1e-8*np.maximum(Q,1)), (self.method,Q.min())
        assert np.all(vm>=-1e-8*np.maximum(m*m,1)), (self.method,vm.min())
        return m,Q,np.maximum(vm,0.)  # Roundoff only; assertions precede this.

    def unconditional(self):
        delta=self.delta
        if self.method=='hebb':
            return np.ones(len(self.ids)),np.full(len(self.ids),1+(self.n+1)*(delta@delta))
        b=self.beta; r=self.rt; bd=self.Bdelta; e=self.diagF
        h=bd@delta+2*(e@(delta**2))
        Q=np.sum(self.Fdelta*r,axis=1)+self.rs@delta
        Q-=2*b*(np.sum(bd*r*delta,axis=1)+2*np.sum(e*r*delta**2,axis=1))
        Q+=b*b*((1+r@delta)*h+4*np.sum(bd*r*delta**2,axis=1)+8*np.sum(e*r*delta**3,axis=1))
        return b*(self.Jpower@delta),Q

def iid_integral(rho,n,p,order=96):
    rho=np.asarray(rho)
    assert np.max(np.abs(rho))<=1+1e-10
    rho=np.clip(rho,-1,1)
    z,wz=hermgauss(order); z=z*np.sqrt(2); wz=wz/np.sqrt(np.pi)
    x,wx=roots_genlaguerre(order,p/2-1)
    wx=np.exp(np.log(wx)-gammaln(p/2)); r=np.sqrt(2*x)
    result=[]
    for start in range(0,len(rho),16):
        h=rho[start:start+16,None,None]
        arg=h*r[None,:,None]+np.sqrt(1-h*h)*z[None,None,:]
        result.extend((ndtr(arg)**(n-1)@wz)@wx)
    return np.array(result)

class Readout:
    def __init__(self,path): self.tables=np.load(path)
    def __call__(self,rho,n,p):
        assert np.max(np.abs(rho))<=1+1e-8
        return np.interp(rho,RHO_GRID,self.tables[f'n{n}_p{p}'])

def oracle_cell(family,d,load,a):
    return family=='bounded' and d in ORACLE_DIMS and load==2 and a==1.5

def loose_readout_bound(U,ids,p,eta=.01):
    """FORMULATION (10), conservative c_n bound; report vacuity honestly."""
    n=U.shape[0]; W=U/np.linalg.norm(U,axis=0)
    rho=W[ids,np.arange(len(ids))]; wrong=W.copy(); wrong[ids,np.arange(len(ids))]=0
    r=np.sqrt(np.maximum(0.,1-rho*rho))
    eps=np.max(np.abs(wrong),axis=0)*(np.sqrt(p)+np.sqrt(2*np.log(3/eta))+
               (1+np.abs(rho))*np.sqrt(2*np.log(6/eta))/np.maximum(r,1e-300))
    return np.minimum(1.,eta+2*(n-1)/np.sqrt(2*np.pi)*eps)
