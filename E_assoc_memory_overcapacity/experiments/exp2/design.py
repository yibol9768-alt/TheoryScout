"""EXP2 fixed design; local project files only, NumPy + standard library."""
from pathlib import Path
import math
import statistics
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/exp2'
N=statistics.NormalDist()
METRICS=('mean_rank','top0.01','top0.1','top0.5')
M=512
LOW_SEEDS=16
P2_SEEDS=112
HIGH_SEEDS=24
FOCI=((128,1.,.6,'top0.01'),(128,2.,.6,'top0.01'),
      (256,1.,1.5,'top0.01'),(256,2.,1.5,'top0.1'))
def cdf(z):return .5*math.erfc(-z/math.sqrt(2))
def phi(z):return math.exp(-z*z/2)/math.sqrt(2*math.pi)
def sign_c(a):return math.sqrt(2)*phi(a)/math.sqrt(cdf(-a))
def pred(c,k,metric):
    return cdf(-c/math.sqrt(2*k)) if metric=='mean_rank' else cdf(c/math.sqrt(k)-N.inv_cdf(1-float(metric[3:])))
def sign_q(a):return 8*a*a*cdf(-a)
def jump_B(a,k):return 8*a**3*phi(a)*math.sqrt(2/(math.pi*k))/sign_q(a)
def adam(z,s,p):
    import numpy as np
    u=np.zeros_like(z);mm=np.zeros_like(z);v=np.zeros_like(z)
    for t in range(1,p['T']+1):
        r=u-z if s is None else u@s-z
        mm=p['b1']*mm+(1-p['b1'])*r;v=p['b2']*v+(1-p['b2'])*r*r
        u-=.2*(mm/(1-p['b1']**t))/(np.sqrt(v/(1-p['b2']**t))+p['e'])
    return u
