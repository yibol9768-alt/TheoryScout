"""Nonrigorous exploration; never used as certificate."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr
from numpy.polynomial.legendre import leggauss
x,w=leggauss(700); z=(x+1)*6; w=w*6*np.sqrt(2/np.pi)*np.exp(-z*z/2)
def moments(a,e,b1=.9,b2=.99):
 h=z/(z+e); f=1-a/(z+e)
 d=a*(h+z*(b1+f)/(1+b1)/(z*np.sqrt((b2+f*f)/(1+b2))+e))
 return np.sum(w*z*d),np.sum(w*d*d)
def signal(a,e,b1=.9,b2=.99):
 B,V=moments(a,e,b1,b2); q=1-2*B+2*V
 return B/np.sqrt(q),(1-B)/np.sqrt(q)
def first(u,e,b1=.9,b2=.99):
 last=0
 for a in np.linspace(.001,3,500):
  if signal(a,e,b1,b2)[0]>u:
   root=brentq(lambda a:signal(a,e,b1,b2)[0]-u,last,a)
   return root,signal(root,e,b1,b2)[1]
  last=a
 return None
def sign(a):
 B=2*a*np.sqrt(2/np.pi)*np.exp(-a*a/2);V=8*a*a*ndtr(-a);q=1-2*B+2*V
 return B/np.sqrt(q),(1-B)/np.sqrt(q)
for u in [.1,.2,.3,.4,.5,.6,.7,.8]:
 print('target',u)
 try:
  a=brentq(lambda a:sign(a)[0]-u,0,.6);print('sign',a,sign(a)[1])
 except ValueError:print('sign not bracketed')
 for e in [.01,.03,.05,.1,.2,.5,1,2,5]:print(e,first(u,e))
