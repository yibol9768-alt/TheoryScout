"""Nonrigorous parameter hints for a state-reset ablation."""
from scipy.integrate import quad
from scipy.optimize import brentq
import math
for e in [.5,1]:
 for b1,b2 in [(0,0),(.9,0),(0,.99),(.9,.99)]:
  def mom(a):
   def d(z):
    h=z/(z+e);f=1-a/(z+e)
    return a*(h+z*(1-a/(z+e)/(1+b1))/(z*math.sqrt((b2+f*f)/(1+b2))+e))
   B=quad(lambda z:z*d(z)*math.sqrt(2/math.pi)*math.exp(-z*z/2),0,12,epsabs=1e-12)[0]
   V=quad(lambda z:d(z)**2*math.sqrt(2/math.pi)*math.exp(-z*z/2),0,12,epsabs=1e-12)[0]
   return B,V
  def H(a):
   B,V=mom(a);return B/math.sqrt(1-2*B+2*V)-.4
  a=brentq(H,.1,e*.95);B,V=mom(a)
  print(e,b1,b2,a,(1-B)/math.sqrt(1-2*B+2*V),flush=True)
