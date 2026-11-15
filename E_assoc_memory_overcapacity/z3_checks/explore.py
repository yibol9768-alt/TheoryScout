"""Exploratory quadrature only; no certified numerical claims."""
import os
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

assert all(int(os.environ.get(k, "999")) <= 2 for k in
           ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"])
cgroup = Path("/sys/fs/cgroup") / Path(next(line.split(":", 2)[2].strip().lstrip("/") for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")))
assert (cgroup / "memory.max").read_text().strip() == "3221225472"
assert (cgroup / "memory.swap.max").read_text().strip() == "0"

def moments(a, e, b1, b2):
    def d(z):
        f = 1-a/(z+e)
        return a*z*(1/(z+e)+(1-a/(z+e)/(1+b1))/(z*math.sqrt((b2+f*f)/(1+b2))+e))
    def integral(kind):
        return quad(lambda z: (z*d(z) if kind == 0 else d(z)**2)*math.sqrt(2/math.pi)*math.exp(-z*z/2), 0, 12, points=[a] if 0<a<12 else None, epsabs=2e-10, limit=150)[0]
    return integral(0), integral(1)

def match(e,b1,b2,r,u):
    def signal(a):
        B,V=moments(a,e,b1,b2)
        q=1-2*(1-r)*B+2*(1-r)*V
        return (r+(1-r)*B)/math.sqrt(q)
    previous=0
    for a in np.geomspace(1e-5, 15, 90):
        if signal(a)>u:
            root=brentq(lambda x:signal(x)-u, previous,a,xtol=1e-11)
            B,V=moments(root,e,b1,b2)
            return root,(1-(1-r)*B)/math.sqrt(1-2*(1-r)*B+2*(1-r)*V)
        previous=a
    return None

if __name__ == "__main__":
    for e in [.001,.01,.1,1,10]:
        for u in [.1,.4,.6,.8]:
            rows=[match(e,b1,b2,0,u) for b1,b2 in [(0,.99),(.9,.99),(0,0),(.9,0)]]
            print(e,u,rows,flush=True)
