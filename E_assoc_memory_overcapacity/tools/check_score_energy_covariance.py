"""Deterministic checks of the exact conditional score/energy covariance.

Small-sphere polynomial quadrature and replay of rep=0 from all six R3 cells.
The replay reuses frozen training draws; this is a post hoc mathematical check.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre


def moments(b, n):
    k=(b+b.T)/2;mu=float(np.trace(k));k2=float(np.sum(k*k))
    k3=float(np.sum((k@k)*k))
    sigma2=2*n/(n+2)*(k2-mu*mu/n)
    central3=8*n*n/((n+2)*(n+4))*(k3-3*mu/n*k2+2*mu**3/n**2)
    norm2=float(np.sum(b*b));kn=float(np.sum((b@k)*b))
    ev=(n*norm2-mu*mu-sigma2)/(n-1)
    covariance=(2*n*n/((n-1)*(n+2))*(kn-mu*norm2/n)
                -(2*mu*sigma2+central3)/(n-1))
    return dict(mu=mu,sigma2=sigma2,expected_V=ev,covariance=covariance)


def quadrature():
    records=[]
    for n in (2,3):
        theta=np.arange(128)*2*math.pi/128
        if n==2:
            w=np.stack((np.cos(theta),np.sin(theta)))
            weights=np.full(128,1/128)
        else:
            z,weights_z=roots_legendre(16)
            radius=np.sqrt(1-z*z)
            w=np.stack(((radius[:,None]*np.cos(theta)).ravel(),
                        (radius[:,None]*np.sin(theta)).ravel(),
                        np.repeat(z,len(theta))))
            weights=np.repeat(weights_z/2/len(theta),len(theta))
        matrices=[np.arange(1,n*n+1,dtype=float).reshape(n,n)/7,
                  np.diag(np.linspace(-.6,.8,n))]
        for index,b in enumerate(matrices):
            raw=b@w;correct=n*np.sum(w*raw,axis=0)
            energy=(n*n*np.sum(raw*raw,axis=0)-correct**2)/(n-1)
            covariance=float(np.sum(weights*correct*energy)
                             -np.sum(weights*correct)*np.sum(weights*energy))
            formula=moments(b,n)["covariance"]
            error=abs(covariance-formula)
            if error>1e-10:
                raise RuntimeError("Sphere moment identity failed")
            records.append(dict(n=n,matrix=index,quadrature=covariance,formula=formula,error=error))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    source=Path(__file__).with_name("diagnose_joint_energy.py")
    design=json.loads((args.run/"design.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest()!=design["source_sha256"]:
        raise RuntimeError("Frozen generator source changed")
    spec=importlib.util.spec_from_file_location("frozen_r3_generator",source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    for cell in design["archived_loads"]:
        d,n,method=cell["d"],cell["n"],cell["method"]
        rng=module.seed_rng(design["seed"],d,module.METHODS.index(method),0,2)
        x,y=rng.normal(size=(2,d,n))/math.sqrt(d)
        u=module.train(d/n*x@x.T,d/math.sqrt(n)*y@x.T,method)
        xy=np.vstack((x,y));f=np.linalg.cholesky(d/n*xy@xy.T)
        b=f[d:].T@u@f[:d]/np.linalg.norm(u)
        exact=moments(b,n)
        with np.load(args.run/f"joint_{d}_{method}_0.npz") as archive:
            for key in ("mu","sigma2","expected_V"):
                if abs(exact[key]-float(archive[key]))>1e-10:
                    raise RuntimeError("Replay differs from frozen run")
            observed=float(np.cov(archive["D"],archive["V"],ddof=1)[0,1])
        rows.append(dict(**cell,rep=0,exact=exact,sampled_covariance=observed))
        del x,y,xy,u,f,b
    output=dict(scope="Post hoc moment check on fixed R3 draws; covariance samples are descriptive",
                polynomial_quadrature=quadrature(),replay=rows)
    with (args.run/"covariance_checks.json").open("x") as handle:
        json.dump(output,handle,indent=2);handle.write("\n")
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
