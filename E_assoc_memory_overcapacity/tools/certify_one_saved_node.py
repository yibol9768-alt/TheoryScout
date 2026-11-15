"""Certify one CF integral for exactly represented floating coefficients.

Excludes error in matrix construction and eigensolution. Replays an old
scatter without new training draws, then treats each coefficient as rational.
"""
from fractions import Fraction as F
import argparse
import importlib.util
import json
import math
from pathlib import Path
import resource
import time

import numpy as np


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    resource.setrlimit(resource.RLIMIT_CPU,(180,185))
    gen=load('r3_generator',root/'tools/diagnose_joint_energy.py')
    moments=load('r5_moments',root/'tools/analyze_joint_gaussian.py')
    integer=load('r9_integer',root/'tools/certify_weighted_chisquare.py')
    old=json.loads((root/'revisions/r3_joint_diagnostic_v2/design.json').read_text())
    case=next(x for x in old['archived_loads'] if x['d']==512 and x['method']=='Adam2')
    saved=next(json.loads(x) for x in (root/'revisions/r8_random_thresholds/nodes.jsonl').read_text().splitlines()
               if json.loads(x)['L']==1 and json.loads(x)['index']==64)
    d,n,method=case['d'],case['n'],case['method'];t=saved['t'];m=n-1
    rng=gen.seed_rng(old['seed'],d,gen.METHODS.index(method),0,2)
    x,y=rng.normal(size=(2,d,n))/math.sqrt(d)
    u=gen.train(d/n*x@x.T,d/math.sqrt(n)*y@x.T,method)
    xy=np.vstack((x,y));chol=np.linalg.cholesky(d/n*xy@xy.T)
    b=chol[d:].T@u@chol[:d]/np.linalg.norm(u);k=(b+b.T)/2;nn=b.T@b
    mom=moments.moments(b,n);v=mom['expected_V'];mu=mom['mu'];alpha=t/(2*math.sqrt(v))
    h=(1+2*alpha*mu/m)*k-alpha*n/m*nn
    constant=-alpha*mu*mu/m-t*math.sqrt(v)/2
    eigen=np.linalg.eigvalsh(h);values=np.r_[eigen+constant/n,constant/n]
    coefficients=[F.from_float(float(a)) for a in values]
    counts=[1]*len(eigen)+[n-len(eigen)]
    args.out.mkdir(parents=True,exist_ok=False)
    input_record=dict(scope='Exact rational representation of computed float coefficients; eigenvalue and matrix errors excluded',
                      case=case,node_index=64,L=1,t=t,
                      coefficients=[str(a) for a in coefficients],counts=counts,
                      saved_float_probability=saved['lower']['probability'])
    (args.out/'input.json').write_text(json.dumps(input_record,indent=2)+'\n')
    print('Recorded',len(coefficients),'coefficient groups; starting integer integral',flush=True)
    del x,y,xy,chol,b,k,nn,u,h
    start=time.monotonic()
    record,enclosure=integer.certify(coefficients,counts,integer.pi_interval(),integer.gauss8())
    record.update(wall_seconds=time.monotonic()-start,
                  saved_float_inside=enclosure.contains(F.from_float(saved['lower']['probability'])),
                  caveat=input_record['scope'])
    (args.out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Complete:',record['probability']['approximate'],'time',record['wall_seconds'],flush=True)


if __name__=='__main__':main()
