"""Audit ideal Adam2 at the explicit exact scatter C*=F F^T.

This is a new conditioning point, not certification of the old R8 scatter.
No fresh training datasets or threshold samples are generated.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import importlib.util
import json
import math
import platform
import struct
import subprocess
import time

import numpy as np


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write_input(path,factor,u,left,b,scale,n):
    d=len(u)
    with path.open('xb') as stream:
        stream.write(struct.pack('<QQ',d,n))
        for array in (factor,u,left,b):stream.write(np.asarray(array,dtype='<f8').tobytes(order='C'))
        stream.write(struct.pack('<d',scale))


def audit(executable,path,out,label,save):
    start=time.monotonic()
    text=subprocess.check_output([str(executable),str(path),str(out/f'{label}_endpoint_intervals.bin')],text=True)
    (out/f'{label}_rounding.txt').write_text(text)
    fields=dict(line.split(maxsplit=1) for line in text.splitlines())
    assert fields['rounding_checks']=='true'
    print(label,fields,flush=True)
    return fields,time.monotonic()-start


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if platform.machine()!='x86_64':raise RuntimeError('Requires x86_64')
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    common=load('r10_common',root/'tools/certify_spectral_residual.py');save=common.save;sha=common.sha
    integer=load('r9_integer',root/'tools/certify_weighted_chisquare.py')
    old=load('r11_check',root/'tools/check_anchor_certificate.py')
    source=root/'tools/rounding_adam2_endpoint.cpp';executable=args.out/'rounding_adam2_endpoint'
    command=['g++','-O2','-std=c++17','-frounding-math','-ffp-contract=off','-fno-fast-math','-mfpmath=sse','-msse2',str(source),'-o',str(executable)]
    subprocess.run(command,check=True)
    save(args.out/'BUILD.json',dict(command=command,source_sha256=sha(source),executable_sha256=sha(executable),
         compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
         assumptions='IEEE binary64 directed arithmetic/SSE2/no FTZ-DAZ; sqrt guesses checked by directed squares; not formal compiler verification'))
    f=np.array([[1,0,0,0],[.25,1,0,0],[.5,.125,1,0],[-.25,.5,.25,1.]])
    toy=args.out/'toy.bin';write_input(toy,f,np.zeros((2,2)),np.zeros((4,2)),np.zeros((4,4)),1.,16)
    fields,_=audit(executable,toy,args.out,'toy',save)
    save(args.out/'toy_run.json',dict(input_sha256=sha(toy),rounding_output=fields))
    generator=load('r3_generator',root/'tools/diagnose_joint_energy.py')
    design=json.loads((root/'revisions/r3_joint_diagnostic_v2/design.json').read_text())
    cell=next(c for c in design['archived_loads'] if c['d']==512 and c['method']=='Adam2')
    d,n,method=cell['d'],cell['n'],cell['method']
    rng=generator.seed_rng(design['seed'],d,generator.METHODS.index(method),0,2)
    x,y=rng.normal(size=(2,d,n))/math.sqrt(d)
    u=generator.train(d/n*x@x.T,d/math.sqrt(n)*y@x.T,method)
    xy=np.vstack((x,y));factor=np.linalg.cholesky(d/n*xy@xy.T)
    left=factor[d:].T@u;scale=float(np.linalg.norm(u));b=left@factor[:d]/scale
    q,oldb,_,_=old.construction_arrays(root/'revisions/r11_anchor_certificate_v2/actual_construction.bin')
    assert np.array_equal(b,np.asarray(oldb).reshape(q,q))
    path=args.out/'actual.bin';write_input(path,factor,u,left,b,scale,n)
    save(args.out/'SCOPE.json',dict(d=d,n=n,conditioning='Exact Cstar=F F^T from saved triangular binary64 F',
         optimizer='Ideal Adam2: lr=3/10 beta1=9/10 beta2=99/100 epsilon=1, bias corrections exact',
         source='Archived R3 replay selects F and approximations; Cstar is a new conditioning point',
         caveat='Does not certify the original R8 scatter or full random-threshold recall',input_sha256=sha(path),
         r11_block_identical=True))
    fields,wall=audit(executable,path,args.out,'actual',save)
    save(args.out/'AUDIT.json',dict(rounding_output=fields,wall_seconds=wall,input_sha256=sha(path)))
    print('Endpoint audit completed; run tools/check_factor_adam2.py to derive eta and reference checks',flush=True)


if __name__=='__main__':main()
