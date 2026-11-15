"""Certify factor and ideal Adam2 residuals for the replayed R8 scatter C0.

C0 is explicitly the lower-triangle symmetric dyadic interpretation, not an
unrounded Gaussian realization. No new training or threshold samples are drawn.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import struct
import subprocess
import time

import numpy as np
from certify_factor_adam2 import load


def save(path,value):
    with path.open('x') as handle:json.dump(value,handle,indent=2);handle.write('\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(out,label,c,inverse,f,u,n,executable):
    path=out/f'{label}.bin'
    with path.open('xb') as stream:
        stream.write(struct.pack('<QQ',len(u),n))
        for array in (c,inverse,f,u):stream.write(np.asarray(array,dtype='<f8').tobytes(order='C'))
    start=time.monotonic()
    result=subprocess.check_output([str(executable),str(path),str(out/f'{label}_endpoint_intervals.bin')],text=True)
    (out/f'{label}_rounding.txt').write_text(result)
    fields=dict(line.split(maxsplit=1) for line in result.splitlines())
    save(out/f'{label}_AUDIT.json',dict(input_sha256=sha(path),rounding_output=fields,wall_seconds=time.monotonic()-start))
    print(label,fields,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if platform.machine()!='x86_64':raise RuntimeError('Requires x86_64')
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    sources=[root/'tools'/name for name in ['rounding_scatter_adam2.cpp','rounding_adam2_endpoint.cpp']]
    executable=args.out/'rounding_scatter_adam2'
    command=['g++','-O2','-std=c++17','-frounding-math','-ffp-contract=off','-fno-fast-math','-mfpmath=sse','-msse2',str(sources[0]),'-o',str(executable)]
    subprocess.run(command,check=True)
    save(args.out/'BUILD.json',dict(command=command,source_sha256={p.name:sha(p) for p in sources},executable_sha256=sha(executable),
         compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],assumptions='IEEE binary64 directed arithmetic/SSE2/no FTZ-DAZ; not formal compiler verification'))
    f=np.array([[1,0,0,0],[.25,1,0,0],[.5,.125,1,0],[-.25,.5,.25,1.]])
    c=f@f.T+np.eye(4)*2**-20
    run(args.out,'toy',c,np.linalg.inv(f),f,np.zeros((2,2)),16,executable)
    generator=load('r3_generator',root/'tools/diagnose_joint_energy.py')
    previous=load('r12_inputs',root/'tools/check_factor_adam2.py')
    d,n,fp,up,_,_,_=previous.inputs(root/'revisions/r12_factor_adam2/actual.bin');q=2*d
    f=np.asarray(fp).reshape(q,q);u=np.asarray(up).reshape(d,d)
    design=json.loads((root/'revisions/r3_joint_diagnostic_v2/design.json').read_text())
    rng=generator.seed_rng(design['seed'],d,generator.METHODS.index('Adam2'),0,2)
    x,y=rng.normal(size=(2,d,n))/math.sqrt(d);xy=np.vstack((x,y))
    raw=d/n*xy@xy.T;c=np.tril(raw)+np.tril(raw,-1).T
    assert np.array_equal(np.linalg.cholesky(c),f)
    run(args.out,'actual',c,np.linalg.solve(f,np.eye(q)),f,u,n,executable)
    save(args.out/'SCOPE.json',dict(d=d,n=n,conditioning='Exact dyadic C0: lower-triangle symmetric interpretation of R8 replay scatter input',
         factor_identical_to_r12=True,optimizer='Ideal Adam2 with exact lr3/10 beta1=9/10 beta2=99/100 epsilon1',
         caveat='C0 is the represented input, not exact unrounded Gaussian draw; only one fixed-threshold lower event is targeted'))


if __name__=='__main__':main()
