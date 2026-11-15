"""Directed-rounding residual certificate for a represented symmetric matrix.

The spectral error theorem uses the polar factor of an approximate eigenbasis;
it does not require eigengaps or exact orthogonality of that basis.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
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
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as handle:
        json.dump(value,handle,indent=2);handle.write('\n')


def matrix_certificate(h,executable,out,prefix,integer):
    h=np.tril(h)+np.tril(h,-1).T
    eigen,q=np.linalg.eigh(h);n=len(eigen)
    binary=out/f'{prefix}.bin'
    with binary.open('xb') as stream:
        stream.write(struct.pack('<Q',n))
        for array in (h,q,eigen):stream.write(np.asarray(array,dtype='<f8').tobytes(order='C'))
    result=subprocess.run([str(executable),str(binary)],text=True,capture_output=True,check=True)
    (out/f'{prefix}_rounding.txt').write_text(result.stdout)
    fields=dict(line.split(maxsplit=1) for line in result.stdout.splitlines())
    assert fields['rounding_checks']=='true'
    delta=F.from_float(float.fromhex(fields['max_gram_hex']))*n
    residual=F.from_float(float.fromhex(fields['max_residual_hex']))*n
    assert delta<1
    I=integer.I
    norm=max(F.from_float(abs(float(v))) for v in eigen)
    inverse=1/(1-I(delta)).sqrt()
    error=I(residual)*inverse+2*I(norm)*(1+I(delta)).sqrt()*(inverse-1)
    epsilon=F(error.hi,integer.SCALE)
    record=dict(dimension=n,matrix_scope='Exact interpretation of lower-triangle symmetric float input',
                input_sha256=sha(binary),rounding_output=fields,gram_operator_bound=str(delta),
                residual_operator_bound=str(residual),eigenvalue_error=error.record(),
                epsilon_upper=str(epsilon),eigenvalues=[str(F.from_float(float(v))) for v in eigen])
    save(out/f'{prefix}_spectrum.json',record)
    return eigen,epsilon,record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if platform.machine()!='x86_64':raise RuntimeError('This directed-rounding driver is restricted to x86_64')
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    integer=load('r9_integer',root/'tools/certify_weighted_chisquare.py')
    source=root/'tools/rounding_spectral_residual.cpp';executable=args.out/'rounding_spectral_residual'
    command=['g++','-O2','-std=c++17','-frounding-math','-ffp-contract=off','-fno-fast-math',
             '-mfpmath=sse','-msse2',str(source),'-o',str(executable)]
    subprocess.run(command,check=True)
    save(args.out/'BUILD.json',dict(command=command,source_sha256=sha(source),executable_sha256=sha(executable),
        compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
        assumptions='IEEE binary64 directed rounding, SSE2, no FTZ/DAZ; preflight checked, not a formal verification of compiler/hardware'))
    had=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],dtype=int)
    tests=[]
    for index,values in enumerate(([-.5,.25,.75,1.5],[-.5,.25,.25+2**-30,1.5])):
        exact=[F.from_float(x) for x in values]
        h=np.empty((4,4))
        for i in range(4):
            for j in range(4):
                value=sum(F(int(had[i,k]*had[j,k]),4)*exact[k] for k in range(4))
                h[i,j]=float(value);assert F.from_float(float(h[i,j]))==value
        approximate,error,record=matrix_certificate(h,executable,args.out,f'toy_{index}',integer)
        assert all(abs(F.from_float(float(v))-target)<=error for v,target in zip(approximate,sorted(exact)))
        tests.append(dict(case=index,contains_known_exact_eigenvalues=True,error=str(error)))
    save(args.out/'toy_checks.json',tests)
    generator=load('r3_generator',root/'tools/diagnose_joint_energy.py')
    moments=load('r5_moments',root/'tools/analyze_joint_gaussian.py')
    old=json.loads((root/'revisions/r3_joint_diagnostic_v2/design.json').read_text())
    saved=next(json.loads(s) for s in (root/'revisions/r8_random_thresholds/nodes.jsonl').read_text().splitlines()
               if json.loads(s)['L']==1 and json.loads(s)['index']==64)
    cell=next(c for c in old['archived_loads'] if c['d']==512 and c['method']=='Adam2')
    d,n,method=cell['d'],cell['n'],cell['method'];m=n-1;t=saved['t']
    rng=generator.seed_rng(old['seed'],d,generator.METHODS.index(method),0,2)
    x,y=rng.normal(size=(2,d,n))/math.sqrt(d)
    u=generator.train(d/n*x@x.T,d/math.sqrt(n)*y@x.T,method)
    xy=np.vstack((x,y));f=np.linalg.cholesky(d/n*xy@xy.T)
    b=f[d:].T@u@f[:d]/np.linalg.norm(u);k=(b+b.T)/2;nn=b.T@b
    mom=moments.moments(b,n);v=mom['expected_V'];mu=mom['mu'];alpha=t/(2*math.sqrt(v))
    h=(1+2*alpha*mu/m)*k-alpha*n/m*nn;constant=-alpha*mu*mu/m-t*math.sqrt(v)/2
    save(args.out/'ACTUAL_SCOPE.json',dict(case=cell,L=1,node_index=64,t=t,constant=str(F.from_float(constant)),
         caveat='Only the represented symmetric H and constant are inputs; construction/optimizer errors excluded'))
    print('Starting directed residual on represented 1024x1024 matrix',flush=True)
    eigen,error,spectrum=matrix_certificate(h,executable,args.out,'actual',integer)
    print('Certified spectral epsilon <=',float(error),flush=True)
    base=[F.from_float(float(v))+F.from_float(constant)/n for v in eigen]
    counts=[1]*len(eigen)+[n-len(eigen)]
    pi=integer.pi_interval();nodes=integer.gauss8();bounds=[];start=time.monotonic()
    for label,sign in (('lower',-1),('upper',1)):
        coefficients=[a+sign*error for a in base]+[F.from_float(constant)/n]
        record,enclosure=integer.certify(coefficients,counts,pi,nodes)
        save(args.out/f'probability_{label}.json',record)
        bounds.append(enclosure)
        print('Completed',label,'integer integration',flush=True)
    final=integer.I.raw(bounds[0].lo,bounds[1].hi)
    save(args.out/'RESULT.json',dict(scope='Spectral residual plus integer integration for represented symmetric matrix and exact constant',
         probability=final.record(),spectral_error_upper=str(error),
         integration_wall_seconds=time.monotonic()-start,
         caveat='Not a certificate of model/optimizer/Cholesky/normalization construction or all R8 nodes'))


if __name__=='__main__':main()
