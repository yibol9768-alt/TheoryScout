"""Certify one anchored lower event from an exact represented score block B.

The input boundary is B, NOT the original Gaussian scatter or optimizer.
Anchors and threshold are exact rational numbers; no exact moments are needed.
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
import sys
import time

import numpy as np


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def float_bounds(value):
    low=high=float(value)
    while F.from_float(low)>value:low=math.nextafter(low,-math.inf)
    while F.from_float(high)<value:high=math.nextafter(high,math.inf)
    return low,high


def construct(b,h,A,D,executable,out,label):
    h=np.tril(h)+np.tril(h,-1).T
    path=out/f'{label}_construction.bin'
    with path.open('xb') as stream:
        stream.write(struct.pack('<Q',len(b)))
        for array in (b,h):stream.write(np.asarray(array,dtype='<f8').tobytes(order='C'))
        stream.write(struct.pack('<4d',*float_bounds(A),*float_bounds(D)))
    result=subprocess.check_output([str(executable),str(path)],text=True)
    (out/f'{label}_construction.txt').write_text(result)
    fields=dict(line.split(maxsplit=1) for line in result.splitlines())
    assert fields['rounding_checks']=='true'
    epsilon=len(b)*F.from_float(float.fromhex(fields['max_entry_error_hex']))
    return h,epsilon,dict(input_sha256=sha(path),scalar_A=str(A),scalar_D=str(D),
                          operator_error=str(epsilon),rounding_output=fields)


def main():
    global sha
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    if platform.machine()!='x86_64' or sys.byteorder!='little':
        raise RuntimeError('The rounding build/input format requires little-endian x86_64')
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    spec=load('r10_spectral',root/'tools/certify_spectral_residual.py')
    integer=load('r9_integer',root/'tools/certify_weighted_chisquare.py')
    sha=spec.sha;save=spec.save;I=integer.I
    build=json.loads((root/'revisions/r10_spectral_certificate/BUILD.json').read_text())
    flags=build['command'][1:8]
    executables={}
    for stem in ['rounding_anchor_construction','rounding_spectral_residual']:
        source=root/'tools'/f'{stem}.cpp';target=args.out/stem
        command=['g++',*flags,str(source),'-o',str(target)]
        subprocess.run(command,check=True)
        executables[stem]=target
        save(args.out/f'{stem}_BUILD.json',dict(command=command,source_sha256=sha(source),executable_sha256=sha(target),
             compiler=build['compiler'],assumptions=build['assumptions']))
    # Exact rational reference checks for the construction kernel.
    tests=[]
    for label,A,D,b in [('toy_positive',F(7,3),F(5,7),np.array([[.5,-.25],[.75,.125]])),
                        ('toy_signed',F(-5,3),F(-9,7),np.array([[-.125,.5],[.25,-.75]]))]:
        h=float(A)*(b+b.T)/2-float(D)*(b.T@b)
        h,error,record=construct(b,h,A,D,executables['rounding_anchor_construction'],args.out,label)
        exact=[]
        for i in range(2):
            for j in range(2):
                k=(F(float(b[i,j]))+F(float(b[j,i])))/2
                nn=sum(F(float(b[l,i]))*F(float(b[l,j])) for l in range(2))
                exact.append(abs(A*k-D*nn-F(float(h[i,j]))))
        assert max(exact)<=error/2
        tests.append(dict(case=label,exact_entry_bounds=True,**record))
    save(args.out/'toy_checks.json',tests)
    generator=load('r3_generator',root/'tools/diagnose_joint_energy.py')
    moments=load('r5_moments',root/'tools/analyze_joint_gaussian.py')
    design=json.loads((root/'revisions/r3_joint_diagnostic_v2/design.json').read_text())
    saved=next(json.loads(s) for s in (root/'revisions/r8_random_thresholds/nodes.jsonl').read_text().splitlines()
               if json.loads(s)['L']==1 and json.loads(s)['index']==64)
    cell=next(c for c in design['archived_loads'] if c['d']==512 and c['method']=='Adam2')
    d,n,method=cell['d'],cell['n'],cell['method'];m=n-1
    rng=generator.seed_rng(design['seed'],d,generator.METHODS.index(method),0,2)
    x,y=rng.normal(size=(2,d,n))/math.sqrt(d)
    u=generator.train(d/n*x@x.T,d/math.sqrt(n)*y@x.T,method)
    xy=np.vstack((x,y));factor=np.linalg.cholesky(d/n*xy@xy.T)
    b=factor[d:].T@u@factor[:d]/np.linalg.norm(u)
    mom=moments.moments(b,n)
    # These floats select anchors only; their moment interpretation is unnecessary.
    t=F.from_float(saved['t']);a=F.from_float(mom['mu'])
    alpha=F.from_float(saved['t']/(2*math.sqrt(mom['expected_V'])))
    energy_anchor=t/(2*alpha)
    A=4*alpha*m+8*alpha*alpha*a;D=4*alpha*alpha*n
    constant=-4*alpha*alpha*a*a-m*t*t
    save(args.out/'SCOPE.json',dict(n=n,block_dimension=len(b),threshold=str(t),center_anchor=str(a),
         alpha=str(alpha),sqrt_energy_anchor=str(energy_anchor),positive_scaling=str(4*alpha*m),
         exact_constant=str(constant),source='R3 replay selects an exact binary64 B input; no model-construction certificate',
         caveat='Certifies the anchored lower event for exact input B; not R8 original scatter/optimizer or complete recall'))
    h=float(A)*(b+b.T)/2-float(D)*(b.T@b)
    start=time.monotonic()
    h,construction,record=construct(b,h,A,D,executables['rounding_anchor_construction'],args.out,'actual')
    save(args.out/'CONSTRUCTION.json',record)
    print('Construction operator error:',float(construction),flush=True)
    eigen,spectral,spectrum=spec.matrix_certificate(h,executables['rounding_spectral_residual'],args.out,'actual',integer)
    epsilon=construction+spectral
    save(args.out/'ERRORS.json',dict(construction=str(construction),spectral=str(spectral),total=str(epsilon)))
    base=[F.from_float(float(v))+constant/n for v in eigen]
    counts=[1]*len(eigen)+[n-len(eigen)]
    pi=integer.pi_interval();nodes=integer.gauss8();bounds=[]
    for label,sign in [('lower',-1),('upper',1)]:
        coefficients=[v+sign*epsilon for v in base]+[constant/n]
        result,enclosure=integer.certify(coefficients,counts,pi,nodes)
        save(args.out/f'probability_{label}.json',result);bounds.append(enclosure)
        print('Completed',label,'integer integral',flush=True)
    interval=I.raw(bounds[0].lo,bounds[1].hi)
    save(args.out/'RESULT.json',dict(probability=interval.record(),wall_seconds=time.monotonic()-start,
         certified_from='Exact saved B, center, alpha, threshold; matrix construction + spectrum + integral',
         caveat='Single anchored lower event, no original model/optimizer certification and no full threshold integration'))


if __name__=='__main__':main()
