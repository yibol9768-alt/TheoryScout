"""Replay the represented-B certificate with standard-library Python only.

Optional kernels use the saved, hashed x86_64 binaries. Integral replay does not
require either NumPy or SciPy. The input boundary remains the recorded B.
"""
from array import array
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import struct
import subprocess
import sys

import certify_weighted_chisquare as integer
from check_spectral_certificate import binary_arrays, endpoints, read, sha


def construction_arrays(path):
    raw=path.read_bytes();q,=struct.unpack('<Q',raw[:8])
    values=array('d');values.frombytes(raw[8:])
    if sys.byteorder!='little':values.byteswap()
    assert len(values)==2*q*q+4
    return q,values[:q*q],values[q*q:2*q*q],values[2*q*q:]


def repeat(directory,out,stem,path,expected):
    text=subprocess.check_output([str(directory/stem),str(path)],text=True)
    (out/f'{path.stem}_{stem}_repeat.txt').write_text(text)
    fields=dict(line.split(maxsplit=1) for line in text.splitlines())
    assert all(fields[key]==value for key,value in expected.items() if key!='wall_seconds')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeat-kernels',action='store_true')
    parser.add_argument('--reintegrate',action='store_true');args=parser.parse_args()
    directory=args.input.resolve();args.out.mkdir(parents=True,exist_ok=False)
    for stem in ['rounding_anchor_construction','rounding_spectral_residual']:
        build=read(directory/f'{stem}_BUILD.json')
        assert sha(directory/stem)==build['executable_sha256']
        assert sha(Path(__file__).with_name(f'{stem}.cpp'))==build['source_sha256']
    scope=read(directory/'SCOPE.json');n=scope['n'];m=n-1
    a,alpha,t=map(F,[scope['center_anchor'],scope['alpha'],scope['threshold']])
    assert alpha>0 and t>0 and F(scope['sqrt_energy_anchor'])==t/(2*alpha)
    assert F(scope['positive_scaling'])==4*alpha*m
    A=4*alpha*m+8*alpha*alpha*a;D=4*alpha*alpha*n
    constant=-4*alpha*alpha*a*a-m*t*t
    assert constant==F(scope['exact_constant'])
    inputs=[('actual',read(directory/'CONSTRUCTION.json'))]
    inputs += [(test['case'],test) for test in read(directory/'toy_checks.json')]
    for label,record in inputs:
        path=directory/f'{label}_construction.bin'
        assert sha(path)==record['input_sha256']
        q,b,h,scalars=construction_arrays(path)
        # With q<=4096 these caps keep every kernel intermediate below 2^620.
        # In particular no infinity-minus-infinity NaN can evade a max check.
        assert 0<q<=4096 and all(abs(v)<=2.**200 for array in (b,h,scalars) for v in array)
        aa,dd=F(record['scalar_A']),F(record['scalar_D'])
        loA,hiA,loD,hiD=map(F,scalars)
        assert loA<=aa<=hiA and loD<=dd<=hiD
        assert all(h[i*q+j]==h[j*q+i] for i in range(q) for j in range(q))
        error=q*F.from_float(float.fromhex(record['rounding_output']['max_entry_error_hex']))
        assert error==F(record['operator_error'])
        if label=='actual':
            assert (aa,dd)==(A,D) and q==scope['block_dimension']
            actual_h=h
        else:
            assert q==2
            for i in range(q):
                for j in range(q):
                    k=(F(b[i*q+j])+F(b[j*q+i]))/2
                    nn=sum(F(b[l*q+i])*F(b[l*q+j]) for l in range(q))
                    assert abs(aa*k-dd*nn-F(h[i*q+j]))<=error/q
        if args.repeat_kernels:
            repeat(directory,args.out,'rounding_anchor_construction',path,record['rounding_output'])
    spectrum=read(directory/'actual_spectrum.json')
    path=directory/'actual.bin';assert sha(path)==spectrum['input_sha256']
    q,h,basis,eigen=binary_arrays(path)
    assert all(abs(v)<=2.**200 for array in (h,basis,eigen) for v in array)
    assert actual_h==h and list(eigen)==sorted(eigen)
    assert list(map(F,eigen))==list(map(F,spectrum['eigenvalues']))
    fields=spectrum['rounding_output']
    delta=q*F.from_float(float.fromhex(fields['max_gram_hex']))
    rho=q*F.from_float(float.fromhex(fields['max_residual_hex']))
    assert delta==F(spectrum['gram_operator_bound'])<1
    assert rho==F(spectrum['residual_operator_bound'])
    inverse=1/(1-integer.I(delta)).sqrt()
    bound=integer.I(rho)*inverse+2*integer.I(max(map(abs,map(F,eigen))))*(1+integer.I(delta)).sqrt()*(inverse-1)
    assert spectrum['eigenvalue_error']==bound.record()
    spectral=F(bound.hi,integer.SCALE)
    assert spectral==F(spectrum['epsilon_upper'])
    errors=read(directory/'ERRORS.json');construction=F(read(directory/'CONSTRUCTION.json')['operator_error'])
    assert F(errors['construction'])==construction and F(errors['spectral'])==spectral
    epsilon=construction+spectral;assert epsilon==F(errors['total'])
    if args.repeat_kernels:
        repeat(directory,args.out,'rounding_spectral_residual',path,fields)
    intervals=[]
    for label,sign in [('lower',-1),('upper',1)]:
        saved=read(directory/f'probability_{label}.json')
        coefficients=[F(v)+constant/n+sign*epsilon for v in eigen]+[constant/n]
        counts=[1]*q+[n-q]
        assert list(map(F,saved['coefficients']))==coefficients and saved['counts']==counts
        if args.reintegrate:
            result,_=integer.certify(coefficients,counts,integer.pi_interval(),integer.gauss8())
            assert result==saved
            (args.out/f'probability_{label}_repeat.json').write_text(json.dumps(result,indent=2)+'\n')
        intervals.append(endpoints(saved['probability']))
    final=read(directory/'RESULT.json')
    assert endpoints(final['probability'])==(intervals[0][0],intervals[1][1])
    result=dict(binary_inputs_checked=4,exact_construction_reference_cases=2,
                exact_scalar_and_constant_checks=True,exact_coefficient_checks=True,
                repeated_kernels=args.repeat_kernels,reintegrated=args.reintegrate,
                probability=final['probability'],input_boundary='Recorded B; original model not certified')
    (args.out/'CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
