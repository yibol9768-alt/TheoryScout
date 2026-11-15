"""Standard-library construction/spectral audit for the new D/W matrices."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import json
import subprocess

import certify_weighted_chisquare as integer
from check_anchor_certificate import construction_arrays
from check_spectral_certificate import binary_arrays,read,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];scope=read(args.input/'SCOPE.json')
    for stem in ['rounding_anchor_construction','rounding_spectral_residual']:
        build=read(args.input/(stem+'_BUILD.json'));assert sha(args.input/stem)==build['executable_sha256']
        assert sha(root/'tools'/(stem+'.cpp'))==build['source_sha256']
    old=(root/'revisions/r11_anchor_certificate_v2/actual_construction.bin').read_bytes();q=scope['q']
    for case in scope['cases']:
        directory=args.input/case['label'];record=read(directory/'CONSTRUCTION.json');path=directory/'actual_construction.bin'
        assert sha(path)==record['input_sha256'] and path.read_bytes()[8:8+8*q*q]==old[8:8+8*q*q]
        cq,b,h,scalars=construction_arrays(path);assert cq==q and (F(record['scalar_A']),F(record['scalar_D']))==(F(case['A']),F(case['Dcoef']))
        assert all(abs(v)<=2.**200 for group in (b,h,scalars) for v in group) and q<=4096
        assert F(scalars[0])<=F(case['A'])<=F(scalars[1]) and F(scalars[2])<=F(case['Dcoef'])<=F(scalars[3])
        assert F(record['operator_error'])==q*F.from_float(float.fromhex(record['rounding_output']['max_entry_error_hex']))
        text=subprocess.check_output([str(args.input.resolve()/'rounding_anchor_construction'),str(path.resolve())],text=True)
        assert dict(line.split(maxsplit=1) for line in text.splitlines())==record['rounding_output']
        (args.out/(case['label']+'_construction_repeat.txt')).write_text(text)
        spectrum=read(directory/'actual_spectrum.json');matrix=directory/'actual.bin';assert sha(matrix)==spectrum['input_sha256']
        nq,matrix_h,basis,eigen=binary_arrays(matrix);assert nq==q and matrix_h==h and list(eigen)==sorted(eigen)
        assert list(map(F,eigen))==list(map(F,spectrum['eigenvalues']))
        assert all(abs(v)<=2.**200 for group in (matrix_h,basis,eigen) for v in group)
        fields=spectrum['rounding_output'];delta=q*F.from_float(float.fromhex(fields['max_gram_hex']));rho=q*F.from_float(float.fromhex(fields['max_residual_hex']));assert delta<1
        inv=1/(1-integer.I(delta)).sqrt();bound=integer.I(rho)*inv+2*integer.I(max(map(abs,map(F,eigen))))*(1+integer.I(delta)).sqrt()*(inv-1)
        assert bound.record()==spectrum['eigenvalue_error'] and F(spectrum['epsilon_upper'])==F(bound.hi,integer.SCALE)
        errors=read(directory/'ERRORS.json');assert F(errors['total'])==F(errors['construction'])+F(errors['spectral'])
        assert F(errors['construction'])==F(record['operator_error']) and F(errors['spectral'])==F(spectrum['epsilon_upper'])
        text=subprocess.check_output([str(args.input.resolve()/'rounding_spectral_residual'),str(matrix.resolve())],text=True)
        got=dict(line.split(maxsplit=1) for line in text.splitlines());assert all(got[k]==v for k,v in fields.items() if k!='wall_seconds')
        (args.out/(case['label']+'_spectrum_repeat.txt')).write_text(text)
    # Reconstruct both tiny references without NumPy or native arithmetic.
    for toy in read(args.input/'TOYS.json'):
        cq,b,h,scalars=construction_arrays(args.input/(toy['case']+'_construction.bin'));assert cq==2
        A=F(toy['scalar_A']);D=F(toy['scalar_D']);error=F(toy['operator_error'])/2
        for i in range(2):
            for j in range(2):
                exact=A*(F(b[i*2+j])+F(b[j*2+i]))/2-D*sum(F(b[k*2+i])*F(b[k*2+j]) for k in range(2))
                assert abs(exact-F(h[i*2+j]))<=error
    result=dict(matrices=2,identical_source_B_bytes=True,construction_scalar_checks=True,spectral_error_checks=True,
        native_kernel_repetitions=4,exact_toy_matrix_cases=2,scope='Bhat D/W matrices; bad-event probability integration still separate')
    with (args.out/'CHECKS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
