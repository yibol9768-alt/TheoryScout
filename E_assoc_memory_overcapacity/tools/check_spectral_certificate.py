"""Check saved spectral inputs with exact rationals; optionally replay integrals.

The checker needs only Python's standard library. --repeat-residual executes the
saved directed-rounding binary; --reintegrate additionally repeats both integer
integrals into a new directory. Neither mode replays the model construction.
"""
from array import array
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import struct
import subprocess
import sys

import certify_weighted_chisquare as integer


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoints(record):
    scale = 1 << record['denominator_power2']
    return F(record['lower_integer']) / scale, F(record['upper_integer']) / scale


def binary_arrays(path):
    raw = path.read_bytes()
    n, = struct.unpack('<Q', raw[:8])
    values = array('d'); values.frombytes(raw[8:])
    if sys.byteorder != 'little':
        values.byteswap()
    assert len(values) == 2*n*n+n
    return n, values[:n*n], values[n*n:2*n*n], values[2*n*n:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeat-residual', action='store_true')
    parser.add_argument('--reintegrate', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    directory = args.input.resolve()
    build = read(directory/'BUILD.json')
    assert sha(directory/'rounding_spectral_residual') == build['executable_sha256']
    assert sha(Path(__file__).with_name('rounding_spectral_residual.cpp')) == build['source_sha256']
    tests = []
    had = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
    for prefix in ['toy_0', 'toy_1', 'actual']:
        saved = read(directory/f'{prefix}_spectrum.json')
        binary = directory/f'{prefix}.bin'
        assert sha(binary) == saved['input_sha256']
        n,h,q,eigen = binary_arrays(binary)
        assert all(h[i*n+j] == h[j*n+i] for i in range(n) for j in range(n))
        assert [F.from_float(x) for x in eigen] == list(map(F,saved['eigenvalues']))
        assert list(eigen) == sorted(eigen)
        delta, rho = F(saved['gram_operator_bound']), F(saved['residual_operator_bound'])
        fields = saved['rounding_output']
        assert delta == n*F.from_float(float.fromhex(fields['max_gram_hex']))
        assert rho == n*F.from_float(float.fromhex(fields['max_residual_hex']))
        assert 0 <= delta < 1 and rho >= 0
        inverse = 1/(1-integer.I(delta)).sqrt()
        error = integer.I(rho)*inverse + 2*integer.I(max(map(abs,map(F,eigen))))*(1+integer.I(delta)).sqrt()*(inverse-1)
        assert saved['eigenvalue_error'] == error.record()
        epsilon = F(saved['epsilon_upper'])
        assert epsilon == F(error.hi, integer.SCALE)
        if prefix.startswith('toy'):
            # Independent exact checks of H, Q^T Q-I and HQ-Q Lambda.
            exact = [F(-1,2),F(1,4),F(3,4),F(3,2)]
            if prefix == 'toy_1': exact[2] = F(1,4)+F(1,2**30)
            hf, qf, ef = list(map(F,h)), list(map(F,q)), list(map(F,eigen))
            for i in range(n):
                for j in range(n):
                    assert hf[i*n+j] == sum(F(had[i][k]*had[j][k],4)*exact[k] for k in range(n))
                    gram = sum(qf[k*n+i]*qf[k*n+j] for k in range(n)) - int(i==j)
                    residual = sum(hf[i*n+k]*qf[k*n+j] for k in range(n)) - qf[i*n+j]*ef[j]
                    assert abs(gram) <= delta/n and abs(residual) <= rho/n
            assert all(abs(a-b) <= epsilon for a,b in zip(ef,sorted(exact)))
        if args.repeat_residual:
            result = subprocess.check_output([str(directory/'rounding_spectral_residual'),str(binary)],text=True)
            (args.out/f'{prefix}_rounding_repeat.txt').write_text(result)
            fields = dict(line.split(maxsplit=1) for line in result.splitlines())
            assert all(fields[key] == saved['rounding_output'][key] for key in fields if key != 'wall_seconds')
        tests.append(dict(case=prefix,dimension=n,input_hash_checked=True,exact_toy_reference=prefix.startswith('toy')))
    spectrum = read(directory/'actual_spectrum.json')
    scope = read(directory/'ACTUAL_SCOPE.json')
    n = scope['case']['n']; eigen = list(map(F,spectrum['eigenvalues']))
    epsilon = F(spectrum['epsilon_upper']); constant = F(scope['constant'])
    bounds = []
    for label,sign in [('lower',-1),('upper',1)]:
        saved = read(directory/f'probability_{label}.json')
        coefficients = [v+constant/n+sign*epsilon for v in eigen]+[constant/n]
        counts = [1]*len(eigen)+[n-len(eigen)]
        assert coefficients == list(map(F,saved['coefficients'])) and counts == saved['counts']
        if args.reintegrate:
            record,_ = integer.certify(coefficients,counts,integer.pi_interval(),integer.gauss8())
            assert record == saved
            (args.out/f'probability_{label}_repeat.json').write_text(json.dumps(record,indent=2)+'\n')
        bounds.append(endpoints(saved['probability']))
    final = read(directory/'RESULT.json')
    assert endpoints(final['probability']) == (bounds[0][0],bounds[1][1])
    validation = dict(cases=tests,repeat_residual=args.repeat_residual,reintegrated=args.reintegrate,
                      coefficient_shifts_exact=True,probability=final['probability'],
                      scope='Represented matrices only; no model-construction or independent review claim')
    (args.out/'CHECKS.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2))


if __name__ == '__main__': main()
