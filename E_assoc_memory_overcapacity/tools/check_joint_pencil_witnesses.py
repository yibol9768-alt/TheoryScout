"""Audit the R39 scalar ledger plus four new joint-event inequalities."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import json
import subprocess
import sys

from construct_moment_energy_witnesses import read,sha,save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('certificate',type=Path)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];certificate=args.certificate.resolve();result=read(certificate/'RESULT.json');extension=read(certificate/'JOINT_EXTENSION.json')
    assert sha(certificate/'RESULT.json')==extension['saved_result_sha256']
    assert sha(root/extension['joint_probability_source'])==extension['joint_probability_source_sha256']
    assert sha(root/'tools/construct_moment_energy_witnesses.py')==extension['original_generator_sha256']
    receipt=read(root/'revisions/R39_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
    for name in ['tools/construct_moment_energy_witnesses.py','tools/check_moment_energy_witnesses.py']:
        assert sha(root/name)==manifest['files'][name]
    with (args.out/'base_audit.log').open('x') as log:
        subprocess.run([sys.executable,'-B',str(root/'tools/check_moment_energy_witnesses.py'),str(certificate),'--out',str(args.out.resolve()/'base_checks.json')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    base=read(args.out/'base_checks.json');pencils=read(root/extension['joint_probability_source']);n=pencils['n'];checks=[]
    for row in result['rows']:
        for witness in row['witnesses']:
            for pencil in pencils['rows']:
                c=F(pencil['c']);lo,hi=map(F,pencil['probability_C0_positive'])
                probability=sum(F(a['probability']) for a in witness['atoms'] if n*c*F(a['D'])-F(a['D'])**2-(n-1)*F(a['sqrt_V'])**2>0)
                assert lo<=probability<=hi
                checks.append(dict(L=row['L'],direction=witness['direction'],c=str(c),exact_joint_probability=str(probability)))
    output=dict(base_scalar_ledger_audit=base,extra_exact_joint_inequalities=2*len(checks),
        total_exact_inequalities=base['exact_scalar_point_inequalities']+2*len(checks),joint_probabilities=checks,
        actual_C0_joint_probability_source_sha256=extension['joint_probability_source_sha256'],
        old_four_laws_excluded_but_new_opposite_sign_laws_exist=True,
        new_laws_do_not_realize_full_C0_geometry_or_D_marginal=True,
        actual_C0_energy_sign_unresolved=True,independent_proof_review=False)
    save(args.out/'CHECKS.json',output)
    print('PASS base moment/scalar audit and16 additional exact joint inequalities; actual C0 sign remains unresolved')


if __name__=='__main__':main()
