"""Add the two new C0 joint-event intervals to the frozen R39 scalar ledger."""
from pathlib import Path
from fractions import Fraction as F
import argparse

import construct_moment_energy_witnesses as legacy

ORIGINAL=legacy.point_constraints


def augmented_constraints(root,candidates):
    rows=ORIGINAL(root,candidates);source=legacy.read(root/'revisions/r41_joint_pencils/RESULT.json')
    n=source['n'];inset=F(1,10**8)
    for node in source['rows']:
        c=F(node['c']);lower,upper=map(F,node['probability_C0_positive'])
        indicator=[F(n*c*d-d*d-(n-1)*s*s>0) for d,s in candidates]
        for side,sign,value in [('lower',-1,lower),('upper',1,upper)]:
            rows.append(dict(kind='joint_pencil',t=str(c),side=side,values=[sign*x for x in indicator],original_rhs=sign*value,rhs=sign*value-inset))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    receipt=legacy.read(root/'revisions/R39_HISTORY.json');manifest=legacy.read(root/receipt['path']/'manifest.json')
    assert legacy.sha(root/'tools/construct_moment_energy_witnesses.py')==manifest['files']['tools/construct_moment_energy_witnesses.py']
    legacy.point_constraints=augmented_constraints
    legacy.main()
    legacy.save(args.out/'JOINT_EXTENSION.json',dict(joint_probability_source='revisions/r41_joint_pencils/RESULT.json',
        joint_probability_source_sha256=legacy.sha(root/'revisions/r41_joint_pencils/RESULT.json'),
        original_generator_sha256=legacy.sha(root/'tools/construct_moment_energy_witnesses.py'),
        saved_result_sha256=legacy.sha(args.out/'RESULT.json'),original_scalar_constraints=54,new_joint_event_constraints=4,
        geometry_of_full_C0_operator_not_realized=True,actual_C0_energy_sign_not_decided=True))


if __name__=='__main__':main()
