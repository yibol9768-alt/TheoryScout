"""Nonzero ambient joint-event thresholds from the two frozen R41 spectra."""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime,timezone
import argparse
import hashlib
import json
import time

import certify_weighted_chisquare as integer


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    source=root/'revisions/r41_joint_pencils';baseline=read(source/'RESULT.json');n=baseline['n'];q=baseline['q']
    receipt=read(root/'revisions/R41_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
    names=['revisions/r41_joint_pencils/RESULT.json','tools/certify_weighted_chisquare.py']
    for row in baseline['rows']:
        names.extend('revisions/r41_joint_pencils/'+row['directory']+'/'+name for name in ['actual_spectrum.json','ERRORS.json'])
    for name in names:assert sha(root/name)==manifest['files'][name]
    hashes={name:sha(root/name) for name in names}
    save(args.out/'DESIGN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),n=n,q=q,
        thresholds=['-1/10','1/10'],source_hashes=hashes,full_ambient_null_multiplicity=n-q,
        event='Y_c=cD-(D^2+(n-1)V)/n >= y; block lambda-y/n+-eps, null -y/n'))
    pi=integer.pi_interval();gauss=integer.gauss8();rows=[]
    for parent in baseline['rows']:
        c=F(parent['c']);directory=source/parent['directory'];eigen=list(map(F,read(directory/'actual_spectrum.json')['eigenvalues']))
        epsilon=F(read(directory/'ERRORS.json')['total_C0_eigenvalue_error']);assert epsilon==F(parent['C0_eigenvalue_error'])
        for y in [F(-1,10),F(1,10)]:
            stem=f'node{len(rows)+1:02d}';interval=[]
            for side,sign in [('lower',-1),('upper',1)]:
                coefficients=[v-y/n+sign*epsilon for v in eigen]+[-y/n]
                record,prob=integer.certify(coefficients,[1]*q+[n-q],pi,gauss,target=F(1,10**6))
                save(args.out/(stem+'_'+side+'.json'),record)
                interval.append(max(F(0),F(prob.lo,integer.SCALE)) if side=='lower' else min(F(1),F(prob.hi,integer.SCALE)))
                print(c,y,side,'complete',flush=True)
            assert 0<=interval[0]<=interval[1]<=1
            row=dict(c=str(c),y=str(y),parent_directory=parent['directory'],stem=stem,
                C0_survival_interval=list(map(str,interval)),approximate=list(map(float,interval)))
            rows.append(row);save(args.out/(stem+'.json'),row)
            print('C0 joint survival',c,y,row['approximate'],flush=True)
    save(args.out/'RESULT.json',dict(n=n,q=q,rows=rows,source_hashes=hashes,new_integer_probability_integrals=8,
        new_matrices=0,new_samples=0,full_null_padding_retained=True,independent_proof_review=False,
        not_recall_or_energy_flip_probability=True,wall_seconds=time.monotonic()-start))


if __name__=='__main__':main()
