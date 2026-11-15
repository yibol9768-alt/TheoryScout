"""Certify D and total-energy W matrices from the frozen represented B."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import json
import shutil
import time

import numpy as np
import certify_anchor_node as anchor
import certify_spectral_residual as spectral
import certify_weighted_chisquare as integer
from check_anchor_certificate import construction_arrays


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    old=root/'revisions/r11_anchor_certificate_v2';receipt=read(root/'revisions/R13_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
    protected=['actual_construction.bin','SCOPE.json']
    for stem in ['rounding_anchor_construction','rounding_spectral_residual']:protected+=[stem,stem+'_BUILD.json']
    for name in protected:assert spectral.sha(old/name)==manifest['files']['revisions/r11_anchor_certificate_v2/'+name]
    for stem in ['rounding_anchor_construction','rounding_spectral_residual']:
        build=read(old/(stem+'_BUILD.json'));assert spectral.sha(root/'tools'/(stem+'.cpp'))==build['source_sha256']
        shutil.copy2(old/stem,args.out/stem);shutil.copy2(old/(stem+'_BUILD.json'),args.out/(stem+'_BUILD.json'))
    q,raw,_,_=construction_arrays(old/'actual_construction.bin');b=np.array(raw,dtype=np.float64).reshape(q,q)
    scope=read(old/'SCOPE.json');n=scope['n'];m=n-1;anchor.sha=spectral.sha;start=time.monotonic();cases=[]
    for label,A,D in [('D',F(1),F(0)),('W',F(0),-F(n,m))]:
        directory=args.out/label;directory.mkdir();h=float(A)*(b+b.T)/2-float(D)*(b.T@b)
        h,construction,record=anchor.construct(b,h,A,D,args.out/'rounding_anchor_construction',directory,'actual')
        spectral.save(directory/'CONSTRUCTION.json',record)
        _,error,spectrum=spectral.matrix_certificate(h,args.out/'rounding_spectral_residual',directory,'actual',integer)
        spectral.save(directory/'ERRORS.json',dict(construction=str(construction),spectral=str(error),total=str(construction+error)))
        cases.append(dict(label=label,A=str(A),Dcoef=str(D),n=n,q=q))
        print(label,'matrix certified, total error',float(construction+error),flush=True)
    tests=[]
    for index,(A,D) in enumerate([(F(1),F(0)),(F(0),-F(n,m))]):
        toy=np.array([[.5,-.25],[.75,.125]]);h=float(A)*(toy+toy.T)/2-float(D)*(toy.T@toy);label='toy'+str(index)
        h,error,record=anchor.construct(toy,h,A,D,args.out/'rounding_anchor_construction',args.out,label)
        for i in range(2):
            for j in range(2):
                exact=A*(F(float(toy[i,j]))+F(float(toy[j,i])))/2-D*sum(F(float(toy[k,i]))*F(float(toy[k,j])) for k in range(2))
                assert abs(exact-F(float(h[i,j])))<=error/2
        tests.append(dict(case=label,exact_entry_bounds=True,**record))
    spectral.save(args.out/'TOYS.json',tests)
    spectral.save(args.out/'SCOPE.json',dict(cases=cases,n=n,q=q,center_anchor=scope['center_anchor'],
        source_B_sha256=spectral.sha(old/'actual_construction.bin'),wall_seconds=time.monotonic()-start,
        target='Exact represented Bhat D=nwKw and W=nw(nN/m)w; bad-event budget for C0 upper transfer',
        caveat='Inherited IEEE/kernel assumptions, not independent proof review'))


if __name__=='__main__':main()
