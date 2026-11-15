"""Construct/spectrally certify a second fixed-alpha family from saved B.

No Gaussian replay or fresh training. The saved-B anchor event and the
subsequent represented-C0 transfer are separate certificate layers.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import platform
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
    assert platform.machine()=='x86_64';args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];source=root/'revisions/r11_anchor_certificate_v2'
    receipt=read(root/'revisions/R13_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
    names=['revisions/r11_anchor_certificate_v2/actual_construction.bin','revisions/r11_anchor_certificate_v2/SCOPE.json']
    for stem in ['rounding_anchor_construction','rounding_spectral_residual']:
        names.extend(['revisions/r11_anchor_certificate_v2/'+stem,'revisions/r11_anchor_certificate_v2/'+stem+'_BUILD.json'])
    for name in names:assert spectral.sha(root/name)==manifest['files'][name]
    frozen=read(root/'revisions/r8_random_thresholds/FINISHED.json')
    assert spectral.sha(root/'revisions/r8_random_thresholds/nodes.jsonl')==frozen['files']['nodes.jsonl']
    kernels={}
    for stem in ['rounding_anchor_construction','rounding_spectral_residual']:
        build=read(source/(stem+'_BUILD.json'))
        assert spectral.sha(source/stem)==build['executable_sha256']
        assert spectral.sha(root/'tools'/(stem+'.cpp'))==build['source_sha256']
        shutil.copy2(source/stem,args.out/stem);shutil.copy2(source/(stem+'_BUILD.json'),args.out/(stem+'_BUILD.json'))
        kernels[stem]=args.out/stem
    scope=read(source/'SCOPE.json');n=scope['n'];m=n-1;a=F(scope['center_anchor']);old_alpha=F(scope['alpha'])
    node=next(json.loads(s) for s in (root/'revisions/r8_random_thresholds/nodes.jsonl').read_text().splitlines()
              if json.loads(s)['L']==5 and json.loads(s)['index']==64)
    t=F.from_float(node['t']);old_b=F(scope['threshold'])/(2*old_alpha)
    # Rounding selects a legal dyadic parameter; it need not enclose a moment.
    alpha=F.from_float(float(t/(2*old_b)));assert alpha>0
    A=4*alpha*m+8*alpha*alpha*a;D=4*alpha*alpha*n;constant=-4*alpha*alpha*a*a-m*t*t
    q,raw_b,_,_=construction_arrays(source/'actual_construction.bin');b=np.array(raw_b,dtype=np.float64).reshape(q,q)
    anchor.sha=spectral.sha;tests=[]
    for label,aa,dd,toy in [('toy_positive',F(7,3),F(5,7),np.array([[.5,-.25],[.75,.125]])),
                          ('toy_signed',F(-5,3),F(-9,7),np.array([[-.125,.5],[.25,-.75]]))]:
        h=float(aa)*(toy+toy.T)/2-float(dd)*(toy.T@toy)
        h,error,record=anchor.construct(toy,h,aa,dd,kernels['rounding_anchor_construction'],args.out,label)
        for i in range(2):
            for j in range(2):
                exact=aa*(F(float(toy[i,j]))+F(float(toy[j,i])))/2-dd*sum(F(float(toy[k,i]))*F(float(toy[k,j])) for k in range(2))
                assert abs(exact-F(float(h[i,j])))<=error/2
        tests.append(dict(case=label,exact_entry_bounds=True,**record))
    spectral.save(args.out/'toy_checks.json',tests)
    spectral.save(args.out/'SCOPE.json',dict(n=n,block_dimension=q,threshold=str(t),center_anchor=str(a),alpha=str(alpha),
        sqrt_energy_anchor=str(t/(2*alpha)),positive_scaling=str(4*alpha*m),exact_constant=str(constant),
        source='Same exact saved R11 B, second fixed alpha aligned at original top5 median',
        caveat='This directory certifies represented-B event; C0 transfer is separate'))
    spectral.save(args.out/'PROVENANCE.json',dict(source_hashes={name:spectral.sha(root/name) for name in names},
        old_alpha=str(old_alpha),old_b=str(old_b),new_alpha=str(alpha),new_samples=0,
        native_kernels_reused_from='R11; source and executable hashes verified; not recompiled'))
    start=time.monotonic();h=float(A)*(b+b.T)/2-float(D)*(b.T@b)
    h,construction,record=anchor.construct(b,h,A,D,kernels['rounding_anchor_construction'],args.out,'actual')
    spectral.save(args.out/'CONSTRUCTION.json',record);print('New construction error',float(construction),flush=True)
    eigen,error,spectrum=spectral.matrix_certificate(h,kernels['rounding_spectral_residual'],args.out,'actual',integer)
    epsilon=construction+error;spectral.save(args.out/'ERRORS.json',dict(construction=str(construction),spectral=str(error),total=str(epsilon)))
    print('New spectrum certified, epsilon',float(epsilon),flush=True)
    pi=integer.pi_interval();nodes=integer.gauss8();bounds=[]
    for label,sign in [('lower',-1),('upper',1)]:
        coefficients=[F.from_float(float(v))+constant/n+sign*epsilon for v in eigen]+[constant/n]
        result,enclosure=integer.certify(coefficients,[1]*q+[n-q],pi,nodes)
        spectral.save(args.out/('probability_'+label+'.json'),result);bounds.append(enclosure)
        print('Saved-B median',label,'integral complete',flush=True)
    spectral.save(args.out/'RESULT.json',dict(probability=integer.I.raw(bounds[0].lo,bounds[1].hi).record(),
        wall_seconds=time.monotonic()-start,certified_from='Exact same saved R11 B; new matrix construction and spectrum',
        caveat='Not itself the C0 model result; see separately shifted node certificates'))


if __name__=='__main__':main()
