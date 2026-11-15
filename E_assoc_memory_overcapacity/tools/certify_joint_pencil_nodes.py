"""C0 probability certificates for two joint K/N pencil signs.

Reuse the saved B and native residual kernels, with a separate exact-C0
operator transfer. No training replay or threshold samples.
"""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime,timezone
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    source=root/'revisions/r11_anchor_certificate_v2';names=['revisions/r11_anchor_certificate_v2/actual_construction.bin','revisions/r11_anchor_certificate_v2/SCOPE.json']
    for stem in ['rounding_anchor_construction','rounding_spectral_residual']:
        names.extend(['revisions/r11_anchor_certificate_v2/'+stem,'revisions/r11_anchor_certificate_v2/'+stem+'_BUILD.json','tools/'+stem+'.cpp'])
    names.extend('tools/'+name for name in ['certify_anchor_node.py','certify_spectral_residual.py','check_anchor_certificate.py','check_spectral_certificate.py','certify_weighted_chisquare.py'])
    history=read(root/'revisions/R13_HISTORY.json');manifest=read(root/history['path']/'manifest.json')
    for name in names:assert spectral.sha(root/name)==manifest['files'][name]
    extra=['revisions/r28_conditional_moments_v2/INPUTS.json']
    history=read(root/'revisions/R28_HISTORY.json');manifest=read(root/history['path']/'manifest.json')
    for name in extra:assert spectral.sha(root/name)==manifest['files'][name]
    names+=extra;hashes={name:spectral.sha(root/name) for name in names}
    inputs=read(root/extra[0]);scope=read(source/'SCOPE.json');n=scope['n'];q=scope['block_dimension']
    assert (n,q)==(inputs['n'],inputs['q']) and inputs['source_hashes']['revisions/r11_anchor_certificate_v2/actual_construction.bin']==hashes['revisions/r11_anchor_certificate_v2/actual_construction.bin']
    eta=F(inputs['eta']);nu=F(inputs['N_operator_perturbation']);kernels={}
    for stem in ['rounding_anchor_construction','rounding_spectral_residual']:
        build=read(source/(stem+'_BUILD.json'));assert spectral.sha(source/stem)==build['executable_sha256']
        shutil.copy2(source/stem,args.out/stem);shutil.copy2(source/(stem+'_BUILD.json'),args.out/(stem+'_BUILD.json'));kernels[stem]=args.out/stem
    coefficients=[F(1,4),F(3,10)]
    spectral.save(args.out/'DESIGN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),n=n,q=q,
        coefficients=list(map(str,coefficients)),eta_K=str(eta),nu_N=str(nu),source_hashes=hashes,
        event='P_C0(u^T(c Kbar-Nbar)u>0)=P_C0(n c D-D^2-(n-1)V>0)',
        no_new_data_or_threshold_bank=True))
    qb,raw,_,_=construction_arrays(source/'actual_construction.bin');assert qb==q
    b=np.array(raw,dtype=np.float64).reshape(q,q);anchor.sha=spectral.sha;rows=[]
    pi=integer.pi_interval();gauss=integer.gauss8()
    for index,c in enumerate(coefficients):
        directory=args.out/f'node{index+1:02d}';directory.mkdir()
        h=float(c)*(b+b.T)/2-b.T@b
        h,construction,record=anchor.construct(b,h,c,F(1),kernels['rounding_anchor_construction'],directory,'actual')
        spectral.save(directory/'CONSTRUCTION.json',record)
        eigen,residual,spectrum=spectral.matrix_certificate(h,kernels['rounding_spectral_residual'],directory,'actual',integer)
        coupling=c*eta+nu;error=construction+residual+coupling
        assert F(float(eigen[0]))+error<0<F(float(eigen[-1]))-error
        spectral.save(directory/'ERRORS.json',dict(construction=str(construction),spectral=str(residual),C0_operator_transfer=str(coupling),total_C0_eigenvalue_error=str(error)))
        bounds=[]
        for side,sign in [('lower',-1),('upper',1)]:
            values=[F(float(v))+sign*error for v in eigen]
            integral,prob=integer.certify(values,[1]*q,pi,gauss,target=F(1,10**6))
            spectral.save(directory/('probability_'+side+'.json'),integral);bounds.append(prob)
            print('c',str(c),side,'comparison integral complete',flush=True)
        lower=max(F(0),F(bounds[0].lo,integer.SCALE));upper=min(F(1),F(bounds[1].hi,integer.SCALE));assert lower<=upper
        rows.append(dict(c=str(c),directory=directory.name,probability_C0_positive=[str(lower),str(upper)],
            approximate=[float(lower),float(upper)],C0_eigenvalue_error=str(error),both_actual_eigenvalue_signs_certified=True))
        print('JOINT C0',str(c),float(lower),float(upper),flush=True)
    assert F(rows[0]['probability_C0_positive'][1])<=F(rows[1]['probability_C0_positive'][0])
    spectral.save(args.out/'RESULT.json',dict(n=n,q=q,rows=rows,source_hashes=hashes,
        new_pencil_matrices=2,new_probability_integrals=4,new_samples=0,new_threshold_integrals=0,
        actual_C0_operator_transfer_included=True,probability_is_not_full_energy_flip=True,
        actual_C0_recall_or_energy_remainder_not_updated=True,independent_proof_review=False,
        wall_seconds=time.monotonic()-start))


if __name__=='__main__':main()
