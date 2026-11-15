"""Audit nonzero joint thresholds, including exact ambient null padding."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import argparse
import json

from certify_joint_pencil_thresholds import read,sha,save
from check_recall_upper import audit_integral
import certify_weighted_chisquare as integer


def references():
    matrices=[[[F(1,4),F(0)],[F(0),F(1,2)]],[[F(1),F(1,2)],[-F(1,4),F(3,4)]],[[F(0),F(1)],[F(-1),F(0)]]]
    checks=wrong_padding=0
    for b,n,front,tail,c,y in product(matrices,[4,6],[(F(1),F(0)),(F(0),F(1)),(F(1),F(-2))],[F(1),F(3)],[F(1,4),F(3,10)],[F(-1,10),F(1,10)]):
        g=list(front)+[tail]*(n-2);norm=sum(x*x for x in g)
        bu=[sum(x*z for x,z in zip(row,front)) for row in b]
        a=sum(x*z for x,z in zip(front,bu));nn=sum(x*x for x in bu)
        D=n*a/norm;W=F(n*n,n-1)*nn/norm;V=W-D*D/(n-1)
        Y=c*D-(D*D+(n-1)*V)/n
        block=c*a-nn-y/n*sum(x*x for x in front);full=block-y/n*sum(x*x for x in g[2:])
        assert full==(Y-y)*norm/n;checks+=1
        wrong_padding+=int((full>=0)!=(block>=0))
    assert wrong_padding>0
    return dict(exact_nonzero_joint_event_and_padding_checks=checks,omitted_null_padding_changes_event_examples=wrong_padding)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('certificate',type=Path);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--reintegrate',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];result=read(args.certificate/'RESULT.json');design=read(args.certificate/'DESIGN.json')
    assert result['source_hashes']==design['source_hashes']
    hist=read(root/'revisions/R41_HISTORY.json');manifest=read(root/hist['path']/'manifest.json')
    for name,digest in result['source_hashes'].items():assert sha(root/name)==digest==manifest['files'][name]
    hist=read(root/'revisions/R27_HISTORY.json');manifest=read(root/hist['path']/'manifest.json');assert sha(root/'tools/check_recall_upper.py')==manifest['files']['tools/check_recall_upper.py']
    parent=root/'revisions/r41_joint_pencils';old=read(parent/'RESULT.json');n=old['n'];q=old['q'];assert (n,q)==(result['n'],result['q'])==(design['n'],design['q'])
    assert [(r['c'],r['y']) for r in result['rows']]==[('1/4','-1/10'),('1/4','1/10'),('3/10','-1/10'),('3/10','1/10')]
    widths=[]
    for row in result['rows']:
        y=F(row['y']);p=next(v for v in old['rows'] if v['c']==row['c']);assert row['parent_directory']==p['directory']
        eigen=list(map(F,read(parent/p['directory']/'actual_spectrum.json')['eigenvalues']));epsilon=F(read(parent/p['directory']/'ERRORS.json')['total_C0_eigenvalue_error'])
        assert epsilon==F(p['C0_eigenvalue_error']);interval=[]
        for side,sign in [('lower',-1),('upper',1)]:
            coefficients=[v-y/n+sign*epsilon for v in eigen]+[-y/n];filename=row['stem']+'_'+side+'.json';saved=read(args.certificate/filename)
            upper,width=audit_integral(args.certificate,filename,coefficients,[1]*q+[n-q]);widths.append(width)
            if args.reintegrate:
                repeated,_=integer.certify(coefficients,[1]*q+[n-q],integer.pi_interval(),integer.gauss8(),target=F(saved['target_width']));assert repeated==saved
            interval.append(max(F(0),F(int(saved['probability']['lower_integer']),integer.SCALE)) if side=='lower' else upper)
        assert row['C0_survival_interval']==list(map(str,interval))
    for p in old['rows']:
        left=next(r for r in result['rows'] if r['c']==p['c'] and F(r['y'])<0)
        right=next(r for r in result['rows'] if r['c']==p['c'] and F(r['y'])>0)
        assert F(right['C0_survival_interval'][1])<=F(p['probability_C0_positive'][0])<=F(p['probability_C0_positive'][1])<=F(left['C0_survival_interval'][0])
    save(args.out,dict(reference=references(),source_hashes=len(result['source_hashes']),coefficient_and_error_audits=8,full_ambient_dimension=n,null_multiplicity=n-q,
        fixed_c_survival_monotonicity_checks=2,max_comparison_integral_width_approximate=float(max(widths)),integer_integrals_recomputed=8 if args.reintegrate else 0,
        prior_spectral_and_C0_transfer_bounds_inherited=True,independent_proof_review=False))
    print('PASS eight integrals and full ambient padding')


if __name__=='__main__':main()
