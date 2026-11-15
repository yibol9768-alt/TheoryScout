"""Standard-library audit of finite-summary-compatible energy witnesses."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import json

from construct_moment_energy_witnesses import read,sha,save,targets,features,contrast_bounds,point_constraints,cdf_bound
import certify_weighted_chisquare as integer

S=integer.SCALE


def reference_checks():
    support=list(map(F,[-1,0]))+[F(1,2),F(1),F(2),F(3)]
    grid=[F(1,2),F(1),F(2),F(3)];count=0
    for weights in [[1]*6,[0,1,2,0,3,1]]:
        law=[F(x,sum(weights)) for x in weights]
        cdf=lambda x:sum(p for t,p in zip(support,law) if t<=x)
        for error in [F(0),F(1,20)]:
            row=dict(t=list(map(str,grid)),sweeps=[dict(lower_CDF=[str(max(F(0),cdf(t)-error)) for t in grid],upper_CDF=[str(min(F(1),cdf(t)+error)) for t in grid])])
            for d in [F(0),F(1,4),F(1,2),F(1),F(3,2),F(2),F(4),F(6)]:
                for s in [F(1,2),F(1),F(3,2),F(2)]:
                    for r in [F(3,4),F(1),F(5,4)]:
                        low,high=contrast_bounds(row,d,s,r*r)
                        actual=cdf(d/s)-cdf(d/r)
                        assert low<=actual<=high;count+=1
    return dict(squared_CDF_lookup_and_signed_contrast_cases_with_atoms=count)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('certificate',type=Path)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];result=read(args.certificate/'RESULT.json');design=read(args.certificate/'DESIGN.json')
    assert design['source_hashes']==result['source_hashes']
    for name,digest in design['source_hashes'].items():assert sha(root/name)==digest
    target=targets(root)
    for key,value in target.items():assert design[key]==value
    assert result['selected_moments']==target['selected_moments'] and result['mean_sqrt_V']==target['mean_sqrt_V']
    selected={k:F(v) for k,v in target['selected_moments'].items()};meanv=selected['mean_V']
    moment_source=read(root/'revisions/r28_conditional_moments_v2/RESULT.json');source=moment_source['C0_moments']
    for key,value in selected.items():
        assert F(int(source[key]['lower_integer']),S)<=value<=F(int(source[key]['upper_integer']),S)
    # Verify the chosen common correlation also lies in its recorded interval.
    corr=source['correlation_DV'];ratio=selected['covariance_DV']**2/(selected['variance_D']*selected['variance_V'])
    assert F(int(corr['lower_integer']),S)**2<=ratio<=F(int(corr['upper_integer']),S)**2
    cdf_source=read(root/'revisions/r23_cdf_dense8/RESULT.json');cdf=cdf_source['rows']
    sphere=root/'revisions/r21_sphere_cdf/RESULT.json'
    assert sha(sphere)==cdf_source['source_hashes']['revisions/r21_sphere_cdf/RESULT.json']
    assert read(sphere)['m']==moment_source['n']-1==16594
    frozen=read(root/'revisions/r32_frozen_energy/RESULT.json')
    assert meanv>F(frozen['scale_minorant'])**2
    baseline=read(root/'revisions/r37_joint_radial_v2/RESULT.json')
    count_atoms=count_moments=count_constraints=0;audits=[]
    assert [row['L'] for row in result['rows']]==[1,5]
    for row in result['rows']:
        length=row['L'];threshold=next(v for v in cdf if v['L']==length)
        assert [w['direction'] for w in row['witnesses']]==['positive','negative']
        for witness in row['witnesses']:
            atoms=[(F(a['D']),F(a['sqrt_V']),F(a['probability'])) for a in witness['atoms']]
            assert all(d>0 and s>0 and p>0 for d,s,p in atoms)
            assert sum(p for d,s,p in atoms)==1
            for index,value in enumerate(map(F,target['moment_targets'])):
                assert sum(p*features(d,s,target)[index] for d,s,p in atoms)==value;count_moments+=1
            raw=lambda a,b:sum(p*d**a*(s*s)**b for d,s,p in atoms)
            mu=raw(1,0);var=raw(2,0)-mu*mu
            measured=dict(mean_D=mu,variance_D=var,third_central_D=sum(p*(d-mu)**3 for d,s,p in atoms),
                fourth_central_D=sum(p*(d-mu)**4 for d,s,p in atoms),mean_V=raw(0,1),
                variance_V=raw(0,2)-raw(0,1)**2,covariance_DV=raw(1,1)-mu*raw(0,1))
            assert measured==selected and sum(p*s for d,s,p in atoms)==F(target['mean_sqrt_V'])
            assert measured['covariance_DV']>0
            inequalities=point_constraints(root,[(d,s) for d,s,p in atoms]);assert len(inequalities)==54
            for constraint in inequalities:
                value=sum(p*x for (d,s,p),x in zip(atoms,constraint['values']))
                assert value<=constraint['original_rhs'];count_constraints+=1
            intervals=[contrast_bounds(threshold,d,s,meanv) for d,s,p in atoms]
            lo=sum(p*v[0] for (d,s,p),v in zip(atoms,intervals));hi=sum(p*v[1] for (d,s,p),v in zip(atoms,intervals))
            assert witness['signed_energy_contrast_interval']==list(map(str,[lo,hi]))
            assert lo>0 if witness['direction']=='positive' else hi<0
            assert witness['sign_certified']
            diagnostics=[]
            for distribution in cdf:
                L=distribution['L'];previous=next(v for v in baseline['rows'] if v['L']==L)
                qa=[sum(p*cdf_bound(distribution,d,s*s,side) for d,s,p in atoms) for side in ['lower','upper']]
                # Shared certified point envelopes imply the inherited R37
                # complete-recall bounds for these generic laws as well.
                qa=[max(qa[0],F(previous['conditional_recall_lower'])),min(qa[1],F(previous['conditional_recall_upper']))]
                assert qa[0]<=qa[1]
                qr=[sum(p*cdf_bound(distribution,d,meanv,side) for d,s,p in atoms) for side in ['lower','upper']]
                old=next(v for v in frozen['rows'] if v['L']==L)
                minorant=sum(p*sum(F(layer['jump_weight']) for layer in old['layers'] if F(layer['left'])<=d<=F(layer['right'])) for d,s,p in atoms)
                assert minorant>=F(old['unit_minus_RMS_frozen_success_lower'])
                diagnostics.append(dict(L=L,actual_Q_interval_with_R37=list(map(str,qa)),frozen_RMS_Qr_CDF_interval=list(map(str,qr)),
                    R32_contrast_minorant=str(minorant),R32_positive_frozen_energy_lower_preserved=True))
            count_atoms+=len(atoms)
            audits.append(dict(L=length,direction=witness['direction'],atoms=len(atoms),signed_energy_contrast_interval=list(map(str,[lo,hi])),
                diagnostics=diagnostics,all_54_saved_scalar_point_bounds_satisfied=True))
    save(args.out,dict(reference=reference_checks(),source_hashes=len(design['source_hashes']),exact_positive_atoms=count_atoms,
        exact_moment_equalities=count_moments,exact_scalar_point_inequalities=count_constraints,
        chosen_correlation_in_R28_interval=True,mean_root_in_R31_interval=True,audits=audits,
        solver_required=False,spherical_order_threshold_CDF_bounds_inherited=True,
        full_C0_quadratic_geometry_and_D_marginal_not_matched=True,
        witnesses_do_not_determine_actual_C0_energy_sign=True,independent_proof_review=False))
    print('PASS four rational laws; 36 moments,216 point bounds; all four signs certified')


if __name__=='__main__':main()
