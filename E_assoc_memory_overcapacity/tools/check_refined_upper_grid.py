"""Audit new C0 upper nodes, retained bounds and removed staircase slack."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import argparse
import json

from refine_recall_upper_grid import read,sha,save,closure,removed_staircase_slack
from certify_recall_upper import upper_expectation
from check_recall_upper import audit_integral
import certify_weighted_chisquare as integer


def references():
    grid=list(map(F,[1,2,3,4]));support=[F(-1),F(0),F(1),F(3,2),F(2),F(3),F(4),F(5)];cases=0
    for weights in [[1]*8,[1,0,2,0,1,0,3,1],[0,0,0,1,2,3,0,2]]:
        law=[F(w,sum(weights)) for w in weights]
        cdf=lambda t:sum(p for p,x in zip(law,support) if x<=t)
        for excess in [F(0),F(1,10),F(1,3)]:
            row=dict(t=list(map(str,grid)),sweeps=[dict(lower_CDF=[str(max(F(0),cdf(t)-excess)) for t in grid],upper_CDF=[str(min(F(1),cdf(t)+excess)) for t in grid])])
            bounds=closure(row)
            for a,b,c,d in product([F(1,4),F(1,2),F(3,4)],repeat=4):
                old={F(1):a,F(3):b};new={**old,F(2):c,F(4):d}
                def step(points,t):
                    value=F(1)
                    for x,upper in sorted(points.items()):
                        if t>x:value=min(value,upper)
                    return value
                before=sum(p*step(old,t) for p,t in zip(law,support));after=sum(p*step(new,t) for p,t in zip(law,support))
                decrease=before-after;assert decrease>=0
                assert upper_expectation(new,bounds['upper'])<=upper_expectation(old,bounds['upper'])
                assert upper_expectation(new,bounds['upper'])>=after
                slack=removed_staircase_slack(old,new,bounds);assert F(slack['lower'])<=decrease<=F(slack['upper'])
                if excess==0:assert upper_expectation(old,bounds['upper'])-upper_expectation(new,bounds['upper'])==decrease
                cases+=1
    return dict(upper_refinement_and_slack_cases_with_atoms=cases,negative_threshold_and_both_tails_retained=True,independent_proof_review=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('certificate',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];result=read(args.certificate/'RESULT.json');design=read(args.certificate/'DESIGN.json')
    assert result['source_hashes']==design['source_hashes']
    for name,digest in result['source_hashes'].items():assert sha(root/name)==digest
    receipt=read(root/'revisions/R27_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
    assert sha(root/'tools/check_recall_upper.py')==manifest['files']['tools/check_recall_upper.py']
    scope=read(root/'revisions/r11_anchor_certificate_v2/SCOPE.json');n=scope['n'];m=n-1;q=scope['block_dimension'];a=F(scope['center_anchor'])
    bad=read(root/'revisions/r27_recall_upper/BAD_EVENTS.json');l0=F(bad['l0']);u0=F(bad['u0']);R=F(bad['radius']);delta=F(bad['union_upper'])
    eta=F(read(root/'revisions/r13_scatter_checks/CHECKS.json')['eta_upper']);families={}
    for label,directory in [('first',root/'revisions/r11_anchor_certificate_v2'),('second',root/'revisions/r25_second_anchor')]:
        families[label]=(F(read(directory/'SCOPE.json')['alpha']),read(directory/'actual_spectrum.json')['eigenvalues'],F(read(directory/'ERRORS.json')['total']))
    old=read(root/'revisions/r27_recall_upper/RESULT.json');oldpoints={F(v['t']):F(v['target_C0_probability_upper']) for v in old['point_uppers']};merged=oldpoints.copy();widths=[];secants=0
    for node,choice in zip(result['new_point_uppers'],design['selection']):
        assert (node['t'],node['family'],node['l'],node['u'])==(choice['t'],choice['family'],choice['l'],choice['u'])
        beta,eigen,error=families[node['family']];t=F(node['t']);l=F(node['l']);u=F(node['u'])
        assert beta==F(node['beta']) and 0<l<=l0<u0<=u and l+u==t/beta
        gap=integer.I(n)*integer.I(eta)*(1+integer.I(t)/integer.I(m).sqrt());gu=F(gap.hi,integer.SCALE);gamma=4*beta*m
        c=4*beta*beta*(R*R-a*a)-4*beta*beta*m*l*u+gamma*gu
        assert c==F(node['shifted_constant']) and gu==F(node['score_gap_upper'])
        upper,width=audit_integral(args.certificate,node['integral_file'],[F(x)+c/n+error for x in eigen]+[c/n],[1]*q+[n-q])
        assert upper==F(node['quadratic_event_upper']) and min(F(1),upper+delta)==F(node['target_C0_probability_upper'])
        assert t not in oldpoints;merged[t]=F(node['target_C0_probability_upper']);widths.append(width)
        for d,root_energy in product([a-R,a,a+R],[l0,(l0+u0)/2,u0]):
            v=root_energy**2;w=v+d*d/m
            U=(1+2*beta*a/m)*d-beta*w+beta*(R*R-a*a)/m-beta*l*u
            assert U>=d-t*root_energy
            scaled=(4*beta*m+8*beta*beta*a)*d-4*beta*beta*m*w+c
            assert scaled==gamma*(U+gu)
            secants+=1
    assert result['all_point_uppers']=={str(t):str(v) for t,v in sorted(merged.items())}
    assert all(merged[t]==v for t,v in oldpoints.items()) and len(merged)==len(oldpoints)+len(widths)
    cdf={row['L']:closure(row) for row in read(root/'revisions/r23_cdf_dense8/RESULT.json')['rows']}
    checked=[]
    for row in result['rows']:
        L=row['L'];previous=next(v for v in old['rows'] if v['L']==L);upper=upper_expectation(merged,cdf[L]['upper']);lower=F(previous['conditional_recall_lower'])
        assert upper==F(row['conditional_recall_upper']) and lower==F(row['conditional_recall_lower']) and lower<=upper<=F(previous['conditional_recall_upper'])
        assert F(row['certified_upper_endpoint_decrease'])==F(previous['conditional_recall_upper'])-upper
        assert removed_staircase_slack(oldpoints,merged,cdf[L])==row['removed_upper_staircase_slack']
        checked.append(dict(L=L,upper=str(upper),upper_endpoint_decrease=row['certified_upper_endpoint_decrease']))
    output=dict(reference=references(),new_point_coefficient_and_error_audits=len(widths),selected_secant_algebra_cases=secants,
        old_points_retained=len(oldpoints),total_upper_points=len(merged),global_bad_budget_reused=True,
        actual_target_upper_and_slack_sums_reconstructed=checked,original_actual_lower_preserved=True,
        max_comparison_integral_width_approximate=float(max(widths)),heuristic_frequencies_not_used_as_certificates=True,
        independent_proof_review=False)
    save(args.out,output);print(json.dumps(output,indent=2))


if __name__=='__main__':main()
