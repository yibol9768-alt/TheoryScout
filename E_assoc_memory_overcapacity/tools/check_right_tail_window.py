"""Audit four right-tail C0 point certificates and actual window contributions."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import argparse
import json

from certify_right_tail_window import read,sha,save,window_budget
from refine_recall_upper_grid import closure,removed_staircase_slack
from certify_recall_upper import upper_expectation
from check_recall_upper import audit_integral
import certify_weighted_chisquare as integer


def references():
    grid=[F(1,2),F(1),F(2),F(3),F(4)];support=[F(-2),F(0),F(1,2),F(3,2),F(3),F(4),F(5)];cases=0
    pairs=[(F(-1),F(1,2)),(F(1),F(1)),(F(3),F(3,2))]
    for weights in [[1,1,1],[1,2,1],[0,1,3]]:
        p=[F(x,sum(weights)) for x in weights]
        survival=lambda t:sum(w for w,(d,r) in zip(p,pairs) if d>=t*r)
        for tw in [[1]*7,[1,0,2,0,1,3,1],[0,0,1,2,3,1,1]]:
            law=[F(x,sum(tw)) for x in tw];cdf=lambda t:sum(w for w,x in zip(law,support) if x<=t)
            true_left=sum(w*survival(t) for w,t in zip(law,support) if t<=grid[0])
            true_right=sum(w*survival(t) for w,t in zip(law,support) if t>grid[-1])
            true_mid=sum(w*survival(t) for w,t in zip(law,support) if grid[0]<t<=grid[-1])
            total=true_left+true_right+true_mid
            for error,total_error,point_error in product([F(0),F(1,20)],[F(0),F(1,10)],[F(0),F(1,10)]):
                row=dict(t=list(map(str,grid)),sweeps=[dict(lower_CDF=[str(max(F(0),cdf(t)-error)) for t in grid],upper_CDF=[str(min(F(1),cdf(t)+error)) for t in grid])])
                points={t:min(F(1),survival(t)+point_error) for t in grid[1:]};bounds=closure(row)
                budget=window_budget(max(F(0),total-total_error),min(F(1),total+total_error),points,bounds,grid[0],grid[-1])
                assert true_left<=F(budget['left_success_contribution_upper'])
                assert true_right<=F(budget['right_success_contribution_upper'])
                assert true_left+true_right<=F(budget['outside_success_contribution_upper'])
                assert F(budget['window_success_contribution_lower'])<=true_mid<=F(budget['window_success_contribution_upper']);cases+=1
    return dict(actual_window_success_reference_cases_with_atoms=cases,dependent_D_V_laws=True,negative_thresholds_and_both_boundaries_covered=True,independent_proof_review=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('certificate',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];result=read(args.certificate/'RESULT.json');design=read(args.certificate/'DESIGN.json')
    assert result['source_hashes']==design['source_hashes']
    for name,digest in result['source_hashes'].items():assert sha(root/name)==digest
    history=read(root/'revisions/R27_HISTORY.json');manifest=read(root/history['path']/'manifest.json')
    assert sha(root/'tools/check_recall_upper.py')==manifest['files']['tools/check_recall_upper.py']
    scope=read(root/'revisions/r11_anchor_certificate_v2/SCOPE.json');n=scope['n'];m=n-1;q=scope['block_dimension'];a=F(scope['center_anchor']);beta=F(scope['alpha'])
    assert F(design['beta'])==beta
    bad=read(root/'revisions/r27_recall_upper/BAD_EVENTS.json');l0=F(bad['l0']);u0=F(bad['u0']);R=F(bad['radius']);delta=F(bad['union_upper'])
    eta=F(read(root/'revisions/r13_scatter_checks/CHECKS.json')['eta_upper']);error=F(read(root/'revisions/r11_anchor_certificate_v2/ERRORS.json')['total'])
    eigen=read(root/'revisions/r11_anchor_certificate_v2/actual_spectrum.json')['eigenvalues']
    baseline=read(root/'revisions/r35_upper_grid/RESULT.json');oldpoints={F(k):F(v) for k,v in baseline['all_point_uppers'].items()};merged=oldpoints.copy();widths=[];algebra=0
    for node,choice in zip(result['new_point_uppers'],design['selection']):
        for key in ['order','t','l','u']:assert node[key]==choice[key]
        t=F(node['t']);l=F(node['l']);u=F(node['u']);gamma=4*beta*m
        assert t>max(oldpoints) and F(node['beta'])==beta and 0<l<=l0<u0<=u and l+u==t/beta
        gap=integer.I(n)*integer.I(eta)*(1+integer.I(t)/integer.I(m).sqrt());gu=F(gap.hi,integer.SCALE)
        c=4*beta*beta*(R*R-a*a)-4*beta*beta*m*l*u+gamma*gu
        assert c==F(node['shifted_constant']) and gu==F(node['score_gap_upper'])
        event,width=audit_integral(args.certificate,node['integral_file'],[F(x)+c/n+error for x in eigen]+[c/n],[1]*q+[n-q])
        assert event==F(node['quadratic_event_upper']) and min(F(1),event+delta)==F(node['target_C0_probability_upper'])
        merged[t]=F(node['target_C0_probability_upper']);widths.append(width)
        for d,r in product([a-R,a,a+R],[l0,(l0+u0)/2,u0]):
            w=r*r+d*d/m;U=(1+2*beta*a/m)*d-beta*w+beta*(R*R-a*a)/m-beta*l*u
            assert U>=d-t*r
            assert (4*beta*m+8*beta*beta*a)*d-4*beta*beta*m*w+c==gamma*(U+gu);algebra+=1
    assert len(merged)==20 and result['all_point_uppers']=={str(t):str(v) for t,v in sorted(merged.items())}
    assert all(merged[t]==v for t,v in oldpoints.items())
    cdf={row['L']:closure(row) for row in read(root/'revisions/r23_cdf_dense8/RESULT.json')['rows']};left=min(cdf[1]['upper']);right=max(merged)
    rows=[]
    for row in result['rows']:
        L=row['L'];previous=next(v for v in baseline['rows'] if v['L']==L);lower=F(previous['conditional_recall_lower']);upper=upper_expectation(merged,cdf[L]['upper'])
        assert (F(row['conditional_recall_lower']),F(row['conditional_recall_upper']))==(lower,upper)
        assert lower<=upper<=F(previous['conditional_recall_upper']) and F(row['upper_endpoint_drop'])==F(previous['conditional_recall_upper'])-upper
        assert row['window_budget']==window_budget(lower,upper,merged,cdf[L],left,right)
        assert row['old_R35_window_budget']==window_budget(lower,F(previous['conditional_recall_upper']),oldpoints,cdf[L],left,max(oldpoints))
        assert row['removed_upper_staircase_slack']==removed_staircase_slack(oldpoints,merged,cdf[L])
        rows.append(dict(L=L,outside_success_upper=row['window_budget']['outside_success_contribution_upper']))
    output=dict(reference=references(),source_hashes=len(result['source_hashes']),new_point_coefficient_error_audits=len(widths),selected_secant_scaling_checks=algebra,
        old_points_retained=len(oldpoints),total_upper_points=len(merged),whole_recall_and_window_budgets_reconstructed=rows,
        no_third_beta_matrix_used=True,previous_lower_unchanged=True,max_comparison_integral_width_approximate=float(max(widths)),independent_proof_review=False)
    save(args.out,output);print(json.dumps(output,indent=2))


if __name__=='__main__':main()
