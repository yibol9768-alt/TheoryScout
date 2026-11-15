"""Extend R35's actual C0 upper envelope and certify threshold-window tails.

Four right-tail points reuse the first alpha matrix. The beta-family screen
is only a heuristic design record; new probabilities come from integration.
"""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime,timezone
import argparse
import hashlib
import json
import time

import certify_weighted_chisquare as integer
from certify_recall_upper import upper_expectation
from refine_recall_upper_grid import closure,removed_staircase_slack


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def window_budget(lower,upper,points,cdf,left,right):
    assert left<right and right==max(points)
    terminal=min(points.values())
    left_upper=cdf['upper'][left]
    right_upper=terminal*(1-cdf['lower'][right])
    outside=left_upper+right_upper
    return dict(left=str(left),right=str(right),terminal_target_upper=str(terminal),
        left_success_contribution_upper=str(left_upper),right_success_contribution_upper=str(right_upper),
        outside_success_contribution_upper=str(outside),
        window_success_contribution_lower=str(max(F(0),lower-outside)),
        window_success_contribution_upper=str(min(upper,cdf['upper'][right]-cdf['lower'][left])),
        target_is_actual_success_contribution_not_just_threshold_tail=True,
        outside_contribution_not_mechanism_effect=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic();protected=[]
    groups=[('R13',['revisions/r11_anchor_certificate_v2/SCOPE.json','revisions/r11_anchor_certificate_v2/ERRORS.json','revisions/r11_anchor_certificate_v2/actual_spectrum.json','revisions/r13_scatter_checks/CHECKS.json','tools/certify_weighted_chisquare.py']),
            ('R35',['revisions/r35_upper_grid/RESULT.json','tools/refine_recall_upper_grid.py','tools/decompose_archived_tangent_loss.py','tools/audit_threshold_cdf_transfer.py','tools/certify_recall_upper.py']),
            ('R23',['revisions/r23_cdf_dense8/RESULT.json']),('R27',['revisions/r27_recall_upper/BAD_EVENTS.json'])]
    for revision,names in groups:
        receipt=read(root/f'revisions/{revision}_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
        for name in names:assert sha(root/name)==manifest['files'][name];protected.append(name)
    baseline=read(root/'revisions/r35_upper_grid/RESULT.json');scope=read(root/'revisions/r11_anchor_certificate_v2/SCOPE.json')
    n=scope['n'];m=n-1;q=scope['block_dimension'];a=F(scope['center_anchor']);beta=F(scope['alpha']);gamma=4*beta*m
    bad=read(root/'revisions/r27_recall_upper/BAD_EVENTS.json');l0=F(bad['l0']);u0=F(bad['u0']);R=F(bad['radius']);delta=F(bad['union_upper'])
    assert bad['n']==n and F(bad['center'])==a and delta==F(baseline['global_bad_event_upper'])
    eta=F(read(root/'revisions/r13_scatter_checks/CHECKS.json')['eta_upper'])
    spectrum=read(root/'revisions/r11_anchor_certificate_v2/actual_spectrum.json')['eigenvalues'];epsilon=F(read(root/'revisions/r11_anchor_certificate_v2/ERRORS.json')['total'])
    cdf={row['L']:closure(row) for row in read(root/'revisions/r23_cdf_dense8/RESULT.json')['rows']}
    assert sha(root/'revisions/r23_cdf_dense8/RESULT.json')==baseline['source_hashes']['revisions/r23_cdf_dense8/RESULT.json']
    oldpoints={F(t):F(v) for t,v in baseline['all_point_uppers'].items()};assert len(oldpoints)==16
    screenfile='revisions/r36_family_screen.json';screen=read(root/screenfile)
    chosen=next(row for row in screen['rows'] if F(row['beta'])==beta)['selection'];assert len(chosen)==4
    selected=[]
    for j,row in enumerate(chosen):
        t=F(row['t']);assert t>max(oldpoints) and t in cdf[1]['upper'] and t in cdf[5]['upper']
        S=t/beta;assert S>u0;l=min(l0,S-u0);u=S-l
        selected.append(dict(order=j+1,t=str(t),l=str(l),u=str(u)))
    hashes={name:sha(root/name) for name in protected};hashes[screenfile]=sha(root/screenfile)
    save(args.out/'DESIGN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),beta=str(beta),selection=selected,
        family_screen_heuristic_only=True,no_new_matrix_selected=True,all_R35_points_preserved=True,source_hashes=hashes))
    pi=integer.pi_interval();gauss=integer.gauss8();I=integer.I;points=[]
    for node in selected:
        t=F(node['t']);l=F(node['l']);u=F(node['u']);assert 0<l<=l0<u0<=u and l+u==t/beta
        gap=I(n)*I(eta)*(1+I(t)/I(m).sqrt());gu=F(gap.hi,integer.SCALE)
        constant=4*beta*beta*(R*R-a*a)-4*beta*beta*m*l*u+gamma*gu
        record,prob=integer.certify([F(v)+constant/n+epsilon for v in spectrum]+[constant/n],[1]*q+[n-q],pi,gauss)
        event=min(F(1),max(F(0),F(prob.hi,integer.SCALE)));target=min(F(1),event+delta)
        stem=f"node{node['order']:02d}";save(args.out/(stem+'_integral.json'),record)
        value=dict(**node,beta=str(beta),shifted_constant=str(constant),score_gap_upper=str(gu),quadratic_event_upper=str(event),target_C0_probability_upper=str(target),integral_file=stem+'_integral.json')
        points.append(value);save(args.out/(stem+'.json'),value)
        print(stem,'t',float(t),'C0 UPPER',float(target),flush=True)
    merged=oldpoints.copy()
    for node in points:merged[F(node['t'])]=F(node['target_C0_probability_upper'])
    rows=[];left=min(cdf[1]['upper']);right=max(merged)
    for length in [1,5]:
        old=next(row for row in baseline['rows'] if row['L']==length);lower=F(old['conditional_recall_lower']);upper=upper_expectation(merged,cdf[length]['upper']);previous=F(old['conditional_recall_upper'])
        assert lower<=upper<=previous
        budget=window_budget(lower,upper,merged,cdf[length],left,right)
        old_budget=window_budget(lower,previous,oldpoints,cdf[length],left,max(oldpoints))
        rows.append(dict(L=length,conditional_recall_lower=str(lower),conditional_recall_upper=str(upper),approximate=[float(lower),float(upper)],
            previous_R35_upper=str(previous),upper_endpoint_drop=str(previous-upper),window_budget=budget,old_R35_window_budget=old_budget,
            removed_upper_staircase_slack=removed_staircase_slack(oldpoints,merged,cdf[length])))
    result=dict(rows=rows,new_point_uppers=points,all_point_uppers={str(t):str(v) for t,v in sorted(merged.items())},
        old_points_retained=len(oldpoints),total_upper_points=len(merged),global_bad_event_upper=str(delta),source_hashes=hashes,
        new_probability_integrals=4,new_matrices=0,new_samples=0,new_CDF_integrals=0,original_actual_lower_preserved=True,
        independent_proof_review=False,probability_enclosures_not_statistical_CI=True,actual_random_energy_remainder_sign_unresolved=True,
        scope='Actual represented-C0 ideal Adam2 one-query recall enclosure and threshold-window success contributions; inherited prior numerical assumptions',wall_seconds=time.monotonic()-start)
    save(args.out/'RESULT.json',result)
    print(json.dumps(dict(rows=[dict(L=r['L'],bounds=r['approximate'],outside_success_upper_pp=100*float(F(r['window_budget']['outside_success_contribution_upper']))) for r in rows],wall_seconds=result['wall_seconds']),indent=2),flush=True)


if __name__=='__main__':main()
