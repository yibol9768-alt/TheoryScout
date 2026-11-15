"""Refine actual C0 recall upper staircases at existing analytic CDF nodes.

Old float samples select locations/families only. All new point bounds use
integer probability integration; R27 points and actual lower bounds remain.
"""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_left
from datetime import datetime,timezone
import argparse
import hashlib
import json
import time

import certify_weighted_chisquare as integer
from decompose_archived_tangent_loss import read_dv
from certify_recall_upper import upper_expectation


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def closure(row):
    t=list(map(F,row['t']));lo=[];previous=F(0)
    for x in map(F,row['sweeps'][-1]['lower_CDF']):previous=max(previous,x);lo.append(previous)
    hi=list(map(F,row['sweeps'][-1]['upper_CDF']))
    for j in range(len(hi)-2,-1,-1):hi[j]=min(hi[j],hi[j+1])
    assert all(0<=a<=b<=1 for a,b in zip(lo,hi))
    return dict(lower=dict(zip(t,lo)),upper=dict(zip(t,hi)))


def removed_staircase_slack(old,new,cdf):
    knots=sorted(set(old)|set(new));a=b=F(1);bottom=top=F(0);pieces=[]
    for j,t in enumerate(knots):
        a=min(a,old.get(t,F(1)));b=min(b,new.get(t,F(1)));gap=a-b;assert gap>=0
        if j+1<len(knots):
            nxt=knots[j+1];ml=max(F(0),cdf['lower'][nxt]-cdf['upper'][t]);mu=cdf['upper'][nxt]-cdf['lower'][t]
        else:ml=1-cdf['upper'][t];mu=1-cdf['lower'][t]
        bottom+=gap*ml;top+=gap*mu
        pieces.append(dict(left=str(t),right=str(knots[j+1]) if j+1<len(knots) else None,
            plateau_difference=str(gap),mass_lower=str(ml),mass_upper=str(mu)))
    return dict(lower=str(bottom),upper=str(top),approximate=[float(bottom),float(top)],pieces=pieces,
        scope='E[g_R27-g_refined] for left-continuous target upper staircases, not a change of actual recall')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=8)
    args=p.parse_args();assert 1<=args.count<=16;args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];start=time.monotonic();protected=[]
    groups=[('R13',['revisions/r11_anchor_certificate_v2/SCOPE.json','revisions/r11_anchor_certificate_v2/ERRORS.json','revisions/r11_anchor_certificate_v2/actual_spectrum.json','revisions/r13_scatter_checks/CHECKS.json','tools/certify_weighted_chisquare.py']),
            ('R25',['revisions/r25_second_anchor/SCOPE.json','revisions/r25_second_anchor/ERRORS.json','revisions/r25_second_anchor/actual_spectrum.json']),
            ('R23',['revisions/r23_cdf_dense8/RESULT.json']),
            ('R27',['revisions/r27_recall_upper/RESULT.json','revisions/r27_recall_upper/BAD_EVENTS.json','revisions/r8_random_thresholds/D_V.npy','tools/decompose_archived_tangent_loss.py','tools/audit_threshold_cdf_transfer.py','tools/certify_recall_upper.py'])]
    for revision,names in groups:
        receipt=read(root/f'revisions/{revision}_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
        for name in names:assert sha(root/name)==manifest['files'][name];protected.append(name)
    baseline=read(root/'revisions/r27_recall_upper/RESULT.json');bad=read(root/'revisions/r27_recall_upper/BAD_EVENTS.json')
    scope=read(root/'revisions/r11_anchor_certificate_v2/SCOPE.json');n=scope['n'];m=n-1;q=scope['block_dimension'];a=F(scope['center_anchor'])
    l0=F(bad['l0']);u0=F(bad['u0']);radius=F(bad['radius']);delta=F(bad['union_upper'])
    assert (l0,u0,radius,bad['n'],F(bad['center']))==(F(7,8),F(47,40),F(6),n,a)
    assert delta==F(baseline['global_bad_event_upper'])
    eta=F(read(root/'revisions/r13_scatter_checks/CHECKS.json')['eta_upper']);families=[]
    for name,directory in [('first',root/'revisions/r11_anchor_certificate_v2'),('second',root/'revisions/r25_second_anchor')]:
        s=read(directory/'SCOPE.json');assert (s['n'],s['block_dimension'],F(s['center_anchor']))==(n,q,a)
        families.append(dict(name=name,beta=F(s['alpha']),spectrum=read(directory/'actual_spectrum.json'),epsilon=F(read(directory/'ERRORS.json')['total'])))
    sourceCDF=read(root/'revisions/r23_cdf_dense8/RESULT.json');cdf={r['L']:closure(r) for r in sourceCDF['rows']}
    oldpoints={F(v['t']):F(v['target_C0_probability_upper']) for v in baseline['point_uppers']}
    for row in sourceCDF['rows']:
        raw=dict(zip(map(F,row['t']),map(F,row['sweeps'][-1]['upper_CDF'])))
        assert upper_expectation(oldpoints,raw)==F(next(r for r in baseline['rows'] if r['L']==row['L'])['conditional_recall_upper'])
    N,D,V=read_dv(root/'revisions/r8_random_thresholds/D_V.npy');assert N==131072 and all(v>0 for v in V)
    candidate_options={t:[] for t in cdf[1]['upper'] if t not in oldpoints};I=integer.I
    for family in families:
        beta=family['beta'];scores=sorted(d-beta*v-beta*(d-a)**2/m+beta*radius*radius/m for d,v in zip(D,V))
        for t,options in candidate_options.items():
            width=t/beta
            if width<=u0:continue
            l=min(l0,width-u0);u=width-l
            gap=I(n)*I(eta)*(1+I(t)/I(m).sqrt());gap_upper=F(gap.hi,integer.SCALE)
            count=N-bisect_left(scores,beta*l*u-gap_upper);estimate=min(F(1),F(count,N)+delta)
            options.append(dict(family=family['name'],estimated_upper=str(estimate),count=count,l=str(l),u=str(u)))
    candidates={t:min(v,key=lambda r:(F(r['estimated_upper']),r['family'])) for t,v in candidate_options.items() if v}
    predicted=oldpoints.copy();selected=[]
    for order in range(args.count):
        previous=sum(upper_expectation(predicted,cdf[L]['upper']) for L in [1,5]);options=[]
        for t,choice in candidates.items():
            trial=predicted.copy();trial[t]=F(choice['estimated_upper'])
            objective=sum(upper_expectation(trial,cdf[L]['upper']) for L in [1,5]);options.append((previous-objective,t,choice))
        gain,t,choice=max(options,key=lambda v:(v[0],v[1]));assert gain>0
        selected.append(dict(order=order+1,t=str(t),predicted_gain=str(gain),**choice));predicted[t]=F(choice['estimated_upper']);del candidates[t]
    save(args.out/'DESIGN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),selection=selected,selection_sample_count=N,
        selection_uses_old_secant_event_frequencies_only=True,heuristic_probabilities_NOT_certificates=True,
        all_old_R27_points_retained=True,source_hashes={name:sha(root/name) for name in protected}))
    print('Selected',[(r['order'],float(F(r['t'])),r['family']) for r in selected],flush=True)
    pi=integer.pi_interval();gauss=integer.gauss8();points=[]
    for choice in selected:
        family=next(f for f in families if f['name']==choice['family']);beta=family['beta'];t=F(choice['t']);l=F(choice['l']);u=F(choice['u'])
        assert 0<l<=l0<u0<=u and l+u==t/beta
        gap=I(n)*I(eta)*(1+I(t)/I(m).sqrt());gu=F(gap.hi,integer.SCALE);gamma=4*beta*m
        constant=4*beta*beta*(radius*radius-a*a)-4*beta*beta*m*l*u+gamma*gu
        coefficients=[F(v)+constant/n+family['epsilon'] for v in family['spectrum']['eigenvalues']]+[constant/n]
        record,prob=integer.certify(coefficients,[1]*q+[n-q],pi,gauss)
        event_upper=min(F(1),max(F(0),F(prob.hi,integer.SCALE)));target_upper=min(F(1),event_upper+delta)
        stem=f"node{choice['order']:02d}";save(args.out/(stem+'_integral.json'),record)
        value=dict(order=choice['order'],t=str(t),family=family['name'],beta=str(beta),l=str(l),u=str(u),shifted_constant=str(constant),
            score_gap_upper=str(gu),quadratic_event_upper=str(event_upper),target_C0_probability_upper=str(target_upper),integral_file=stem+'_integral.json')
        points.append(value);save(args.out/(stem+'.json'),value);print(stem,'t',float(t),'C0 UPPER',float(target_upper),flush=True)
    merged=oldpoints.copy()
    for p in points:
        t=F(p['t']);merged[t]=min(merged.get(t,F(1)),F(p['target_C0_probability_upper']))
    rows=[]
    for length in [1,5]:
        old=next(r for r in baseline['rows'] if r['L']==length);upper=upper_expectation(merged,cdf[length]['upper']);lower=F(old['conditional_recall_lower']);old_upper=F(old['conditional_recall_upper'])
        assert lower<=upper<=old_upper
        rows.append(dict(L=length,conditional_recall_lower=str(lower),conditional_recall_upper=str(upper),approximate=[float(lower),float(upper)],
            old_R27_upper=str(old_upper),certified_upper_endpoint_decrease=str(old_upper-upper),decrease_pp_approximate=100*float(old_upper-upper),
            removed_upper_staircase_slack=removed_staircase_slack(oldpoints,merged,cdf[length])))
    result=dict(rows=rows,new_point_uppers=points,all_point_uppers={str(t):str(v) for t,v in sorted(merged.items())},
        old_point_count=len(oldpoints),new_point_count=len(points),global_bad_event_upper=str(delta),
        source_hashes={name:sha(root/name) for name in protected},new_matrices=0,new_samples=0,new_CDF_integrals=0,new_probability_integrals=len(points),
        original_R26_lower_preserved=True,all_R27_upper_points_retained=True,independent_proof_review=False,statistical_confidence_interval=False,
        scope='Actual represented-C0 ideal Adam2 one-query full random-threshold recall enclosure; inherited R13 coupling, R27 tails, R23 CDF and spectral assumptions',wall_seconds=time.monotonic()-start)
    save(args.out/'RESULT.json',result);print(json.dumps(dict(bounds=[dict(L=r['L'],bounds=r['approximate'],upper_drop_pp=r['decrease_pp_approximate']) for r in rows],wall_seconds=result['wall_seconds']),indent=2),flush=True)


if __name__=='__main__':main()
