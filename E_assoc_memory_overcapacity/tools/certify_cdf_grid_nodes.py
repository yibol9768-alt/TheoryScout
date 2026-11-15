"""Add C0 point certificates at existing analytic CDF grid positions.

Old finite-bank frequencies choose locations only. Probabilities come from
integer interval integration of the frozen first-alpha matrix, with C0 shift.
The complete R25 two-family staircase is retained.
"""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_left
import argparse
import hashlib
import json
import time

import certify_weighted_chisquare as integer
from decompose_archived_tangent_loss import read_dv


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def merge(points):
    points=sorted(points.items());ts=[v[0] for v in points];heights=[v[1] for v in points]
    for j in range(len(heights)-2,-1,-1):heights[j]=max(heights[j],heights[j+1])
    jumps=[heights[j]-(heights[j+1] if j+1<len(heights) else 0) for j in range(len(heights))]
    return ts,heights,jumps


def expectation(points,cdf):
    ts,_,jumps=merge(points);return sum(w*cdf[t] for t,w in zip(ts,jumps))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=8);args=p.parse_args();assert 1<=args.count<=16
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic()
    groups=[('R13',['revisions/r11_anchor_certificate_v2/SCOPE.json','revisions/r11_anchor_certificate_v2/ERRORS.json',
                   'revisions/r11_anchor_certificate_v2/actual_spectrum.json','revisions/r13_scatter_checks/CHECKS.json',
                   'tools/certify_weighted_chisquare.py']),
            ('R23',['revisions/r23_cdf_dense8/RESULT.json']),
            ('R25',['revisions/r25_second_anchor_nodes/RESULT.json'])]
    protected=[]
    for revision,names in groups:
        receipt=read(root/f'revisions/{revision}_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
        for name in names:assert sha(root/name)==manifest['files'][name];protected.append(name)
    bank=root/'revisions/r8_random_thresholds';finished=read(bank/'FINISHED.json')
    assert sha(bank/'D_V.npy')==finished['files']['D_V.npy'];protected.append('revisions/r8_random_thresholds/D_V.npy')
    scope=read(root/groups[0][1][0]);errors=read(root/groups[0][1][1]);spectrum=read(root/groups[0][1][2])
    n=scope['n'];m=n-1;q=len(spectrum['eigenvalues']);a=F(scope['center_anchor']);alpha=F(scope['alpha']);gamma=4*alpha*m
    eta=F(read(root/'revisions/r13_scatter_checks/CHECKS.json')['eta_upper']);epsilon=F(errors['total'])
    baseline=read(root/'revisions/r25_second_anchor_nodes/RESULT.json');basepoints=dict(zip(map(F,baseline['t']),map(F,baseline['merged_heights'])))
    sourceCDF=read(root/'revisions/r23_cdf_dense8/RESULT.json');cdf={}
    for row in sourceCDF['rows']:
        last=row['sweeps'][-1]
        cdf[row['L']]=dict(lower=dict(zip(map(F,row['t']),map(F,last['lower_CDF']))),
                          upper=dict(zip(map(F,row['t']),map(F,last['upper_CDF']))))
    for row in baseline['rows']:assert expectation(basepoints,cdf[row['L']]['lower'])==F(row['conditional_recall_lower'])
    # Frequencies are a location-selection heuristic, never a point certificate.
    M,D,V=read_dv(bank/'D_V.npy');assert M==131072 and all(v>0 for v in V)
    cuts=sorted(max(F(0),4*alpha*(d-alpha*v-alpha*(d-a)**2/m)) for d,v in zip(D,V))
    candidates={t:F(M-bisect_left(cuts,t*t),M) for t in cdf[1]['lower'] if t not in basepoints}
    predicted=basepoints.copy();selected=[]
    for index in range(args.count):
        old=expectation(predicted,cdf[1]['lower']);values=[]
        for t,estimate in candidates.items():
            trial=predicted.copy();trial[t]=estimate;score=expectation(trial,cdf[1]['lower'])
            values.append((score-old,t,estimate,score))
        gain,t,estimate,score=max(values);assert gain>0
        selected.append(dict(selection_order=index+1,t=str(t),approximate_t=float(t),
            empirical_probability_heuristic=str(estimate),predicted_increment=str(gain),predicted_lower=str(score)))
        predicted[t]=estimate;del candidates[t]
    save(args.out/'DESIGN.json',dict(new_point_count=args.count,first_alpha=str(alpha),center=str(a),
        source_hashes={name:sha(root/name) for name in protected},selection=selected,selection_target_L=1,
        selection_only_uses_old_sample_frequencies=True,selection_probabilities_are_NOT_certified=True,
        no_new_random_samples=True,baseline_R25_preserved=True))
    print('Selected locations',[(v['selection_order'],v['approximate_t']) for v in selected],flush=True)
    pi=integer.pi_interval();gauss=integer.gauss8();I=integer.I;points=[]
    for node in selected:
        t=F(node['t']);gap=I(n)*I(eta)*(1+I(t)/I(m).sqrt());gap_upper=F(gap.hi,integer.SCALE)
        constant=-4*alpha*alpha*a*a-m*t*t-gamma*gap_upper
        coefficients=[F(v)+constant/n-epsilon for v in spectrum['eigenvalues']]+[constant/n]
        record,prob=integer.certify(coefficients,[1]*q+[n-q],pi,gauss)
        stem=f"node{node['selection_order']:02d}";save(args.out/(stem+'_integral.json'),record)
        lower=max(F(0),F(prob.lo,integer.SCALE))
        value=dict(selection_order=node['selection_order'],t=str(t),target_probability_lower=str(lower),
            score_gap_upper=str(gap_upper),shifted_constant=str(constant),comparison_event_probability=prob.record(),
            integral_file=stem+'_integral.json',empirical_probability_heuristic=node['empirical_probability_heuristic'])
        points.append(value);save(args.out/(stem+'.json'),value)
        print(stem,'t',float(t),'C0 LOWER',float(lower),flush=True)
    merged=basepoints.copy()
    for node in points:merged[F(node['t'])]=max(merged.get(F(node['t']),F(0)),F(node['target_probability_lower']))
    ts,heights,jumps=merge(merged);rows=[]
    for length in [1,5]:
        bottom=expectation(merged,cdf[length]['lower']);top=expectation(merged,cdf[length]['upper'])
        previous=next(r for r in baseline['rows'] if r['L']==length);old=F(previous['conditional_recall_lower']);assert bottom>=old
        rows.append(dict(L=length,conditional_recall_lower=str(bottom),approximate=float(bottom),
            baseline_R25_lower=str(old),gain=str(bottom-old),staircase_upper=str(top),upper_is_not_target_recall_upper=True))
    result=dict(new_C0_nodes=points,rows=rows,t=list(map(str,ts)),merged_heights=list(map(str,heights)),jumps=list(map(str,jumps)),
        baseline_preserved=True,new_matrix_families=0,new_CDF_integrals=0,new_samples=0,
        source_hashes={name:sha(root/name) for name in protected},wall_seconds=time.monotonic()-start,
        target='R13 exact represented C0 ideal Adam2 / complete random-threshold conditional LOWER',
        selection_frequencies_not_used_as_certificates=True,independent_review=False,
        scope='New first-alpha point integrals; R25 two-family bounds retained; R23 analytic CDF inherited')
    save(args.out/'RESULT.json',result)
    print(json.dumps(dict(lower=[dict(L=r['L'],lower=r['approximate']) for r in rows],wall_seconds=result['wall_seconds']),indent=2),flush=True)


if __name__=='__main__':main()
