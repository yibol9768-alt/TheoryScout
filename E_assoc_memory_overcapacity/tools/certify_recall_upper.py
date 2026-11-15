"""Full conditional C0 recall UPPER via secants and certified bad events.

Reuses two anchored matrices; new D/W spectra certify a common good event.
Old empirical scores select node positions only. No Monte Carlo probabilities.
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


def upper_expectation(points,cdf):
    if not points:return F(1)
    items=sorted(points.items());previous=F(1);total=F(0)
    for t,value in items:
        current=min(previous,value);total+=(previous-current)*cdf[t];previous=current
    return previous+total


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrices',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--nodes',type=int,default=8);args=p.parse_args();assert 1<=args.nodes<=16
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic()
    assert read(args.matrices/'checks/CHECKS.json')['native_kernel_repetitions']==4
    protected=[]
    groups=[('R13',['revisions/r11_anchor_certificate_v2/SCOPE.json','revisions/r11_anchor_certificate_v2/ERRORS.json',
                   'revisions/r11_anchor_certificate_v2/actual_spectrum.json','revisions/r11_anchor_certificate_v2/actual_construction.bin',
                   'revisions/r13_scatter_checks/CHECKS.json','tools/certify_weighted_chisquare.py']),
            ('R25',['revisions/r25_second_anchor/SCOPE.json','revisions/r25_second_anchor/ERRORS.json',
                    'revisions/r25_second_anchor/actual_spectrum.json','revisions/r25_second_anchor/PROVENANCE.json']),
            ('R23',['revisions/r23_cdf_dense8/RESULT.json']),('R26',['revisions/r26_grid_nodes/RESULT.json'])]
    for rev,names in groups:
        receipt=read(root/f'revisions/{rev}_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
        for name in names:assert sha(root/name)==manifest['files'][name];protected.append(name)
    s1=read(root/'revisions/r11_anchor_certificate_v2/SCOPE.json');n=s1['n'];m=n-1;q=s1['block_dimension'];a=F(s1['center_anchor'])
    ms=read(args.matrices/'SCOPE.json');assert (ms['n'],ms['q'])==(n,q)
    assert ms['source_B_sha256']==sha(root/'revisions/r11_anchor_certificate_v2/actual_construction.bin')
    for case in ms['cases']:
        expected=(F(1),F(0)) if case['label']=='D' else (F(0),-F(n,m))
        assert (F(case['A']),F(case['Dcoef']))==expected
    prov=read(root/'revisions/r25_second_anchor/PROVENANCE.json')
    assert prov['source_hashes']['revisions/r11_anchor_certificate_v2/actual_construction.bin']==ms['source_B_sha256']
    l0=F(7,8);u0=F(47,40);radius=F(6);M=abs(a)+radius;eta=F(read(root/'revisions/r13_scatter_checks/CHECKS.json')['eta_upper'])
    pi=integer.pi_interval();gauss=integer.gauss8();I=integer.I
    bad_specs=[('D_low','D',-1,a-radius),('D_high','D',1,a+radius),
               ('W_low','W',-1,l0*l0+M*M/m),('W_high','W',1,u0*u0)]
    bad=[]
    for label,matrix,sign,threshold in bad_specs:
        directory=args.matrices/matrix;spectrum=read(directory/'actual_spectrum.json');epsilon=F(read(directory/'ERRORS.json')['total'])
        constant=-sign*threshold
        coefficients=[sign*F(v)+constant/n+epsilon for v in spectrum['eigenvalues']]+[constant/n]
        record,prob=integer.certify(coefficients,[1]*q+[n-q],pi,gauss)
        assert prob.lo<=integer.SCALE and prob.hi>=0
        upper=min(F(1),max(F(0),F(prob.hi,integer.SCALE)))
        save(args.out/(label+'_integral.json'),record)
        bad.append(dict(label=label,matrix=matrix,sign=sign,threshold=str(threshold),event_upper=str(upper),integral_file=label+'_integral.json'))
        print(label,'tail UPPER',float(upper),flush=True)
    delta=min(F(1),sum(F(v['event_upper']) for v in bad));assert delta<1
    save(args.out/'BAD_EVENTS.json',dict(l0=str(l0),u0=str(u0),radius=str(radius),center=str(a),n=n,
        events=bad,union_upper=str(delta),scope='Exact represented Bhat bad event; no independence between D/W assumed'))
    families=[]
    for name,directory in [('first',root/'revisions/r11_anchor_certificate_v2'),('second',root/'revisions/r25_second_anchor')]:
        scope=read(directory/'SCOPE.json');assert (scope['n'],scope['block_dimension'],F(scope['center_anchor']))==(n,q,a)
        families.append(dict(name=name,beta=F(scope['alpha']),spectrum=read(directory/'actual_spectrum.json'),
                             epsilon=F(read(directory/'ERRORS.json')['total'])))
    sourceCDF=read(root/'revisions/r23_cdf_dense8/RESULT.json');cdf={}
    for row in sourceCDF['rows']:cdf[row['L']]=dict(zip(map(F,row['t']),map(F,row['sweeps'][-1]['upper_CDF'])))
    bank=root/'revisions/r8_random_thresholds';frozen=read(bank/'FINISHED.json');assert sha(bank/'D_V.npy')==frozen['files']['D_V.npy']
    protected.append('revisions/r8_random_thresholds/D_V.npy')
    count,D,V=read_dv(bank/'D_V.npy');assert all(v>0 for v in V)
    cuts=sorted(d*d/v if d>=0 else F(0) for d,v in zip(D,V));candidates={t:F(count-bisect_left(cuts,t*t),count) for t in cdf[1]}
    predicted={};selected=[]
    for order in range(args.nodes):
        old=sum(upper_expectation(predicted,cdf[L]) for L in [1,5]);options=[]
        for t,estimate in candidates.items():
            trial=predicted.copy();trial[t]=estimate;score=sum(upper_expectation(trial,cdf[L]) for L in [1,5]);options.append((old-score,t,estimate))
        gain,t,estimate=max(options);assert gain>0
        selected.append(dict(order=order+1,t=str(t),approximate_t=float(t),empirical_selection_probability=str(estimate),predicted_decrement=str(gain)))
        predicted[t]=estimate;del candidates[t]
    save(args.out/'DESIGN.json',dict(selection=selected,nodes=args.nodes,selection_target='Sum of L1/L5 heuristic upper expectations',
        empirical_probabilities_are_NOT_certificates=True,source_hashes={name:sha(root/name) for name in protected}))
    print('Upper node selection',[(v['order'],v['approximate_t']) for v in selected],'bad budget',float(delta),flush=True)
    points=[]
    for choice in selected:
        t=F(choice['t']);eligible=[f for f in families if t/f['beta']>u0];assert eligible
        family=min(eligible,key=lambda f:abs(t/f['beta']-(l0+u0)))
        beta=family['beta'];width=t/beta;l=min(l0,width-u0);u=width-l
        assert 0<l<=l0<u0<=u and l+u==t/beta
        gamma=4*beta*m;gap=I(n)*I(eta)*(1+I(t)/I(m).sqrt());gap_upper=F(gap.hi,integer.SCALE)
        constant=4*beta*beta*(radius*radius-a*a)-4*beta*beta*m*l*u+gamma*gap_upper
        coefficients=[F(v)+constant/n+family['epsilon'] for v in family['spectrum']['eigenvalues']]+[constant/n]
        record,prob=integer.certify(coefficients,[1]*q+[n-q],pi,gauss)
        event_upper=min(F(1),max(F(0),F(prob.hi,integer.SCALE)));target_upper=min(F(1),event_upper+delta)
        stem=f"upper_node{choice['order']:02d}";save(args.out/(stem+'_integral.json'),record)
        value=dict(order=choice['order'],t=str(t),family=family['name'],beta=str(beta),l=str(l),u=str(u),
            shifted_constant=str(constant),score_gap_upper=str(gap_upper),quadratic_event_upper=str(event_upper),
            target_C0_probability_upper=str(target_upper),integral_file=stem+'_integral.json')
        points.append(value);save(args.out/(stem+'.json'),value)
        print(stem,'t',float(t),'C0 UPPER',float(target_upper),flush=True)
    upperpoints={F(v['t']):F(v['target_C0_probability_upper']) for v in points};baseline=read(root/'revisions/r26_grid_nodes/RESULT.json');rows=[]
    for length in [1,5]:
        upper=upper_expectation(upperpoints,cdf[length]);lower=F(next(v for v in baseline['rows'] if v['L']==length)['conditional_recall_lower'])
        assert lower<=upper<=1
        rows.append(dict(L=length,conditional_recall_lower=str(lower),conditional_recall_upper=str(upper),
            approximate=[float(lower),float(upper)],width_approximate=float(upper-lower),
            target_upper_is_not_merely_staircase_upper=True))
    result=dict(rows=rows,point_uppers=points,global_bad_event_upper=str(delta),new_samples=0,new_CDF_integrals=0,
        new_D_W_matrices=2,source_hashes={name:sha(root/name) for name in protected},
        source_bad_matrix_files={str(p.relative_to(args.matrices)):sha(p) for p in args.matrices.rglob('*') if p.is_file()},
        wall_seconds=time.monotonic()-start,statistical_confidence_event_used=False,independent_review=False,
        target='R13 exact represented C0 ideal Adam2 / one-query full random-threshold conditional recall',
        scope='Target lower inherited from R26; target upper from secants, certified bad events and analytic CDF; inherits R9-R13 assumptions')
    save(args.out/'RESULT.json',result);print(json.dumps(dict(enclosures=[dict(L=v['L'],bounds=v['approximate']) for v in rows],wall_seconds=result['wall_seconds']),indent=2),flush=True)


if __name__=='__main__':main()
