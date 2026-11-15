"""Probability lower bounds for unit versus RMS-frozen energy, true C0 D law.

Q_1-Q_r is a specified counterfactual contrast, not the effect of actual
random V. No Gaussian score approximation or D/V independence is used.
"""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_left,bisect_right
from datetime import datetime,timezone
import argparse
import hashlib
import json
import math
import time

import certify_weighted_chisquare as integer

S=integer.SCALE
I=integer.I


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def restore(r):return I.raw(int(r['lower_integer']),int(r['upper_integer']))


def profile(row,scale):
    t=list(map(F,row['t']));lo=[];previous=F(0)
    for x in map(F,row['sweeps'][-1]['lower_CDF']):previous=max(previous,x);lo.append(previous)
    hi=list(map(F,row['sweeps'][-1]['upper_CDF']))
    for j in range(len(hi)-2,-1,-1):hi[j]=min(hi[j],hi[j+1])
    assert all(0<=a<=b<=1 for a,b in zip(lo,hi))
    knots=sorted(set(t+[scale*x for x in t]));cells=[]
    for a,b in zip(knots,knots[1:]):
        left=bisect_right(t,a)-1;right=bisect_left(t,b/scale)
        value=max(F(0),(lo[left] if left>=0 else F(0))-(hi[right] if right<len(t) else F(1)))
        cells.append(dict(left=str(a),right=str(b),minorant=str(value),lower_CDF_index=left,upper_CDF_index=right))
    peak=max(range(len(cells)),key=lambda j:F(cells[j]['minorant']));height=F(cells[peak]['minorant']);assert height>0
    layers=[]
    for fraction in [F(1,4),F(1,2)]:
        level=height*fraction;left=right=peak
        while left>0 and F(cells[left-1]['minorant'])>=level:left-=1
        while right+1<len(cells) and F(cells[right+1]['minorant'])>=level:right+=1
        layers.append(dict(left=cells[left]['left'],right=cells[right]['right'],height=str(level),first_cell=left,last_cell=right))
    assert F(layers[0]['left'])<=F(layers[1]['left'])<F(layers[1]['right'])<=F(layers[0]['right'])
    return dict(L=row['L'],cells=cells,layers=layers,peak_cell=peak,peak_height=str(height),scale=str(scale))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic();protected=[]
    groups=[('R28',['revisions/r28_conditional_moments_v2/INPUTS.json','revisions/r28_conditional_moments_v2/RESULT.json','tools/certify_weighted_chisquare.py']),
            ('R27',['revisions/r27_bad_matrices/SCOPE.json','revisions/r27_bad_matrices/D/ERRORS.json','revisions/r27_bad_matrices/D/actual_spectrum.json']),
            ('R23',['revisions/r23_cdf_dense8/RESULT.json','revisions/r21_sphere_cdf/RESULT.json'])]
    for revision,names in groups:
        receipt=read(root/f'revisions/{revision}_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
        for name in names:assert sha(root/name)==manifest['files'][name];protected.append(name)
    inputs=read(root/'revisions/r28_conditional_moments_v2/INPUTS.json');moments=read(root/'revisions/r28_conditional_moments_v2/RESULT.json')
    n=inputs['n'];q=inputs['q'];m=n-1;assert (n,q)==(moments['n'],moments['q'])
    scope=read(root/'revisions/r27_bad_matrices/SCOPE.json');assert (scope['n'],scope['q'])==(n,q)
    assert scope['source_B_sha256']==inputs['source_hashes']['revisions/r11_anchor_certificate_v2/actual_construction.bin']
    cdf=read(root/'revisions/r23_cdf_dense8/RESULT.json')
    assert sha(root/'revisions/r21_sphere_cdf/RESULT.json')==cdf['source_hashes']['revisions/r21_sphere_cdf/RESULT.json']
    assert read(root/'revisions/r21_sphere_cdf/RESULT.json')['m']==m
    scale=F(10281,10000);RMS=restore(moments['C0_moments']['mean_V']).sqrt();assert 1<scale<=F(RMS.lo,S)
    negative=F(1,1<<100)
    designs=[profile(row,scale) for row in cdf['rows']]
    for d in designs:assert sum(math.comb(m,k) for k in range(d['L']))*(1<<100)<=(1<<m)
    design=dict(created_utc=datetime.now(timezone.utc).isoformat(),n=n,q=q,scale_lower=str(scale),true_RMS=RMS.record(),
        negative_threshold_probability_upper=str(negative),profiles=designs,levels=['1/4','1/2'],
        selection_uses_only_certified_threshold_CDF=True,empirical_score_data_used=False,
        source_hashes={name:sha(root/name) for name in protected})
    save(args.out/'DESIGN.json',design)
    spectrum=read(root/'revisions/r27_bad_matrices/D/actual_spectrum.json')['eigenvalues'];assert len(spectrum)==q
    epsilon=F(read(root/'revisions/r27_bad_matrices/D/ERRORS.json')['total'])+F(inputs['eta'])
    pi=integer.pi_interval();gauss=integer.gauss8();nodes={};records=[]
    for d in designs:
        for layer in d['layers']:
            for side,key,sign in [('lower','left',-1),('upper','right',1)]:
                threshold=F(layer[key]);nodekey=(threshold,side)
                if nodekey in nodes:continue
                coefficients=[F(v)-threshold/n+sign*epsilon for v in spectrum]+[-threshold/n]
                integral,prob=integer.certify(coefficients,[1]*q+[n-q],pi,gauss,target=F(1,10**6))
                value=max(F(0),min(F(1),F(prob.lo if side=='lower' else prob.hi,S)))
                stem=f'node{len(records)+1:02d}';save(args.out/(stem+'_integral.json'),integral)
                record=dict(threshold=str(threshold),side=side,spectral_and_C0_error=str(epsilon),
                    C0_survival_bound=str(value),comparison_probability=prob.record(),
                    comparison_interval_is_NOT_C0_survival_interval=True,integral_file=stem+'_integral.json')
                nodes[nodekey]=record;records.append(record);save(args.out/(stem+'.json'),record)
                print(stem,'D >=',float(threshold),side,float(value),flush=True)
    rows=[]
    for d in designs:
        total=F(0);previous=F(0);layers=[]
        for layer in d['layers']:
            left=F(layer['left']);right=F(layer['right']);height=F(layer['height'])
            mass=max(F(0),F(nodes[(left,'lower')]['C0_survival_bound'])-F(nodes[(right,'upper')]['C0_survival_bound']))
            total+=(height-previous)*mass
            layers.append(dict(**layer,C0_interval_probability_lower=str(mass),jump_weight=str(height-previous)));previous=height
        bound=total-negative
        rows.append(dict(L=d['L'],layers=layers,minorant_expectation_lower=str(total),
            unit_minus_RMS_frozen_success_lower=str(bound),lower_approximate=float(bound),
            probability_point_lower_approximate=100*float(bound),lower_positive=bound>0))
    result=dict(rows=rows,n=n,q=q,true_RMS=RMS.record(),scale_minorant=str(scale),nodes=records,
        new_probability_integrals=len(records),new_matrices=0,new_samples=0,D_V_independence_assumed=False,
        Gaussian_correct_score_approximation=False,independent_proof_review=False,
        actual_random_energy_recall_change_NOT_bounded=True,actual_R27_recall_enclosures_unchanged=True,
        scope='Counterfactual success Q1=P_C0(D>=T) versus Qr=P_C0(D>=sqrt(EV)T); actual D law, not actual random-energy recall difference',
        wall_seconds=time.monotonic()-start)
    save(args.out/'RESULT.json',result);print(json.dumps(dict(rows=rows,wall_seconds=result['wall_seconds']),indent=2),flush=True)


if __name__=='__main__':main()
