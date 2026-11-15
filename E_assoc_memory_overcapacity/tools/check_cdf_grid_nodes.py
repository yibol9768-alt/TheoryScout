"""Audit added first-alpha C0 nodes and retained R25 staircase."""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_left
import argparse
import hashlib
import json

import certify_weighted_chisquare as integer


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    result=read(args.input/'RESULT.json');design=read(args.input/'DESIGN.json')
    for name,digest in result['source_hashes'].items():assert sha(root/name)==digest
    assert result['source_hashes']==design['source_hashes'] and design['selection_probabilities_are_NOT_certified']
    scope=read(root/'revisions/r11_anchor_certificate_v2/SCOPE.json');spectrum=read(root/'revisions/r11_anchor_certificate_v2/actual_spectrum.json')
    epsilon=F(read(root/'revisions/r11_anchor_certificate_v2/ERRORS.json')['total']);eta=F(read(root/'revisions/r13_scatter_checks/CHECKS.json')['eta_upper'])
    n=scope['n'];m=n-1;q=len(spectrum['eigenvalues']);a=F(scope['center_anchor']);alpha=F(scope['alpha']);gamma=4*alpha*m
    baseline=read(root/'revisions/r25_second_anchor_nodes/RESULT.json');cdf=read(root/'revisions/r23_cdf_dense8/RESULT.json')
    base=dict(zip(map(F,baseline['t']),map(F,baseline['merged_heights'])));allCDF={F(t) for t in cdf['rows'][0]['t']}
    assert len(result['new_C0_nodes'])==design['new_point_count']==8
    merged=base.copy();widths=[];useful=0
    for value,choice in zip(result['new_C0_nodes'],design['selection']):
        assert value['t']==choice['t'] and value['selection_order']==choice['selection_order']
        t=F(value['t']);assert t in allCDF and t not in base
        gap=integer.I(n)*integer.I(eta)*(1+integer.I(t)/integer.I(m).sqrt());gap_upper=F(gap.hi,integer.SCALE)
        constant=-4*alpha*alpha*a*a-m*t*t-gamma*gap_upper
        assert constant==F(value['shifted_constant']) and gap_upper==F(value['score_gap_upper'])
        integral=read(args.input/value['integral_file'])
        assert list(map(F,integral['coefficients']))==[F(v)+constant/n-epsilon for v in spectrum['eigenvalues']]+[constant/n]
        assert integral['counts']==[1]*q+[n-q]
        pr=integral['probability'];ar=integral['arithmetic_only'];finite=integral['finite_quadrature_error'];tail=integral['infinite_tail_error']
        assert int(pr['lower_integer'])==int(ar['lower_integer'])-int(finite['upper_integer'])-int(tail['upper_integer'])
        assert int(pr['upper_integer'])==int(ar['upper_integer'])+int(finite['upper_integer'])+int(tail['upper_integer'])
        lower=max(F(0),F(int(pr['lower_integer']),integer.SCALE));assert lower==F(value['target_probability_lower']) and lower<=1
        width=F(int(pr['upper_integer'])-int(pr['lower_integer']),integer.SCALE);assert width<=F(integral['target_width']);widths.append(width)
        oldts=sorted(base);index=bisect_left(oldts,t);oldvalue=base[oldts[index]] if index<len(oldts) else F(0)
        useful+=int(lower>oldvalue);merged[t]=lower
    ts=sorted(merged);heights=[merged[t] for t in ts]
    for j in range(len(ts)-2,-1,-1):heights[j]=max(heights[j],heights[j+1])
    jumps=[heights[j]-(heights[j+1] if j+1<len(ts) else 0) for j in range(len(ts))]
    assert list(map(F,result['merged_heights']))==heights and list(map(F,result['jumps']))==jumps
    rows=[]
    oldts=sorted(base)
    old_on_union=[base[oldts[j]] if (j:=bisect_left(oldts,t))<len(oldts) else F(0) for t in ts]
    differences=[new-old for new,old in zip(heights,old_on_union)];assert all(d>=0 for d in differences)
    for row in result['rows']:
        source=next(r for r in cdf['rows'] if r['L']==row['L']);look={F(t):i for i,t in enumerate(source['t'])};last=source['sweeps'][-1]
        bottom=sum(w*F(last['lower_CDF'][look[t]]) for t,w in zip(ts,jumps))
        top=sum(w*F(last['upper_CDF'][look[t]]) for t,w in zip(ts,jumps))
        assert bottom==F(row['conditional_recall_lower']) and top==F(row['staircase_upper'])
        previous=next(r for r in baseline['rows'] if r['L']==row['L']);old=F(previous['conditional_recall_lower'])
        assert bottom>=old and bottom-old==F(row['gain'])
        floor=F(bottom.numerator*10**8//bottom.denominator,10**8)
        # Difference of two staircases is a nonnegative interval function,
        # not generally monotone. Bound each disjoint interval's CDF mass.
        gain_low=gain_high=F(0);previous_low=previous_high=F(0)
        for t,difference in zip(ts,differences):
            low=F(last['lower_CDF'][look[t]]);high=F(last['upper_CDF'][look[t]])
            gain_low+=difference*max(F(0),low-previous_high)
            gain_high+=difference*min(F(1),high-previous_low)
            previous_low,previous_high=low,high
        assert 0<=gain_low<=gain_high
        rows.append(dict(L=row['L'],lower_floor8=f'{float(floor):.8f}',approximate=float(bottom),gain_pp=float(100*(bottom-old)),
                         staircase_upper_approximate=float(top),
                         actual_staircase_expectation_gain_lower=str(gain_low),actual_staircase_expectation_gain_upper=str(gain_high),
                         actual_staircase_expectation_gain_pp_approximate=[float(100*gain_low),float(100*gain_high)]))
    result=dict(new_node_audits=8,merged_jump_sum_audits=2,new_nodes_above_previous_staircase=useful,
        baseline_two_families_preserved=True,cdf_nodes_already_certified=True,new_matrices=0,new_CDF_integrals=0,new_samples=0,
        max_comparison_integral_width_approximate=float(max(widths)),rows=rows,
        source_result_sha256=sha(args.input/'RESULT.json'),selection_predictions_not_certificates=True,
        scope='Represented C0 conditional LOWER; staircase upper is not target upper; no independent proof review')
    with (args.out/'CHECKS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
