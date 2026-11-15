"""Audit the new C0 node coefficients and two-family CDF propagation."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import hashlib
import json

import certify_weighted_chisquare as integer


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--anchor',type=Path,required=True)
    p.add_argument('--nodes',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];s=read(args.anchor/'SCOPE.json')
    result=read(args.nodes/'RESULT.json');spectrum=read(args.anchor/'actual_spectrum.json');errors=read(args.anchor/'ERRORS.json')
    native=read(args.anchor/'checks/CHECKS.json');assert native['repeated_kernels'] and native['exact_coefficient_checks'] and native['exact_scalar_and_constant_checks']
    for name,digest in result['source_new_anchor_files'].items():assert sha(args.anchor/name)==digest
    n=s['n'];m=n-1;q=s['block_dimension'];alpha=F(s['alpha']);a=F(s['center_anchor']);gamma=4*alpha*m
    new=(args.anchor/'actual_construction.bin').read_bytes();old=(root/'revisions/r11_anchor_certificate_v2/actual_construction.bin').read_bytes()
    assert new[8:8+8*q*q]==old[8:8+8*q*q]
    eta=F(read(root/'revisions/r13_scatter_checks/CHECKS.json')['eta_upper']);assert eta==F(result['C0_eta_unchanged'])
    epsilon=F(errors['total']);assert epsilon==F(errors['construction'])+F(errors['spectral'])
    widths=[];better=0;I=integer.I
    assert len(result['new_C0_nodes'])==9
    for v in result['new_C0_nodes']:
        t=F(v['t']);gap=I(n)*I(eta)*(1+I(t)/I(m).sqrt());upper=F(gap.hi,integer.SCALE)
        constant=-4*alpha*alpha*a*a-m*t*t-gamma*upper
        assert constant==F(v['shifted_constant']) and upper==F(v['score_gap_upper'])
        integral=read(args.nodes/v['integral_file'])
        assert list(map(F,integral['coefficients']))==[F(x)+constant/n-epsilon for x in spectrum['eigenvalues']]+[constant/n]
        assert integral['counts']==[1]*q+[n-q]
        prob=integral['probability'];arith=integral['arithmetic_only'];f=integral['finite_quadrature_error'];tail=integral['infinite_tail_error']
        assert int(prob['lower_integer'])==int(arith['lower_integer'])-int(f['upper_integer'])-int(tail['upper_integer'])
        assert int(prob['upper_integer'])==int(arith['upper_integer'])+int(f['upper_integer'])+int(tail['upper_integer'])
        lower=max(F(0),F(int(prob['lower_integer']),integer.SCALE));assert lower==F(v['target_probability_lower']) and lower<=1
        width=F(int(prob['upper_integer'])-int(prob['lower_integer']),integer.SCALE);assert width<=F(integral['target_width']);widths.append(width)
        better+=int(lower>F(v['old_alpha_lower']))
    oldpoints=read(root/'revisions/r20_threshold_family/RESULT.json')['certified_nodes']
    assert sha(root/'revisions/r20_threshold_family/RESULT.json')==result['source_old_point_sha256']
    cdf=read(root/'revisions/r23_cdf_dense8/RESULT.json');assert sha(root/'revisions/r23_cdf_dense8/RESULT.json')==result['source_CDF_sha256']
    ts=sorted({F(v['t']) for v in oldpoints});heights=[]
    for t in ts:
        values=[F(v['target_probability_lower']) for v in oldpoints+result['new_C0_nodes'] if F(v['t'])==t]
        heights.append(max(values))
    for j in range(len(ts)-2,-1,-1):heights[j]=max(heights[j],heights[j+1])
    jumps=[heights[j]-(heights[j+1] if j+1<len(ts) else 0) for j in range(len(ts))]
    assert list(map(F,result['merged_heights']))==heights and list(map(F,result['jumps']))==jumps
    summary=[]
    for row in result['rows']:
        source=next(v for v in cdf['rows'] if v['L']==row['L']);lookup={F(t):i for i,t in enumerate(source['t'])};last=source['sweeps'][-1]
        bottom=sum(w*F(last['lower_CDF'][lookup[t]]) for t,w in zip(ts,jumps))
        top=sum(w*F(last['upper_CDF'][lookup[t]]) for t,w in zip(ts,jumps))
        assert bottom==F(row['conditional_recall_lower']) and top==F(row['merged_staircase_upper'])
        assert bottom>=F(row['old_R23_lower']) and bottom-F(row['old_R23_lower'])==F(row['gain'])
        floor=F(bottom.numerator*10**8//bottom.denominator,10**8)
        summary.append(dict(L=row['L'],conditional_recall_lower_floor8=f'{float(floor):.8f}',approximate=float(bottom),
            gain_pp=float(100*F(row['gain'])),staircase_upper_approximate=float(top)))
    out=dict(identical_source_B_bytes=True,C0_eta_unchanged=True,new_node_coefficient_audits=9,
        node_integral_width_max_approximate=float(max(widths)),new_nodes_better_than_old=better,points_kept_old_if_better=True,
        merged_jump_sum_audits=2,rows=summary,no_new_CDF_integrals=True,no_new_samples=True,
        source_result_sha256=sha(args.nodes/'RESULT.json'),scope='Conditional represented-C0 LOWER; comparison-event widths are not target recall errors; no independent proof review')
    with (args.out/'CHECKS.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
