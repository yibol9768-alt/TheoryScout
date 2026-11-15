"""Audit certified bad-event tails, C0 secant upper nodes and target bounds."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import hashlib
import json

import certify_weighted_chisquare as integer


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_integral(directory,filename,coefficients,counts):
    record=read(directory/filename);assert list(map(F,record['coefficients']))==coefficients and record['counts']==counts
    pr=record['probability'];ar=record['arithmetic_only'];fe=record['finite_quadrature_error'];tail=record['infinite_tail_error']
    assert int(pr['lower_integer'])==int(ar['lower_integer'])-int(fe['upper_integer'])-int(tail['upper_integer'])
    assert int(pr['upper_integer'])==int(ar['upper_integer'])+int(fe['upper_integer'])+int(tail['upper_integer'])
    assert int(pr['lower_integer'])<=integer.SCALE and int(pr['upper_integer'])>=0
    width=F(int(pr['upper_integer'])-int(pr['lower_integer']),integer.SCALE);assert width<=F(record['target_width'])
    return min(F(1),max(F(0),F(int(pr['upper_integer']),integer.SCALE))),width


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrices',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];result=read(args.input/'RESULT.json')
    for name,digest in result['source_hashes'].items():assert sha(root/name)==digest
    for name,digest in result['source_bad_matrix_files'].items():assert sha(args.matrices/name)==digest
    native=read(args.matrices/'checks/CHECKS.json');assert native['native_kernel_repetitions']==4
    scope=read(root/'revisions/r11_anchor_certificate_v2/SCOPE.json');n=scope['n'];m=n-1;q=scope['block_dimension'];a=F(scope['center_anchor'])
    bad=read(args.input/'BAD_EVENTS.json');l0=F(bad['l0']);u0=F(bad['u0']);R=F(bad['radius']);M=abs(a)+R
    assert (l0,u0,R,F(bad['center']),bad['n'])==(F(7,8),F(47,40),F(6),a,n)
    cases={v['label']:v for v in read(args.matrices/'SCOPE.json')['cases']};assert set(cases)=={'D','W'}
    assert (F(cases['D']['A']),F(cases['D']['Dcoef']))==(F(1),F(0))
    assert (F(cases['W']['A']),F(cases['W']['Dcoef']))==(F(0),-F(n,m))
    expected={'D_low':('D',-1,a-R),'D_high':('D',1,a+R),'W_low':('W',-1,l0*l0+M*M/m),'W_high':('W',1,u0*u0)}
    tails=[];widths=[];counts=[1]*q+[n-q]
    for event in bad['events']:
        matrix,sign,threshold=expected[event['label']]
        assert (event['matrix'],event['sign'],F(event['threshold']))==(matrix,sign,threshold)
        spectrum=read(args.matrices/matrix/'actual_spectrum.json');epsilon=F(read(args.matrices/matrix/'ERRORS.json')['total']);c=-sign*threshold
        upper,width=audit_integral(args.input,event['integral_file'],[sign*F(v)+c/n+epsilon for v in spectrum['eigenvalues']]+[c/n],counts)
        assert upper==F(event['event_upper']);tails.append(upper);widths.append(width)
    delta=min(F(1),sum(tails));assert delta==F(bad['union_upper'])==F(result['global_bad_event_upper'])
    eta=F(read(root/'revisions/r13_scatter_checks/CHECKS.json')['eta_upper']);families={}
    for name,directory in [('first',root/'revisions/r11_anchor_certificate_v2'),('second',root/'revisions/r25_second_anchor')]:
        s=read(directory/'SCOPE.json');assert (s['n'],s['block_dimension'],F(s['center_anchor']))==(n,q,a)
        families[name]=(F(s['alpha']),read(directory/'actual_spectrum.json'),F(read(directory/'ERRORS.json')['total']))
    assert len(result['point_uppers'])==8;points={}
    for v in result['point_uppers']:
        beta,spectrum,epsilon=families[v['family']];assert beta==F(v['beta'])
        t=F(v['t']);l=F(v['l']);u=F(v['u']);assert 0<l<=l0<u0<=u and l+u==t/beta
        gap=integer.I(n)*integer.I(eta)*(1+integer.I(t)/integer.I(m).sqrt());gu=F(gap.hi,integer.SCALE)
        c=4*beta*beta*(R*R-a*a)-4*beta*beta*m*l*u+4*beta*m*gu
        assert c==F(v['shifted_constant']) and gu==F(v['score_gap_upper'])
        event_upper,width=audit_integral(args.input,v['integral_file'],[F(x)+c/n+epsilon for x in spectrum['eigenvalues']]+[c/n],counts)
        assert event_upper==F(v['quadratic_event_upper']);target=min(F(1),event_upper+delta)
        assert target==F(v['target_C0_probability_upper']);points[t]=target;widths.append(width)
    cdf=read(root/'revisions/r23_cdf_dense8/RESULT.json');baseline=read(root/'revisions/r26_grid_nodes/RESULT.json');rows=[]
    for row in result['rows']:
        source=next(v for v in cdf['rows'] if v['L']==row['L']);lookup=dict(zip(map(F,source['t']),map(F,source['sweeps'][-1]['upper_CDF'])))
        previous=F(1);total=F(0)
        for t,value in sorted(points.items()):
            current=min(previous,value);assert t in lookup;total+=(previous-current)*lookup[t];previous=current
        upper=previous+total;lower=F(next(v for v in baseline['rows'] if v['L']==row['L'])['conditional_recall_lower'])
        assert (lower,upper)==(F(row['conditional_recall_lower']),F(row['conditional_recall_upper'])) and 0<=lower<=upper<=1
        low_integer=lower.numerator*10**8//lower.denominator;high_integer=-((-upper.numerator*10**8)//upper.denominator)
        rows.append(dict(L=row['L'],lower_floor8=f'{low_integer/10**8:.8f}',upper_ceil8=f'{high_integer/10**8:.8f}',
            bounds_approximate=[float(lower),float(upper)],width_pp_approximate=float(100*(upper-lower))))
    out=dict(bad_tail_coefficient_audits=4,point_upper_coefficient_audits=8,target_upper_jump_sum_audits=2,
        native_D_W_checks_reused=True,global_bad_event_upper=str(delta),global_bad_event_upper_approximate=float(delta),
        max_comparison_integral_width_approximate=float(max(widths)),rows=rows,
        target_upper_not_only_lower_staircase=True,probability_enclosure_not_statistical_confidence_interval=True,
        original_R26_lower_retained=True,source_result_sha256=sha(args.input/'RESULT.json'),
        scope='Represented C0 conditional full random-threshold probability; inherited numeric assumptions; not independent proof review')
    with (args.out/'CHECKS.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
