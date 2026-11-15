"""Transfer the new fixed-alpha family to R13 C0, then reuse R23 CDF bounds."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import time

import certify_weighted_chisquare as integer
from check_anchor_certificate import construction_arrays


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic()
    # The native construction/spectral checker must have completed first.
    check=read(args.input/'checks/CHECKS.json');assert check['repeated_kernels']
    for revision,names in [('R13',['revisions/r13_scatter_checks/CHECKS.json','revisions/r11_anchor_certificate_v2/actual_construction.bin']),
                           ('R23',['revisions/r20_threshold_family/RESULT.json','revisions/r23_cdf_dense8/RESULT.json'])]:
        receipt=read(root/('revisions/'+revision+'_HISTORY.json'));manifest=read(root/receipt['path']/'manifest.json')
        for name in names:assert sha(root/name)==manifest['files'][name]
    q,b,_,_=construction_arrays(args.input/'actual_construction.bin');oq,old_b,_,_=construction_arrays(root/'revisions/r11_anchor_certificate_v2/actual_construction.bin');assert q==oq and b==old_b
    scope=read(args.input/'SCOPE.json');errors=read(args.input/'ERRORS.json');spectrum=read(args.input/'actual_spectrum.json')
    n=scope['n'];m=n-1;a=F(scope['center_anchor']);alpha=F(scope['alpha']);gamma=4*alpha*m
    epsilon=F(errors['total']);eta=F(read(root/'revisions/r13_scatter_checks/CHECKS.json')['eta_upper'])
    original=read(root/'revisions/r20_threshold_family/RESULT.json')['certified_nodes'];selected=[v for v in original if v['L']==5];assert len(selected)==9
    I=integer.I;pi=integer.pi_interval();gauss=integer.gauss8();points=[]
    for node in selected:
        t=F(node['t']);constant=-4*alpha*alpha*a*a-m*t*t
        gap=I(n)*I(eta)*(1+I(t)/I(m).sqrt());gap_upper=F(gap.hi,integer.SCALE);shifted=constant-gamma*gap_upper
        coefficients=[F(v)+shifted/n-epsilon for v in spectrum['eigenvalues']]+[shifted/n]
        record,prob=integer.certify(coefficients,[1]*q+[n-q],pi,gauss)
        stem=f"L5_node{node['index']:03d}";save(args.out/(stem+'_integral.json'),record)
        lower=max(F(0),F(prob.lo,integer.SCALE))
        value=dict(t=str(t),L=5,index=node['index'],alpha=str(alpha),target_probability_lower=str(lower),
            old_alpha_lower=node['target_probability_lower'],score_gap_upper=str(gap_upper),shifted_constant=str(shifted),
            comparison_event_probability=prob.record(),integral_file=stem+'_integral.json')
        points.append(value);save(args.out/(stem+'.json'),value)
        print(stem,'new C0 LOWER',float(lower),'old',float(F(node['target_probability_lower'])),flush=True)
    ts=sorted({F(v['t']) for v in original});old_heights=[max(F(v['target_probability_lower']) for v in original if F(v['t'])==t) for t in ts]
    merged=[max([old]+[F(v['target_probability_lower']) for v in points if F(v['t'])==t]) for t,old in zip(ts,old_heights)]
    for j in range(len(ts)-2,-1,-1):merged[j]=max(merged[j],merged[j+1])
    jumps=[merged[j]-(merged[j+1] if j+1<len(ts) else 0) for j in range(len(ts))]
    cdf=read(root/'revisions/r23_cdf_dense8/RESULT.json');rows=[]
    for row in cdf['rows']:
        lookup={F(t):i for i,t in enumerate(row['t'])};last=row['sweeps'][-1]
        low=[F(last['lower_CDF'][lookup[t]]) for t in ts];high=[F(last['upper_CDF'][lookup[t]]) for t in ts]
        bound=sum(w*f for w,f in zip(jumps,low));stair_upper=sum(w*f for w,f in zip(jumps,high))
        old=F(last['staircase_expectation_lower']);assert bound>=old
        rows.append(dict(L=row['L'],conditional_recall_lower=str(bound),approximate=float(bound),old_R23_lower=str(old),
            gain=str(bound-old),merged_staircase_upper=str(stair_upper),upper_is_not_target_recall_upper=True))
    result=dict(new_C0_nodes=points,rows=rows,t=list(map(str,ts)),merged_heights=list(map(str,merged)),jumps=list(map(str,jumps)),
        new_samples=0,C0_eta_unchanged=str(eta),new_alpha=str(alpha),independent_review=False,
        source_new_anchor_files={str(p.relative_to(args.input)):sha(p) for p in args.input.glob('*') if p.is_file()},
        source_old_point_sha256=sha(root/'revisions/r20_threshold_family/RESULT.json'),
        source_CDF_sha256=sha(root/'revisions/r23_cdf_dense8/RESULT.json'),wall_seconds=time.monotonic()-start,
        scope='Exact represented C0 ideal Adam2 conditional LOWER, two certified alpha families; inherited R9-R13 arithmetic assumptions')
    save(args.out/'RESULT.json',result);print(json.dumps(dict(lower=[dict(L=r['L'],lower=r['approximate']) for r in rows],wall_seconds=result['wall_seconds']),indent=2))


if __name__=='__main__':main()
