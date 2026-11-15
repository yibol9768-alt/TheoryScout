"""Reuse R13's certified matrix for a positive-threshold lower-bound family.

Standard library only. Fixed a,alpha make the matrix independent of t;
b(t)=t/(2*alpha) is legal for t>0. Only the constant changes. This does
not certify the archived threshold sampler or a complete recall interval.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import time

import certify_weighted_chisquare as integer
from audit_threshold_cdf_transfer import npy_thresholds, envelopes, references


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def algebra_checks():
    checks=0
    for m in [1,3,17]:
        for a in [F(-1,2),F(0),F(3,2)]:
            for alpha in [F(1,3),F(2)]:
                for d in [F(-2),F(0),F(5,2)]:
                    for root_v in [F(0),F(1,2),F(2)]:
                        for t in [F(1,4),F(1),F(3)]:
                            b=t/(2*alpha);v=root_v**2;w=v+d*d/m
                            lower=(1+2*alpha*a/m)*d-alpha*w-alpha*a*a/m-t*t/(4*alpha)
                            assert d-t*root_v-lower==alpha*((root_v-b)**2+(d-a)**2/m)
                            assert 4*alpha*m*lower==(4*alpha*m+8*alpha**2*a)*d-4*alpha**2*m*w-4*alpha**2*a*a-m*t*t
                            checks+=1
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];start=time.monotonic()
    receipt=read(root/'revisions/R13_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
    names=['revisions/r11_anchor_certificate_v2/SCOPE.json',
           'revisions/r11_anchor_certificate_v2/ERRORS.json',
           'revisions/r11_anchor_certificate_v2/actual_spectrum.json',
           'revisions/r13_scatter_checks/CHECKS.json',
           'revisions/r13_scatter_checks/RESULT.json']
    for name in names:assert sha(root/name)==manifest['files'][name],name
    source=root/'revisions/r8_random_thresholds';finished=read(source/'FINISHED.json')
    for name in ['thresholds.npy','nodes.jsonl']:
        assert sha(source/name)==finished['files'][name];names.append(str((source/name).relative_to(root)))
    # Bind the statistical radius to R19's independently checked rational record.
    r19=read(root/'revisions/r19_threshold_cdf/RESULTS.json');epsilon=F(r19['epsilon'])
    radius=read(root/'revisions/r19_threshold_cdf/RADIUS.json')
    count,banks=npy_thresholds(source/'thresholds.npy');assert count==r19['sample_count']==131072
    assert epsilon==F(radius['epsilon'])==F(4401,1000000)
    x=2*count*epsilon*epsilon;term=total=F(1)
    for k in range(1,65):term=term*x/k;total+=term
    assert total>=160 and total==F(radius['exp_taylor_lower'])
    scope=read(root/names[0]);errors=read(root/names[1]);spectrum=read(root/names[2]);checks=read(root/names[3])
    n=scope['n'];m=n-1;q=len(spectrum['eigenvalues'])
    a=F(scope['center_anchor']);alpha=F(scope['alpha']);gamma=4*alpha*m
    assert gamma==F(scope['positive_scaling']) and 0<q<n
    eta=F(checks['eta_upper']);spectral_error=F(errors['total'])
    pi=integer.pi_interval();gauss=integer.gauss8();I=integer.I
    records=[json.loads(line) for line in (source/'nodes.jsonl').read_text().splitlines()]
    selected=[r for r in records if r['index'] in [0,16,32,48,64,80,96,112,128]]
    assert len(selected)==18
    design=dict(n=n,q=q,a=str(a),alpha=str(alpha),gamma=str(gamma),eta=str(eta),
                common_spectral_error=str(spectral_error),node_indices=[0,16,32,48,64,80,96,112,128],
                sample_count=count,epsilon=str(epsilon),new_samples=0,
                source_hashes={name:sha(root/name) for name in names},
                scope='Exact represented C0 ideal Adam2; certified fixed-positive-threshold LOWER nodes only',
                caveat='R19 random-threshold transfer assumes ideal IID samples; archived sampler input errors not certified')
    save(args.out/'DESIGN.json',design)
    certified=[]
    for node in selected:
        t=F.from_float(node['t']);assert t>0
        constant=-4*alpha**2*a*a-m*t*t
        gap=I(n)*I(eta)*(1+I(t)/I(m).sqrt());gap_upper=F(gap.hi,integer.SCALE)
        shifted=constant-gamma*gap_upper
        if t==F(scope['threshold']):
            assert constant==F(scope['exact_constant'])
            assert gap_upper==F(checks['score_gap_upper'])
        coefficients=[F(v)+shifted/n-spectral_error for v in spectrum['eigenvalues']]+[shifted/n]
        record,enclosure=integer.certify(coefficients,[1]*q+[n-q],pi,gauss)
        label=f"L{node['L']}_node{node['index']:03d}"
        save(args.out/(label+'_integral.json'),record)
        lower=max(F(0),F(enclosure.lo,integer.SCALE))
        result=dict(L=node['L'],index=node['index'],t=str(t),b=str(t/(2*alpha)),
                    exact_constant=str(constant),score_gap_upper=str(gap_upper),shifted_constant=str(shifted),
                    target_probability_lower=str(lower),lower_comparison_event_probability=enclosure.record(),
                    integral_file=label+'_integral.json',wall_seconds=time.monotonic()-start)
        if t==F(scope['threshold']):
            assert lower==F(read(root/'revisions/r13_scatter_checks/RESULT.json')['ideal_fixed_threshold_success_lower_bound'])
        certified.append(result);save(args.out/(label+'.json'),result)
        print(label,'t',float(t),'target LOWER',float(lower),flush=True)
    rows=[]
    for column,length in enumerate([1,5]):
        levels=[]
        for groups in [1,2,4,8]:
            indices={0,*range(128//groups,129,128//groups)}
            points=[(F(c['t']),F(c['target_probability_lower']),F(1)) for c in certified if c['L']==length and c['index'] in indices]
            mean,_,osc,_=envelopes([F.from_float(t) for t in banks[column]],points)
            lower=max(F(0),mean-epsilon*osc)
            levels.append(dict(groups=groups,certified_nodes=len(points),empirical_mean_lower=str(mean),
                               oscillation=str(osc),ideal_iid_model_lower=str(lower),approximate=float(lower)))
        rows.append(dict(L=length,levels=levels))
    result=dict(certified_nodes=certified,random_threshold_model_lower=rows,
                exact_algebra_checks=algebra_checks(),exact_cdf_transfer_checks=references(),
                source_hashes=design['source_hashes'],new_samples=0,
                fixed_node_lower_numerical_inclusion=True,full_random_threshold_numerical_certification=False,
                independent_review=False,wall_seconds=time.monotonic()-start,
                limitations=['Floating threshold generation/normalization not enclosed',
                             'No C0 random-threshold upper certificate',
                             'Numerics inherit R9-R13 proof, compiler and IEEE assumptions',
                             'Not unrounded Gaussian data or single-dataset recall guarantee'])
    save(args.out/'RESULT.json',result)
    print(json.dumps(dict(nodes=len(certified),algebra_checks=result['exact_algebra_checks'],
                         model_lower=[dict(L=r['L'],lower=r['levels'][-1]['approximate']) for r in rows],
                         wall_seconds=result['wall_seconds']),indent=2),flush=True)


if __name__=='__main__':main()
