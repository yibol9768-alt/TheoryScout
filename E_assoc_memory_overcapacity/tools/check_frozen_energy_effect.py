"""Finite-reference and coefficient/jump audit for R32 counterfactual contrast."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import json

from certify_frozen_energy_effect import profile,read,sha,save,restore,I,S


def references():
    points=0;contrasts=0;quadratics=0
    grid=[F(k,4) for k in range(1,25)];support=[F(-2),F(0),F(1),F(2),F(3)]
    for weights in [[1,1,2,3,1],[0,0,1,2,3],[1,0,1,0,1]]:
        p=[F(w,sum(weights)) for w in weights]
        cdf=lambda x:sum(q for q,t in zip(p,support) if t<=x)
        for scale in [F(3,2),F(2)]:
            for error in [F(0),F(1,100)]:
                row=dict(L=1,t=list(map(str,grid)),sweeps=[dict(lower_CDF=[str(max(F(0),cdf(t)-error)) for t in grid],upper_CDF=[str(min(F(1),cdf(t)+error)) for t in grid])])
                d=profile(row,scale)
                def minorant(x):
                    value=F(0);previous=F(0)
                    for layer in d['layers']:
                        h=F(layer['height'])
                        if F(layer['left'])<=x<=F(layer['right']):value+=h-previous
                        previous=h
                    return value
                probes=set(grid+[scale*x for x in grid])
                for layer in d['layers']:
                    a=F(layer['left']);b=F(layer['right']);probes.update([a,b,(a+b)/2,a-F(1,1000),b+F(1,1000)])
                for x in probes:
                    assert x>=0 and 0<=minorant(x)<=cdf(x)-cdf(x/scale);points+=1
                scores=[F(-3),F(0),F(1),F(5,4),F(2),F(3),F(4),F(5)];prob=[F(1,len(scores))]*len(scores)
                negative=cdf(F(0));lower=F(0);previous=F(0)
                for layer in d['layers']:
                    a=F(layer['left']);b=F(layer['right']);h=F(layer['height'])
                    Sa=sum(q for q,x in zip(prob,scores) if x>=a);Sb=sum(q for q,x in zip(prob,scores) if x>=b)
                    mass=max(F(0),Sa-Sb);lower+=(h-previous)*mass;previous=h
                for true_scale in [scale,scale+F(1,3)]:
                    contrast=sum(q*(cdf(x)-cdf(x/true_scale)) for q,x in zip(prob,scores))
                    assert contrast>=lower-negative;contrasts+=1
    for eigen in [[F(-1,2),F(3,4)],[F(-1),F(-1,4),F(1,2)]]:
        q=len(eigen)
        for n in [q+1,q+3]:
            for threshold in [F(-1),F(0),F(2),F(4)]:
                eps=F(1,16);actual=[x+(-1)**j*eps/2 for j,x in enumerate(eigen)]
                for g in [[F(1)]*n,[F(j+1,3) for j in range(n)],[F((-1)**j,2) for j in range(n)]]:
                    total=sum(x*x for x in g);D=F(n)*sum(x*y*y for x,y in zip(actual,g))/total
                    forms=[]
                    for spectrum in [[x-eps for x in eigen],actual,[x+eps for x in eigen]]:
                        forms.append(sum((spectrum[j]-threshold/n)*g[j]**2 for j in range(q))-threshold/n*sum(g[j]**2 for j in range(q,n)))
                    assert forms[0]<=forms[1]<=forms[2]
                    assert (D>=threshold)==(forms[1]>=0)
                    assert not(forms[0]>=0 and D<threshold) and not(D>=threshold and forms[2]<0);quadratics+=1
    return dict(pointwise_minorant_checks_with_atoms=points,finite_counterfactual_contrast_checks=contrasts,
                exact_survival_quadratic_comparison_cases=quadratics,independent_proof_review=False)


def audit(directory):
    root=Path(__file__).resolve().parents[1];design=read(directory/'DESIGN.json');result=read(directory/'RESULT.json')
    for name,digest in design['source_hashes'].items():assert sha(root/name)==digest
    cdf=read(root/'revisions/r23_cdf_dense8/RESULT.json');scale=F(design['scale_lower'])
    expected=[profile(row,scale) for row in cdf['rows']];assert expected==design['profiles']
    inp=read(root/'revisions/r28_conditional_moments_v2/INPUTS.json');n=inp['n'];q=inp['q']
    error=F(read(root/'revisions/r27_bad_matrices/D/ERRORS.json')['total'])+F(inp['eta'])
    eigen=list(map(F,read(root/'revisions/r27_bad_matrices/D/actual_spectrum.json')['eigenvalues']))
    assert eigen==sorted(eigen) and len(eigen)==q
    nodes={};count=0
    for node in result['nodes']:
        t=F(node['threshold']);side=node['side'];sign=-1 if side=='lower' else 1
        record=read(directory/node['integral_file']);coeff=[x-t/n+sign*error for x in eigen]+[-t/n]
        assert list(map(F,record['coefficients']))==coeff and record['counts']==[1]*q+[n-q]
        arithmetic=restore(record['arithmetic_only']);width=restore(record['finite_quadrature_error']).hi+restore(record['infinite_tail_error']).hi
        probability=I.raw(arithmetic.lo-width,arithmetic.hi+width)
        assert probability.record()==record['probability']==node['comparison_probability']
        bound=max(F(0),min(F(1),F(probability.lo if side=='lower' else probability.hi,S)))
        assert bound==F(node['C0_survival_bound'])
        assert F(node['spectral_and_C0_error'])==error and F(record['target_width'])==F(1,10**6)
        nodes[(t,side)]=bound;count+=1
    for row,d in zip(result['rows'],expected):
        total=F(0);previous=F(0)
        for layer,record in zip(d['layers'],row['layers']):
            mass=max(F(0),nodes[(F(layer['left']),'lower')]-nodes[(F(layer['right']),'upper')]);h=F(layer['height'])
            assert mass==F(record['C0_interval_probability_lower']) and h-previous==F(record['jump_weight'])
            total+=(h-previous)*mass;previous=h
        assert total==F(row['minorant_expectation_lower'])
        assert total-F(design['negative_threshold_probability_upper'])==F(row['unit_minus_RMS_frozen_success_lower'])>0
    return dict(source_hashes=len(design['source_hashes']),CDF_profiles_reconstructed=2,
                score_integral_coefficient_error_audits=count,counterfactual_jump_sums_reconstructed=2,
                comparison_intervals_not_actual_C0_intervals=True,actual_random_energy_recall_not_bounded=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('certificate',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=dict(reference=references(),arithmetic_audit=audit(args.certificate));save(args.out,out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
