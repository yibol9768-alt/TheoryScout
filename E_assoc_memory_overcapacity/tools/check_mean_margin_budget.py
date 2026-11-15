"""Exact finite-distribution references and arithmetic audit for R31."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import json

from certify_mean_margin_budget import I,S,read,sha,save,restore,root_energy_bounds,threshold_mean,intersection


def references():
    root_cases=0;threshold_cases=0;margin_cases=0;transfer_cases=0
    values=[F(0),F(1),F(4),F(9)];roots=[F(0),F(1),F(2),F(3)]
    weight_vectors=[[1,1,1,1],[1,3,0,0],[2,0,3,1],[0,1,4,2],[3,2,1,0],[0,3,0,1]]
    for weights in weight_vectors:
        probs=[F(w,sum(weights)) for w in weights];mean=sum(p*v for p,v in zip(probs,values));var=sum(p*(v-mean)**2 for p,v in zip(probs,values));expect=sum(p*v for p,v in zip(probs,roots))
        for cutoff in [F(1,4),F(1,2),F(3,4)]:
            if cutoff**2>=mean:continue
            bad=sum(p for p,v in zip(probs,values) if v<cutoff**2)
            r=root_energy_bounds(I(mean),I(var),F(9),cutoff,bad)
            assert r['mean_sqrt_energy'].contains(expect)
            exact_gap=I(mean).sqrt()-expect
            assert r['Jensen_gap'].lo<=exact_gap.lo<=exact_gap.hi<=r['Jensen_gap'].hi
            assert r['moment_only_lower'].lo<=I(expect).hi;root_cases+=1
        # Dependent D,V; T is the separate conditional-geometry factor only.
        for scores in [[-2,3,1,4],[4,-1,2,0],[0,1,2,3]]:
            mu=sum(p*d for p,d in zip(probs,scores))
            for t_weights in [[1,2,3,1],[0,1,0,3]]:
                tv=[F(-2),F(0),F(1,2),F(3)];tp=[F(p,sum(t_weights)) for p in t_weights];tau=sum(p*t for p,t in zip(tp,tv))
                actual=sum(p*q*(d-t*r) for p,d,r in zip(probs,scores,roots) for q,t in zip(tp,tv))
                assert actual==mu-tau*expect
                mu0=F(7,3);t0=F(5,2)
                assert actual-(mu0-t0)==(mu-mu0)-tau*(expect-1)-(tau-t0);margin_cases+=1
    support=[F(-3),F(-1),F(0),F(1,2),F(1),F(2),F(3)]
    for weights in [[1]*7,[1,0,2,0,1,3,1],[0,0,0,1,2,3,4],[3,2,1,0,0,0,0]]:
        p=[F(w,sum(weights)) for w in weights];actual=sum(q*x for q,x in zip(p,support));negative=sum(q for q,x in zip(p,support) if x<=0)
        for grid in [[F(1,2),F(1),F(2),F(3)],[F(1,4),F(3,4),F(3,2),F(5,2),F(3)]]:
            cdf=[sum(q for q,x in zip(p,support) if x<=t) for t in grid]
            for error in [F(0),F(1,10)]:
                lo=[max(F(0),x-error) for x in cdf];hi=[min(F(1),x+error) for x in cdf]
                got,_=threshold_mean(grid,lo,hi,F(4),negative);assert got.contains(actual);threshold_cases+=1
    # Equal margin means AND variances can coexist with different recall.
    a=[(F(1,2),F(-3)),(F(1,2),F(3))];b=[(F(9,10),F(-1)),(F(1,10),F(9))]
    assert sum(p*x for p,x in a)==sum(p*x for p,x in b)==0
    assert sum(p*x*x for p,x in a)==sum(p*x*x for p,x in b)==9
    assert sum(p for p,x in a if x>=0)==F(1,2) and sum(p for p,x in b if x>=0)==F(1,10)
    # Exact single-entry operator perturbations, with rational unit directions.
    blocks=[[[F(1,2),F(-1,4)],[F(3,4),F(1,8)]],[[F(0),F(1)],[F(-1),F(0)]]]
    for matrix in blocks:
        beta=sum(abs(x) for row in matrix for x in row)
        khat=sum(abs((matrix[i][j]+matrix[j][i])/2) for i in range(2) for j in range(2))
        for eta in [F(1,16),F(1,1024)]:
            for i in range(2):
                for j in range(2):
                    changed=[row.copy() for row in matrix];changed[i][j]+=eta
                    nu=eta*(2*beta+eta);kappa=khat+eta
                    for n in [2,3,5]:
                        for w in [[F(1),F(0)],[F(0),F(1)],[F(3,5),F(4,5)],[F(-3,5),F(4,5)]]:
                            def energy(b):
                                bw=[sum(b[i][j]*w[j] for j in range(2)) for i in range(2)]
                                D=n*sum(w[i]*bw[i] for i in range(2));return F(n*n,n-1)*sum(v*v for v in bw)-D*D/(n-1)
                            v0=energy(changed);vh=energy(matrix)
                            assert 0<=v0<=F(n*n,n-1)*(beta+eta)**2
                            assert abs(v0-vh)<=F(n*n,n-1)*(nu+2*kappa*eta);transfer_cases+=1
    return dict(exact_sqrt_energy_and_Jensen_cases=root_cases,finite_threshold_mean_cases=threshold_cases,
                dependent_D_V_mean_budget_cases=margin_cases,exact_energy_transfer_cases=transfer_cases,
                same_mean_and_variance_different_recall_example=True,independent_proof_review=False)


def audit(directory):
    root=Path(__file__).resolve().parents[1];inputs=read(directory/'INPUTS.json');result=read(directory/'RESULT.json')
    for name,digest in inputs['source_hashes'].items():assert sha(root/name)==digest
    source=read(root/'revisions/r28_conditional_moments_v2/INPUTS.json');n=source['n'];m=n-1
    assert inputs['n']==result['n']==n
    eta=F(source['eta']);nu=F(source['N_operator_perturbation']);kappa=F(source['common_K_norm'])
    assert F(inputs['energy_operator_perturbation'])==F(n*n,m)*(nu+2*kappa*eta)
    assert F(inputs['energy_support_upper'])==F(n*n,m)*(F(source['Nhat_operator_norm_upper'])+nu)
    events=read(root/'revisions/r27_recall_upper/BAD_EVENTS.json')
    delta=sum(F(e['event_upper']) for e in events['events'] if e['label'] in ['D_low','D_high','W_low'])
    assert delta==F(inputs['C0_low_energy_probability_upper'])
    assert F(inputs['C0_root_cutoff'])**2+F(inputs['energy_operator_perturbation'])<F(inputs['Bhat_root_cutoff'])**2
    moments=read(root/'revisions/r28_conditional_moments_v2/RESULT.json')['C0_moments']
    expected=root_energy_bounds(restore(moments['mean_V']),restore(moments['variance_V']),F(inputs['energy_support_upper']),F(inputs['C0_root_cutoff']),F(inputs['C0_low_energy_probability_upper']))
    assert {k:v.record() for k,v in expected.items()}==result['energy']
    cdf=read(root/'revisions/r23_cdf_dense8/RESULT.json');details=read(directory/'THRESHOLD_MEANS.json')
    mu=restore(moments['mean_D']);mu0=restore(read(root/'revisions/r29_scalar_center_fine/RESULT.json')['theory_center'])
    for row,source,detail in zip(result['rows'],cdf['rows'],details):
        sweep=source['sweeps'][-1];tau,d=threshold_mean(list(map(F,source['t'])),list(map(F,sweep['lower_CDF'])),list(map(F,sweep['upper_CDF'])),F(inputs['threshold_support_abs_upper']),F(inputs['negative_threshold_probability_upper']))
        assert detail==dict(L=row['L'],**d) and tau.record()==row['threshold_mean']
        e=expected['mean_sqrt_energy'];u=expected['RMS'];gap=expected['Jensen_gap'];t0=F(row['threshold_reference'])
        quantities=dict(actual_mean_margin=mu-tau*e,unit_energy_mean_margin=mu-tau,center_shift=mu-mu0,
            mean_energy_effect=-tau*(e-1),threshold_mean_effect=I(t0)-tau,RMS_freeze_mean_margin=mu-tau*u,
            fluctuation_correction_to_RMS_freeze=tau*gap)
        quantities['total_shift_from_reference']=intersection(quantities['center_shift']+quantities['mean_energy_effect']+quantities['threshold_mean_effect'],quantities['actual_mean_margin']-(mu0-t0))
        for name,value in quantities.items():assert value.record()==row[name],name
    return dict(source_hashes=len(inputs['source_hashes']),energy_transfer_and_low_event_reconstructed=True,energy_records_reconstructed=len(expected),threshold_rows_and_margin_budgets_reconstructed=2,mean_margin_not_recall=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('certificate',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=dict(reference=references(),arithmetic_audit=audit(args.certificate));save(args.out,out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
