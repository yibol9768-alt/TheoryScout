"""Certified sqrt-energy Jensen gap and actual one-query mean-margin budget.

No D/V independence or Gaussian score approximation. This is not a recall
probability decomposition. Inherits R23/R27/R28/R29 numerical certificates.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse
import hashlib
import json
import math
import time

from certify_weighted_chisquare import I,SCALE as S


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def restore(r):
    assert r['denominator_power2']==160
    return I.raw(int(r['lower_integer']),int(r['upper_integer']))
def intersection(a,b):return I.raw(max(a.lo,b.lo),min(a.hi,b.hi))


def root_energy_bounds(mean,variance,cap,cutoff,bad_upper):
    u=mean.sqrt();h=I(cap);l=I(cutoff);delta=I(bad_upper)
    assert mean.lo>l.square().hi and h.lo>=mean.hi and variance.lo>0
    coefficient=1/(2*u*(l+u).square())
    residual=u/2-coefficient*mean.square();assert residual.lo>0
    gap_low=variance/(2*u*(h.sqrt()+u).square())
    gap_high=coefficient*variance+delta*residual
    gap=I.raw(gap_low.lo,gap_high.hi);assert gap.lo>0
    root=I.raw(u.lo-gap.hi,u.hi-gap.lo)
    # Sharp moment-only lower bound, attained by a distribution on {0,c}.
    moment_only=mean*mean.sqrt()/(mean.square()+variance).sqrt()
    root=intersection(root,I.raw(moment_only.lo,u.hi))
    return dict(RMS=u,mean_sqrt_energy=root,Jensen_gap=gap,
                quadratic_coefficient=coefficient,low_event_residual=residual,
                moment_only_lower=moment_only)


def threshold_mean(t,lower,upper,cap,negative_upper):
    assert len(t)==len(lower)==len(upper) and 0<t[0]<t[-1]<=cap
    assert all(a<b for a,b in zip(t,t[1:]))
    lo=[];previous=F(0)
    for x in lower:previous=max(previous,x);lo.append(previous)
    hi=list(upper)
    for j in range(len(hi)-2,-1,-1):hi[j]=min(hi[j],hi[j+1])
    assert all(0<=a<=b<=1 for a,b in zip(lo,hi))
    lb=-cap*negative_upper;ub=F(0);prev=F(0);prevlo=F(0);cells=[]
    for x,a,b in zip(t,lo,hi):
        low=(x-prev)*(1-b);high=(x-prev)*(1-prevlo)
        lb+=low;ub+=high;cells.append(dict(left=str(prev),right=str(x),survival_integral_lower=str(low),survival_integral_upper=str(high)))
        prev=x;prevlo=a
    right=(cap-t[-1])*(1-lo[-1]);ub+=right
    return I.raw(I(lb).lo,I(ub).hi),dict(cells=cells,right_tail_upper=str(right),negative_part_lower=str(-cap*negative_upper),
         lower_mean=str(lb),upper_mean=str(ub),support_abs_upper=str(cap),negative_probability_upper=str(negative_upper))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic();protected=[]
    groups=[('R28',['revisions/r28_conditional_moments_v2/RESULT.json','revisions/r28_conditional_moments_v2/INPUTS.json','tools/certify_weighted_chisquare.py']),
            ('R27',['revisions/r27_recall_upper/BAD_EVENTS.json']),('R23',['revisions/r23_cdf_dense8/RESULT.json']),
            ('R29',['revisions/r29_scalar_center_fine/RESULT.json'])]
    for revision,names in groups:
        receipt=read(root/f'revisions/{revision}_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
        for name in names:assert sha(root/name)==manifest['files'][name];protected.append(name)
    result=read(root/'revisions/r28_conditional_moments_v2/RESULT.json');inputs=read(root/'revisions/r28_conditional_moments_v2/INPUTS.json')
    n=result['n'];m=n-1;assert inputs['n']==n
    eta=F(inputs['eta']);nu=F(inputs['N_operator_perturbation']);kappa=F(inputs['common_K_norm']);rho=F(inputs['Nhat_operator_norm_upper'])+nu
    energy_perturbation=F(n*n,m)*(nu+2*kappa*eta);energy_cap=F(n*n,m)*rho
    bad=read(root/'revisions/r27_recall_upper/BAD_EVENTS.json');assert bad['n']==n
    anchor=F(bad['center']);radius=F(bad['radius']);lhat=F(bad['l0']);l=F(874999,1000000)
    specs={e['label']:e for e in bad['events']}
    assert F(specs['D_low']['threshold'])==anchor-radius and specs['D_low']['sign']==-1
    assert F(specs['D_high']['threshold'])==anchor+radius and specs['D_high']['sign']==1
    assert F(specs['W_low']['threshold'])==lhat*lhat+(abs(anchor)+radius)**2/m and specs['W_low']['sign']==-1
    assert energy_perturbation<=F(1,1000000) and l*l+energy_perturbation<lhat*lhat
    delta=sum(F(specs[label]['event_upper']) for label in ['D_low','D_high','W_low']);assert 0<delta<1
    moments=result['C0_moments'];mean=restore(moments['mean_V']);variance=restore(moments['variance_V'])
    energy=root_energy_bounds(mean,variance,energy_cap,l,delta)
    negative=F(1,1<<100);Tcap=math.isqrt(m)+1;assert Tcap*Tcap>=m
    cdf=read(root/'revisions/r23_cdf_dense8/RESULT.json');mu=restore(moments['mean_D'])
    center=read(root/'revisions/r29_scalar_center_fine/RESULT.json');mu0=restore(center['theory_center']);center_shift=restore(center['C0_minus_theory_center'])
    assert (mu-mu0).record()==center_shift.record()
    rows=[];threshold_details=[]
    for row in cdf['rows']:
        length=row['L'];assert length in [1,5]
        # Sphere normalization preserves normal-coordinate signs exactly.
        assert sum(math.comb(m,k) for k in range(length))*(1<<100)<=(1<<m)
        sweep=row['sweeps'][-1]
        tau,details=threshold_mean(list(map(F,row['t'])),list(map(F,sweep['lower_CDF'])),list(map(F,sweep['upper_CDF'])),F(Tcap),negative)
        assert tau.lo>0;threshold_details.append(dict(L=length,**details))
        root=energy['mean_sqrt_energy'];RMS=energy['RMS'];gap=energy['Jensen_gap']
        t0=F(4) if length==1 else F(7,2)
        mean_margin=mu-tau*root;unit_energy_margin=mu-tau
        energy_effect=-tau*(root-1);threshold_effect=I(t0)-tau
        total_shift=intersection(center_shift+energy_effect+threshold_effect,mean_margin-(mu0-t0))
        rows.append(dict(L=length,threshold_reference=str(t0),threshold_mean=tau.record(),actual_mean_margin=mean_margin.record(),
            unit_energy_mean_margin=unit_energy_margin.record(),center_shift=center_shift.record(),mean_energy_effect=energy_effect.record(),
            threshold_mean_effect=threshold_effect.record(),total_shift_from_reference=total_shift.record(),
            RMS_freeze_mean_margin=(mu-tau*RMS).record(),fluctuation_correction_to_RMS_freeze=(tau*gap).record(),
            all_margin_quantities_in_score_units_not_pp=True))
    save(args.out/'INPUTS.json',dict(n=n,energy_operator_perturbation=str(energy_perturbation),energy_support_upper=str(energy_cap),
        Bhat_root_cutoff=str(lhat),C0_root_cutoff=str(l),C0_low_energy_probability_upper=str(delta),
        threshold_support_abs_upper=Tcap,negative_threshold_probability_upper=str(negative),source_hashes={name:sha(root/name) for name in protected}))
    save(args.out/'THRESHOLD_MEANS.json',threshold_details)
    output=dict(n=n,energy={k:v.record() for k,v in energy.items()},rows=rows,new_samples=0,new_matrices=0,new_probability_integrals=0,
        D_V_independence_assumed=False,conditional_threshold_geometry_used=True,independent_proof_review=False,
        numerical_assumptions_inherited=True,mean_margin_not_recall_probability=True,scope='One-query conditional mean margins at fixed represented C0, ideal Adam2; not population bias or recall attribution',wall_seconds=time.monotonic()-start)
    save(args.out/'RESULT.json',output)
    print(json.dumps(dict(energy={k:output['energy'][k]['approximate'] for k in ['RMS','mean_sqrt_energy','Jensen_gap']},
        rows=[{k:(v['approximate'] if isinstance(v,dict) and 'approximate' in v else v) for k,v in row.items()} for row in rows],wall_seconds=output['wall_seconds']),indent=2))


if __name__=='__main__':main()
