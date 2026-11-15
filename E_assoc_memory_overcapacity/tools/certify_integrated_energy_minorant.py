"""Integrate stored positive density pieces over common threshold segments.

Frozen R52 rows remain unchanged. No density/CDF probability integral is
recomputed; only a piecewise-constant certified minorant is integrated.
"""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_right
import argparse
import certify_tail_controlled_energy as previous

base=previous.base
energy=previous.zero.flips.energy
I=base.integer.I


class Integrated(energy.TwoSided):
    def __init__(self,upper,lower):
        super().__init__(upper,lower)
        self.pieces={r['L']:[(F(b['left']),F(b['right']),F(b['density_lower'])) for b in r['bins']]
                     for r in lower['rows']}

    def integral_lower(self,length,start,end):
        if end<=start:return F(0)
        # The stored outside-grid minorant is zero. Only intersecting bins
        # contribute; an endpoint zero does not erase interior contributions.
        if not hasattr(self,'integral_prefix'):self.integral_prefix={}
        if length not in self.integral_prefix:
            pieces=self.pieces[length];starts=[];prefix=[F(0)]
            for j,(a,b,h) in enumerate(pieces):
                assert a<b and h>=0
                if j:assert a==pieces[j-1][1]
                starts.append(a);prefix.append(prefix[-1]+h*(b-a))
            self.integral_prefix[length]=(starts,prefix)
        starts,prefix=self.integral_prefix[length];pieces=self.pieces[length]
        def primitive(x):
            j=bisect_right(starts,x)-1
            if j<0:return F(0)
            a,b,h=pieces[j]
            return prefix[j]+h*max(F(0),min(x,b)-a)
        value=primitive(end)-primitive(start)
        assert 0<=value<=1
        return value

    def signed_minorant(self,length,box,meanv):
        low,high=super().signed_minorant(length,box,meanv)
        dl,dh,vl,vh=box;ml,mh=meanv
        if vh<=ml:
            start=base.hi(I(dh)/I(ml).sqrt())
            end=base.lo(I(dl)/I(vh).sqrt())
            low=max(low,self.integral_lower(length,start,end))
        if vl>=mh:
            start=base.hi(I(dh)/I(vl).sqrt())
            end=base.lo(I(dl)/I(mh).sqrt())
            high=min(high,-self.integral_lower(length,start,end))
        assert low<=high
        return low,high


def sources(root):
    old=base.read(root/'revisions/r52_augmented_zero_certificates.json')
    receipt=base.read(root/'revisions/R52_HISTORY.json');manifest=base.read(root/receipt['path']/'manifest.json')
    assert base.sha(root/'revisions/r52_augmented_zero_certificates.json')==manifest['files']['revisions/r52_augmented_zero_certificates.json']
    for n,h in old['source_hashes'].items():assert base.sha(root/n)==h,n
    names=list(old['source_hashes'])+['revisions/r52_augmented_zero_certificates.json','tools/certify_integrated_energy_minorant.py']
    return {n:base.sha(root/n) for n in names}


def model(root):
    d,rows,old=previous.model(root)
    upper=base.read(root/'revisions/r44_sphere_density.json');lower=base.read(root/'revisions/r45_sphere_density_lower_v2.json')
    minorant=Integrated(upper,lower)
    ds=list(map(F,d['D_grid']));vs=list(map(F,d['V_grid']));meanv=tuple(map(F,d['mean_V_interval']))
    boxes=[(a,b,c,e) for a,b in zip(ds,ds[1:]) for c,e in zip(vs,vs[1:])]
    negative=F(d['negative_threshold_upper']);new={};counts=[]
    for r in base.read(root/'revisions/r23_cdf_dense8/RESULT.json')['rows']:
        length=r['L'];vals=[minorant.kernel(r,box,meanv,negative) for box in boxes]+[old[length][-1]]
        assert all(a[2:]==b[2:] and a[0]<=b[0]<=b[1]<=a[1] for a,b in zip(old[length],vals))
        new[length]=vals
        counts.append(dict(L=length,lower_coefficients_strictly_tightened=sum(b[0]>a[0] for a,b in zip(old[length],vals)),
            upper_coefficients_strictly_tightened=sum(b[1]<a[1] for a,b in zip(old[length],vals))))
    return d,rows,old,new,counts


def references():
    obj=Integrated.__new__(Integrated);obj.pieces={1:[(F(0),F(1),F(0)),(F(1),F(2),F(1)),(F(2),F(3),F(0))]}
    checks=0
    for start in [F(0),F(1,2),F(1),F(3,2),F(2),F(3)]:
        for end in [F(0),F(1),F(3,2),F(2),F(5,2),F(3)]:
            exact=max(F(0),min(end,F(2))-max(start,F(1)))
            assert obj.integral_lower(1,start,end)==exact;checks+=1
    assert obj.integral_lower(1,F(0),F(3))==1 # minimum over the entire segment is zero
    containment=0
    for dl,dh in [(F(2),F(5,2)),(F(3),F(4))]:
        for vl,vh,ml,mh in [(F(1),F(4),F(9),F(16)),(F(9),F(16),F(1),F(4))]:
            positive=vh<=ml
            start=base.hi(I(dh)/I(ml if positive else vl).sqrt())
            end=base.lo(I(dl)/I(vh if positive else mh).sqrt())
            for score in [dl,(dl+dh)/2,dh]:
                for v in [vl,vh]:
                    for mean in [ml,mh]:
                        if end>start:
                            lower=I(score)/I(mean if positive else v).sqrt()
                            upper=I(score)/I(v if positive else mean).sqrt()
                            assert base.lo(lower)<=start and end<=base.hi(upper)
                        containment+=1
    return dict(piecewise_integral_exact_cases=checks,common_segment_containment_cases=containment,
                zero_endpoint_positive_interior_counterexample=True)


def propose(root,out):
    import numpy as np
    from scipy.optimize import linprog
    d,rows,old,kernels,counts=model(root)
    costs=[]
    for vals in kernels.values():costs.append([previous.zero.flips.envelopes(v[0],v[1])['total'][0] for v in vals])
    allowed=[i for i in range(d['good_box_cells']+1) if all(c[i]==0 for c in costs)]
    scales=[max(F(1),abs(r['rhs']),max(abs(x) for x in r['coefficients'])) for r in rows]
    matrix=np.array([[float(r['coefficients'][i]/s) for i in allowed]+[1.] for r,s in zip(rows,scales)])
    rhs=np.array([float(r['rhs']/s) for r,s in zip(rows,scales)])
    fit=linprog(np.array([0.]*len(allowed)+[-1.]),A_ub=matrix,b_ub=rhs,A_eq=np.array([[1.]*len(allowed)+[0.]]),b_eq=[1.],bounds=[(0,None)]*len(allowed)+[(None,None)],method='highs')
    assert fit.success,fit.message
    mass=[[i,F.from_float(float(p))] for i,p in zip(allowed,fit.x[:-1]) if p>0]
    j=max(range(len(mass)),key=lambda j:mass[j][1]);mass[j][1]+=1-sum(p for i,p in mass)
    mass=[[i,str(p)] for i,p in mass];check=previous.audit(d,rows,kernels,mass,84)
    base.save(out,dict(mass=mass,float_scaled_phase_I_margin=float(fit.x[-1]),allowed_common_zero_cells=len(allowed),exact_minimum_slack=check['minimum_raw_slack']))
    print('PROPOSED',check['positive_cells'],'cells; all84 rows and six integrated lower objectives zero',flush=True)


def certify(root,out):
    hashes=sources(root);d,rows,old,kernels,counts=model(root)
    p=base.read(root/'revisions/r53_integrated_mass_proposal.json');check=previous.audit(d,rows,kernels,p['mass'],84)
    assert check['minimum_raw_slack']==p['exact_minimum_slack']
    prior_mass=base.read(root/'revisions/r52_mass_proposals.json')['augmented84_mass'];old_costs=[]
    for length,values in kernels.items():
        env=[previous.zero.flips.envelopes(v[0],v[1]) for v in values]
        for component in ['positive','negative','total']:
            cost=sum(F(w)*env[i][component][0] for i,w in prior_mass)
            old_costs.append(dict(L=length,component=component,new_lower_cost_of_R52_witness=str(cost)))
    assert any(F(r['new_lower_cost_of_R52_witness'])>0 for r in old_costs)
    hashes['revisions/r53_integrated_mass_proposal.json']=base.sha(root/'revisions/r53_integrated_mass_proposal.json')
    base.save(out,dict(source_hashes=hashes,constraints=84,cells=d['good_box_cells']+1,counts=counts,
        original_recall_coefficients_unchanged=True,all_signed_kernel_intervals_nested=True,
        reference=references(),R52_witness_new_costs=old_costs,new_exact_zero_witness=check,
        six_integrated_lower_LP_optima_exactly_zero=True,new_density_or_CDF_probability_integrals=0,
        actual_C0_zero_flips_or_complete_moment_nonidentifiability_claimed=False,
        actual_recall_and_energy_contrast_probability_enclosures_unchanged=True,
        prior_density_CDF_spectral_moment_C0_certificates_inherited=True,independent_proof_review=False))
    print('EXACT integrated kernels nested;',counts,'; six new LP lower optima still zero',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--propose',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.propose:propose(root,a.out)
    else:certify(root,a.out)

if __name__=='__main__':main()
