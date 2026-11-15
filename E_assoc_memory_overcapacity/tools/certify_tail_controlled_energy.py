"""Exact zero-cost witnesses after global and tail-controlled moment cuts.

SciPy only proposes one mass vector. Replay uses rational arithmetic only.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse
import certify_energy_tail_moment as tail

base=tail.base
zero=tail.zero


def model(root):
    zero.sources(root)
    t=tail.compute(root);saved=base.read(root/'revisions/r52_tail_second_moment.json')
    for k,v in t.items():assert saved[k]==v,k
    for n,h in saved['source_hashes'].items():assert base.sha(root/n)==h,n
    g=tail.global_cut.compute(root);old=base.read(root/'revisions/r51_variance_cut.json')
    for k,v in g.items():assert old[k]==v,k
    d,rows,k=zero.flips.energy.build(root)
    for cut in [g,t]:rows.append(dict(label=cut['label'],coefficients=list(map(F,cut['coefficients'])),rhs=F(cut['rhs'])))
    assert len(rows)==84
    return d,rows,k


def audit(d,rows,kernels,mass,number):
    weights={}
    for i,p in mass:
        assert isinstance(i,int) and 0<=i<=d['good_box_cells'] and i not in weights
        weights[i]=F(p);assert weights[i]>0
    assert sum(weights.values())==1
    slacks=[]
    for row in rows[:number]:
        lhs=sum(p*row['coefficients'][i] for i,p in weights.items());slack=row['rhs']-lhs
        assert slack>=0,row['label']
        slacks.append(dict(label=row['label'],slack=str(slack)))
    proofs=[]
    for length,values in kernels.items():
        cells=[zero.flips.envelopes(v[0],v[1]) for v in values]
        for component in ['positive','negative','total']:
            cost=[v[component][0] for v in cells]
            assert all(c>=0 for c in cost) and all(cost[i]==0 for i in weights)
            value=sum(cost[i]*p for i,p in weights.items());dual,parts=base.simplex_dual(cost,rows[:number],[])
            assert value==dual==0
            proofs.append(dict(L=length,component=component,exact_primal='0',exact_dual='0'))
    ds=list(map(F,d['D_grid']));vs=list(map(F,d['V_grid']));nv=len(vs)-1
    strips=sorted({(vs[i%nv],vs[i%nv+1]) for i in weights if i<d['good_box_cells']})
    return dict(positive_cells=len(weights),mass_sum='1',outside_mass=str(weights.get(d['good_box_cells'],F(0))),
        occupied_V_strips=[list(map(str,s)) for s in strips],inequalities_checked=number,slacks=slacks,
        minimum_raw_slack=str(min(F(s['slack']) for s in slacks)),six_exact_zero_optima=proofs,
        witness_is_cell_mass_vector_not_actual_C0_law=True,actual_zero_flips_claimed=False)


def sources(root):
    names=list(tail.sources(root))+['tools/certify_tail_controlled_energy.py','revisions/r52_tail_second_moment.json',
        'revisions/r52_mass_proposals.json']
    return {n:base.sha(root/n) for n in names}


def propose(root,out):
    d,rows,kernels=model(root);old=base.read(root/'revisions/r51_energy_zero_witness.json')
    eps=F(1,1000000);mix=[[i,str((1-eps)*F(p))] for i,p in old['mass_by_cell']]+[[d['good_box_cells'],str(eps)]]
    a=audit(d,rows,kernels,mix,83)
    violation=sum(F(p)*rows[83]['coefficients'][i] for i,p in mix)-rows[83]['rhs'];assert violation>0
    import numpy as np
    from scipy.optimize import linprog
    costs=[]
    for vals in kernels.values():costs.append([zero.flips.envelopes(v[0],v[1])['total'][0] for v in vals])
    allowed=[i for i in range(d['good_box_cells']+1) if all(c[i]==0 for c in costs)]
    scales=[max(F(1),abs(r['rhs']),max(abs(x) for x in r['coefficients'])) for r in rows]
    matrix=np.array([[float(r['coefficients'][i]/s) for i in allowed]+[1.] for r,s in zip(rows,scales)])
    rhs=np.array([float(r['rhs']/s) for r,s in zip(rows,scales)])
    eq=np.array([[1.]*len(allowed)+[0.]])
    fit=linprog(np.array([0.]*len(allowed)+[-1.]),A_ub=matrix,b_ub=rhs,A_eq=eq,b_eq=[1.],bounds=[(0,None)]*len(allowed)+[(None,None)],method='highs')
    assert fit.success,fit.message
    proposal=[[i,F.from_float(float(p))] for i,p in zip(allowed,fit.x[:-1]) if p>0]
    index=max(range(len(proposal)),key=lambda j:proposal[j][1]);proposal[index][1]+=1-sum(p for i,p in proposal)
    mass=[[i,str(p)] for i,p in proposal]
    check=audit(d,rows,kernels,mass,84)
    base.save(out,dict(exact_global_cut_mixture=mix,global_cut_mixture_tail_cut_violation=str(violation),
        augmented84_mass=mass,float_proposal_only=dict(scaled_phase_I_margin=float(fit.x[-1]),allowed_all_zero_cells=len(allowed)),
        exact_minimum_slack=check['minimum_raw_slack']))
    print('PROPOSED',a['positive_cells'],'cells for83 rows;',check['positive_cells'],'for84 rows; exact verification passed',flush=True)


def certify(root,out):
    source=sources(root);d,rows,kernels=model(root);p=base.read(root/'revisions/r52_mass_proposals.json')
    old83=audit(d,rows,kernels,p['exact_global_cut_mixture'],83)
    aug84=audit(d,rows,kernels,p['augmented84_mass'],84)
    v=sum(F(w)*rows[83]['coefficients'][i] for i,w in p['exact_global_cut_mixture'])-rows[83]['rhs']
    assert str(v)==p['global_cut_mixture_tail_cut_violation'] and v>0
    assert aug84['minimum_raw_slack']==p['exact_minimum_slack']
    result=dict(source_hashes=source,global83=old83,tail_controlled84=aug84,
        global83_witness_violates_tail_controlled_row_by=str(v),all_six_augmented_lower_optima_exactly_zero=True,
        good_window_energy_second_moment_positive_but_recall_flip_lower_still_zero=True,
        full_moment_nonidentifiability_not_claimed=True,no_new_actual_energy_contrast_or_recall_enclosure=True,
        prior_C0_density_CDF_spectral_moment_inputs_inherited=True,independent_proof_review=False)
    base.save(out,result)
    print('EXACT:83 and84 row relaxations both have six zero lower optima; no actual zero flip claim',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--propose',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.propose:propose(root,a.out)
    else:certify(root,a.out)

if __name__=='__main__':main()
