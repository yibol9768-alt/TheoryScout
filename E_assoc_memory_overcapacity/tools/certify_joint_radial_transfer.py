"""Joint radial-mixture constraints; floating LP proposes exact rational duals.

SciPy is used only to select multipliers. Every reported bound is reconstructed
with rational arithmetic, including box correction for the full dual residual.
"""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time

import certify_sphere_threshold_cdf as sphere
from refine_sphere_cdf import RadialCDF


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, value):
    with p.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def mixture_constraints(ts, radial, node):
    """Two globally valid inequalities A F <= b for positive mixture x."""
    x = F(node['x']); values = [radial(x/t) for t in ts]
    low = [v[0] for v in values]; high = [v[1] for v in values]
    a = [low[j]-(low[j+1] if j+1 < len(ts) else 0) for j in range(len(ts))]
    b = [1-high[0]]+[high[j-1]-high[j] for j in range(1, len(ts))]
    assert all(v >= 0 for v in a+b)
    return [(a, F(node['upper'])), ([-v for v in b], high[-1]-F(node['lower']))]


def objective(ts, kind, lower, upper):
    if kind == 'gap':
        constant,a=objective(ts,'upper',lower,upper)
        _,b=objective(ts,'lower',lower,upper)
        return constant,[x-y for x,y in zip(a,b)]
    weights = [F(0)]*len(ts); index = {t:j for j,t in enumerate(ts)}
    if kind == 'lower':
        for t, jump in zip(lower['t'], lower['jumps']): weights[index[F(t)]] += F(jump)
        return F(0), weights
    previous = F(1)
    for t, value in sorted((F(t),F(v)) for t,v in upper['all_point_uppers'].items()):
        current = min(previous,value); weights[index[t]] += previous-current; previous=current
    return previous, weights


def rational_dual(c, low, high, constraints, multipliers):
    """Any nonnegative multipliers work, even a poorly solved floating LP."""
    residual = c.copy(); value = F(0)
    for index, weight in multipliers:
        weight=F(weight); assert weight>=0 and 0<=index<len(constraints)
        a,b=constraints[index];value-=weight*b
        for j,v in enumerate(a): residual[j]+=weight*v
    correction=sum(v*(lo if v>=0 else hi) for v,lo,hi in zip(residual,low,high))
    return value+correction, dict(unconstrained_dual=str(value),box_residual_correction=str(correction),
                                  maximum_absolute_residual=str(max(map(abs,residual))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    root=Path(__file__).resolve().parents[1]
    groups={'R21':['revisions/r21_sphere_cdf/RESULT.json','revisions/r21_sphere_cdf/RADII.json',
                   'tools/certify_sphere_threshold_cdf.py','tools/certify_weighted_chisquare.py'],
            'R23':['revisions/r23_cdf_dense8/RESULT.json','tools/refine_sphere_cdf.py'],
            'R26':['revisions/r26_grid_nodes/RESULT.json'],
            'R36':['revisions/r36_right_tail/RESULT.json']}
    hashes={}
    for rev,names in groups.items():
        receipt=read(root/f'revisions/{rev}_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
        for name in names: assert sha(root/name)==manifest['files'][name];hashes[name]=sha(root/name)
    cdf=read(root/'revisions/r23_cdf_dense8/RESULT.json');lower=read(root/'revisions/r26_grid_nodes/RESULT.json')
    upper=read(root/'revisions/r36_right_tail/RESULT.json');m=read(root/'revisions/r21_sphere_cdf/RESULT.json')['m']
    radial=RadialCDF(read(root/'revisions/r21_sphere_cdf/RADII.json')['records'])
    # This deterministic candidate design does not use threshold or score banks.
    steps=[-40,-20,-10,-5,-2,2,5,10,20,40]
    save(args.out/'DESIGN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),steps_per_thousand=steps,
         source_hashes=hashes,no_new_radial_or_score_integrals=True,floating_solver_is_multiplier_proposal_only=True))
    import numpy as np
    from scipy.optimize import linprog
    pi=sphere.integer.pi_interval();rows=[];gaussian=[]
    for source in cdf['rows']:
        length=source['L'];ts=list(map(F,source['t']));last=source['sweeps'][-1]
        low=list(map(F,last['lower_CDF']));high=list(map(F,last['upper_CDF']));constraints=[];nodes=[]
        for x in sorted({t*(1+F(step,1000)) for t in ts for step in steps}):
            h=sphere.gaussian_order_cdf(m,length,x,pi)
            node=dict(x=str(x),lower=str(F(h.lo,sphere.S)),upper=str(F(h.hi,sphere.S)))
            nodes.append(node);constraints.extend(mixture_constraints(ts,radial,node))
        for j in range(len(ts)-1):
            a=[F(0)]*len(ts);a[j]=F(1);a[j+1]=F(-1);constraints.append((a,F(0)))
        array=np.array([[float(v) for v in a] for a,b in constraints]);rhs=np.array([float(b) for a,b in constraints])
        proofs={}
        for kind,base,sign in [('lower','lower',1),('upper','upper',-1),
                                ('gap_lower','gap',1),('gap_upper','gap',-1)]:
            constant,c=objective(ts,base,lower,upper)
            target=[sign*v for v in c]
            fit=linprog(np.array(list(map(float,target))),A_ub=array,b_ub=rhs,
                        bounds=list(zip(map(float,low),map(float,high))),method='highs')
            assert fit.success,fit.message
            multipliers=[(j,str(F.from_float(float(max(0,-weight))))) for j,weight in enumerate(fit.ineqlin.marginals) if weight<0]
            bound,parts=rational_dual(target,low,high,constraints,multipliers)
            final=constant+sign*bound
            proofs[kind]=dict(constant=str(constant),multipliers=multipliers,parts=parts,
                 certified_staircase_expectation=str(final),float_optimum_diagnostic=constant.__float__()+sign*fit.fun)
        previous=next(r for r in upper['rows'] if r['L']==length)
        lo=max(F(previous['conditional_recall_lower']),F(proofs['lower']['certified_staircase_expectation']))
        hi=min(F(previous['conditional_recall_upper']),F(proofs['upper']['certified_staircase_expectation']))
        assert 0<=lo<=hi<=1
        gaplo=max(F(0),F(proofs['gap_lower']['certified_staircase_expectation']))
        gaphi=min(hi-lo,F(proofs['gap_upper']['certified_staircase_expectation']))
        assert 0<=gaplo<=gaphi<=hi-lo
        rows.append(dict(L=length,conditional_recall_lower=str(lo),conditional_recall_upper=str(hi),
             approximate=[float(lo),float(hi)],previous_interval=[previous['conditional_recall_lower'],previous['conditional_recall_upper']],
             constraints=len(constraints),duals=proofs,envelope_gap_expectation=[str(gaplo),str(gaphi)],
             combined_threshold_propagation_slack=[str(hi-lo-gaphi),str(hi-lo-gaplo)]))
        gaussian.append(dict(L=length,nodes=nodes))
        print('L',length,'bounds',float(lo),float(hi),'dual nnz',[len(v['multipliers']) for v in proofs.values()],flush=True)
    save(args.out/'GAUSSIAN.json',dict(m=m,rows=gaussian,arithmetic='outward 160-bit Gaussian order CDF'))
    save(args.out/'RESULT.json',dict(rows=rows,source_hashes=hashes,m=m,gaussian_file_sha256=sha(args.out/'GAUSSIAN.json'),
         new_gaussian_order_CDF_evaluations=sum(len(r['nodes']) for r in gaussian),new_matrices=0,new_samples=0,
         new_score_or_radial_integrals=0,independent_proof_review=False,exact_dual_box_correction=True,
         no_claim_of_certified_global_LP_optimality=True,wall_seconds=time.monotonic()-start,
         scope='Fixed represented C0 ideal Adam2 one-query conditional recall; stronger joint threshold propagation only; prior spectral, tail, radial and arithmetic assumptions inherited'))


if __name__=='__main__':main()
