"""Uniform-CDF propagation for archived R8 thresholds; no new samples.

Node floats remain uncertified diagnostics. A separate field propagates the
R13 certified fixed-node lower bound under an IDEAL IID threshold model;
threshold generation/normalization rounding is explicitly not certified.
"""
from array import array
import argparse
import ast
from bisect import bisect_left,bisect_right
from fractions import Fraction as F
from itertools import product
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
def exact(value):return F.from_float(float(value))


def npy_thresholds(path):
    with path.open('rb') as stream:
        assert stream.read(6)==b'\x93NUMPY';version=tuple(stream.read(2));assert version in [(1,0),(2,0)]
        width=2 if version==(1,0) else 4
        size=int.from_bytes(stream.read(width),'little');header=ast.literal_eval(stream.read(size).decode('latin1'))
        assert header['descr']=='<f8' and not header['fortran_order']
        data=array('d');data.frombytes(stream.read())
        if sys.byteorder!='little':data.byteswap()
        count,columns=header['shape'];assert columns==2 and len(data)==2*count
        assert all(math.isfinite(x) for x in data)
        return count,[list(data[j::2]) for j in range(2)]


def envelopes(samples,points):
    points=sorted(points);xs=[p[0] for p in points]
    assert len(set(xs))==len(xs)
    lower=[p[1] for p in points];upper=[p[2] for p in points]
    for j in range(len(points)-2,-1,-1):lower[j]=max(lower[j],lower[j+1])
    for j in range(1,len(points)):upper[j]=min(upper[j],upper[j-1])
    lc=[0]*(len(points)+1);uc=[0]*(len(points)+1)
    for t in samples:
        lc[bisect_left(xs,t)]+=1;uc[bisect_right(xs,t)]+=1
    lvalues=lower+[F(0)];uvalues=[F(1)]+upper
    meanL=sum(c*v for c,v in zip(lc,lvalues))/len(samples)
    meanU=sum(c*v for c,v in zip(uc,uvalues))/len(samples)
    return meanL,meanU,lower[0],1-upper[-1]


def references():
    distributions=[tuple(F(x,4) for x in v) for v in product(range(5),repeat=3) if sum(v)==4]
    functions=[tuple(F(x,2) for x in v) for v in product(range(3),repeat=3) if v[0]>=v[1]>=v[2]]
    checks=0
    for p in distributions:
        for q in distributions:
            eps=max(abs(sum(p[:j])-sum(q[:j])) for j in range(1,4))
            for g in functions:
                delta=abs(sum((a-b)*v for a,b,v in zip(p,q,g)))
                assert delta<=eps*(g[0]-g[-1]);checks+=1
    samples=[F(-2),F(-1),F(0),F(1),F(2)]
    got=envelopes(samples,[(F(-1),F(3,5),F(9,10)),(F(1),F(1,10),F(2,5))])
    assert got==(F(3,5)*F(2,5)+F(1,10)*F(2,5),F(1,5)+F(9,10)*F(2,5)+F(2,5)*F(2,5),F(3,5),F(3,5))
    return checks


def radius(count,families,alpha):
    target=2*families/alpha
    estimate=math.sqrt(math.log(float(target))/(2*count))
    epsilon=F(math.ceil(estimate*10**6),10**6)
    while True:
        exponent=2*count*epsilon*epsilon;term=F(1);lower=F(1)
        for k in range(1,65):term=term*exponent/k;lower+=term
        if lower>=target:break
        epsilon+=F(1,10**6)
    return epsilon,dict(epsilon=str(epsilon),exponent=str(exponent),exp_taylor_lower=str(lower),
                        target_exp=str(target),terms=65,verified_2J_exp_minus_2M_epsilon2_le_alpha=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    source=root/'revisions/r8_random_thresholds';design=read(source/'design.json');finished=read(source/'FINISHED.json')
    protected=[source/name for name in ['design.json','thresholds.npy','nodes.jsonl','scatter.json','results.json']]
    for path in protected:assert sha(path)==finished['files'][path.name]
    count,banks=npy_thresholds(source/'thresholds.npy');assert count==design['thresholds']==131072
    eps,certificate=radius(count,F(2),F(1,40));save(args.out/'RADIUS.json',certificate)
    nodes=[json.loads(s) for s in (source/'nodes.jsonl').read_text().splitlines()]
    bad=exact(read(source/'scatter.json')['bad_event_bound']);rows=[]
    for column,length in enumerate([1,5]):
        records=sorted([r for r in nodes if r['L']==length],key=lambda r:r['index'])
        ordered=sorted(banks[column]);group=count//design['groups']
        expected=[ordered[0]]+[ordered[k-1] for k in range(group,count+1,group)]
        assert [r['t'] for r in records]==expected
        levels=[1,8,16,32,64,128];level_results=[]
        for groups in levels:
            selected=[records[0]]+[records[index] for index in range(128//groups,129,128//groups)]
            points=[(exact(r['t']),max(F(0),min(F(1),exact(r['lower']['probability']))),
                     max(F(0),min(F(1),exact(r['upper_quadratic']['probability'])+bad))) for r in selected]
            meanL,meanU,oscL,oscU=envelopes(list(map(exact,banks[column])),points)
            low=max(F(0),meanL-eps*oscL);high=min(F(1),meanU+eps*oscU)
            level_results.append(dict(groups=groups,nodes=len(points),empirical_lower=str(meanL),empirical_upper=str(meanU),
                 lower_oscillation=str(oscL),upper_oscillation=str(oscU),
                 float_node_statistical_diagnostic=[float(low),float(high)],
                 caveat='R8 node probabilities/bad-event term uncertified; not C0 or a fully certified confidence interval'))
        rows.append(dict(L=length,levels=level_results))
    # A distinct target: the R13 exact C0 model and its ONE certified lower node.
    anchor=read(root/'revisions/r11_anchor_certificate_v2/SCOPE.json');t0=F(anchor['threshold'])
    result13=read(root/'revisions/r13_scatter_checks/RESULT.json');p0=F(result13['ideal_fixed_threshold_success_lower_bound'])
    receipt=read(root/'revisions/R13_HISTORY.json');history=read(root/receipt['path']/'manifest.json')
    for name in ['revisions/r13_scatter_checks/RESULT.json','revisions/r11_anchor_certificate_v2/SCOPE.json']:
        assert sha(root/name)==history['files'][name]
    selected=next(r for r in nodes if r['L']==1 and r['index']==64);assert exact(selected['t'])==t0
    successes=sum(exact(t)<=t0 for t in banks[0]);assert successes==65536
    lower=p0*max(F(0),F(successes,count)-eps)
    uniform_cases=references()
    output=dict(sample_count=count,cdf_families=2,alpha='1/40',epsilon=str(eps),epsilon_approximate=float(eps),
        interpretation='Ideal IID threshold model, uniform over bounded monotone functions and adaptive certified nodes; no new draws',
        monotone_transfer_exact_checks=uniform_cases,staircase_tie_check=True,float_node_diagnostics=rows,
        separate_C0_single_certified_node=dict(t=str(t0),node_lower=str(p0),empirical_cdf=str(F(successes,count)),
             ideal_iid_random_threshold_lower=str(lower),approximate=float(lower),
             caveat='Only a model implication using archived floats; threshold generation/normalization not certified; lower is loose, not end-to-end certification'),
        threshold_input_intervals_available=False,full_numerical_certification=False,
        source_hashes={str(p.relative_to(root)):sha(p) for p in protected})
    save(args.out/'RESULTS.json',output)
    save(args.out/'COMPACT_BANK.json',dict(thresholds=[[float(t) for t in bank] for bank in banks]))
    print(json.dumps(dict(epsilon=float(eps),exact_transfer_checks=uniform_cases,
         float_diagnostics=[dict(L=r['L'],full=r['levels'][-1]['float_node_statistical_diagnostic']) for r in rows],
         separate_C0_single_node_lower=float(lower),full_numerical_certification=False),indent=2))


if __name__=='__main__':main()
