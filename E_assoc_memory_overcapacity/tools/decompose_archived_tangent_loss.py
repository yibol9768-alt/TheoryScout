"""Exact event arithmetic on frozen R8 floats; post hoc mechanism diagnostic.

This is NOT a C0 probability certificate or new confirmation experiment.
All-pairs means use two existing banks, not M*N independent observations.
"""
from array import array
from fractions import Fraction as F
from pathlib import Path
from bisect import bisect_left,bisect_right
import argparse
import ast
import hashlib
import json
import math
import sys
import time

from audit_threshold_cdf_transfer import npy_thresholds


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def read_dv(path):
    with path.open('rb') as f:
        assert f.read(6)==b'\x93NUMPY';version=tuple(f.read(2));assert version in [(1,0),(2,0)]
        size=int.from_bytes(f.read(2 if version==(1,0) else 4),'little');header=ast.literal_eval(f.read(size).decode('latin1'))
        assert header['descr']=='<f8' and not header['fortran_order'] and header['shape'][0]==2
        values=array('d');values.frombytes(f.read())
        if sys.byteorder!='little':values.byteswap()
        count=header['shape'][1];assert len(values)==2*count and all(math.isfinite(v) for v in values)
        return count,list(map(F.from_float,values[:count])),list(map(F.from_float,values[count:]))


def cutoffs(d,v,a,alpha,m):
    c=alpha*(d-a)**2/m;b=d-c
    result=(d*d/v if d>=0 else F(0),max(F(0),4*alpha*(d-alpha*v)),
            b*b/v if b>=0 else F(0),max(F(0),4*alpha*(d-alpha*v-c)))
    s,e,center,both=result
    assert 0<=both<=e<=s and 0<=both<=center<=s
    return result


def partition(counts,both_single,denominator):
    s,e,c,both=counts
    cats=dict(center_alone_only=e-both_single,energy_alone_only=c-both_single,
              either_alone=s-e-c+both_single,joint_only=both_single-both)
    assert all(n>=0 for n in cats.values()) and sum(cats.values())==s-both
    center2=s-c+e-both;energy2=s-e+c-both;assert center2+energy2==2*(s-both)
    return dict(event_counts=dict(zip(['success','energy_only_tangent','center_only_tangent','double_tangent'],counts)),
                event_rates={k:float(F(n,denominator)) for k,n in zip(['success','energy_only_tangent','center_only_tangent','double_tangent'],counts)},
                loss_categories=cats,loss_rate=str(F(s-both,denominator)),
                center_two_order_allocation=str(F(center2,2*denominator)),
                energy_two_order_allocation=str(F(energy2,2*denominator)),
                denominator=denominator)


def grid_means(samples,nodes,sorted_query_cutoffs,M):
    xs=sorted(t*t for t in nodes);assert len(xs)==len(set(xs))
    counts=[M-bisect_left(sorted_query_cutoffs,x) for x in xs]
    low=high=0
    for x in samples:
        j=bisect_left(xs,x);low+=counts[j] if j<len(xs) else 0
        j=bisect_right(xs,x)-1;high+=counts[j] if j>=0 else M
    return dict(nodes=len(xs),lower_numerator=low,upper_numerator=high,
                lower_rate=str(F(low,M*len(samples))),upper_rate=str(F(high,M*len(samples))))


def algebra_checks():
    checks=0
    for m in [1,15,16594]:
        for a in [F(-1),F(0),F(3)]:
            for alpha in [F(1,3),F(2)]:
                for d in [F(-1),F(0),F(1,2),F(2),F(4)]:
                    for root in [F(1,2),F(1),F(2)]:
                        v=root*root;cuts=cutoffs(d,v,a,alpha,m)
                        for t in [F(1,4),F(1),F(2),F(3)]:
                            score=d-t*root;c=alpha*(d-a)**2/m;e=alpha*(root-t/(2*alpha))**2
                            exact=[score>=0,score-e>=0,score-c>=0,score-e-c>=0]
                            assert exact==[x>=t*t for x in cuts]
                            assert score-(score-e-c)==e+c;checks+=1
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic()
    source=root/'revisions/r8_random_thresholds';finished=read(source/'FINISHED.json')
    names=['D_V.npy','thresholds.npy','design.json','scatter.json','results.json','nodes.jsonl']
    for name in names:assert sha(source/name)==finished['files'][name]
    h=read(root/'revisions/R21_HISTORY.json');manifest=read(root/h['path']/'manifest.json')
    for name in ['revisions/r20_threshold_family/RESULT.json','revisions/r20_threshold_family/DESIGN.json']:
        assert sha(root/name)==manifest['files'][name]
    design=read(root/'revisions/r20_threshold_family/DESIGN.json');a=F(design['a']);alpha=F(design['alpha']);m=design['n']-1
    M,D,V=read_dv(source/'D_V.npy');N,banks=npy_thresholds(source/'thresholds.npy');assert M==N==131072 and all(v>0 for v in V)
    assert all(t>0 for bank in banks for t in bank)  # No filtering or silent negative-branch discard.
    bank_squares=[[F.from_float(t)**2 for t in bank] for bank in banks]
    scatter=read(source/'scatter.json');old_a=F.from_float(scatter['moments']['mu']);old_b=F.from_float(math.sqrt(scatter['moments']['expected_V']))
    cuts=[[],[],[],[]];old_q=[]
    for d,v in zip(D,V):
        for column,value in zip(cuts,cutoffs(d,v,a,alpha,m)):column.append(value)
        base=v+(d-old_a)**2/m+old_b*old_b
        old_q.append((2*old_b*d/base)**2 if d>=0 else F(0))
    print('Exact cutoffs computed',time.monotonic()-start,flush=True)
    old=read(source/'results.json');rows=[];sorted_q=sorted(cuts[3])
    points=read(root/'revisions/r20_threshold_family/RESULT.json')['certified_nodes'];coarse={F(v['t']) for v in points}
    fine={F.from_float(json.loads(line)['t']) for line in (source/'nodes.jsonl').read_text().splitlines()};assert coarse<=fine
    for column,length in enumerate([1,5]):
        samples=bank_squares[column];ordered=sorted(samples);paired=[0]*4;paired_min=0;product_counts=[0]*4;product_min=0;old_paired=0;old_product=0
        for i,values in enumerate(zip(*cuts)):
            paired=[s+int(x>=samples[i]) for s,x in zip(paired,values)];paired_min+=int(min(values[1:3])>=samples[i])
            product_counts=[s+bisect_right(ordered,x) for s,x in zip(product_counts,values)]
            product_min+=bisect_right(ordered,min(values[1:3]));old_paired+=int(old_q[i]>=samples[i])
            old_product+=bisect_right(ordered,old_q[i])
        ref=next(r for r in old['rows'] if r['L']==length)
        assert paired[0]==ref['references']['actual']['count']
        assert old_paired==ref['references']['lower_quadratic']['count']
        assert sum(M-bisect_left(sorted_q,t) for t in samples)==product_counts[3]
        low18=grid_means(samples,coarse,sorted_q,M);low258=grid_means(samples,fine,sorted_q,M)
        assert low18['lower_numerator']<=low258['lower_numerator']<=product_counts[3]<=low258['upper_numerator']<=low18['upper_numerator']
        den=M*N;parts=dict(tangent=product_counts[0]-product_counts[3],
                         remaining_fine_mesh=product_counts[3]-low258['lower_numerator'],
                         coarse_to_fine_mesh=low258['lower_numerator']-low18['lower_numerator'])
        assert sum(parts.values())==product_counts[0]-low18['lower_numerator']
        row=dict(L=length,paired=partition(paired,paired_min,M),all_pairs=partition(product_counts,product_min,den),
                 R8_fixed_b_lower_paired_count=old_paired,R8_pair_counts_exactly_recovered=True,
                 R8_fixed_b_lower_all_pairs_count=old_product,
                 R8_fixed_b_lower_all_pairs_rate=str(F(old_product,den)),
                 R8_fixed_b_tangent_loss_rate=str(F(product_counts[0]-old_product,den)),
                 grid18=low18,grid258=low258,loss_to_empirical_grid18_numerators=parts,
                 loss_to_empirical_grid18_rates={k:str(F(v,den)) for k,v in parts.items()})
        rows.append(row);print('L',length,'all-pairs success/double',product_counts[0]/den,product_counts[3]/den,
                               'loss parts pp',{k:100*v/den for k,v in parts.items()},flush=True)
    output=dict(sample_queries=M,sample_thresholds=N,rows=rows,a=str(a),alpha=str(alpha),m=m,
                exact_algebra_checks=algebra_checks(),nonpositive_thresholds=0,no_samples_discarded=True,
                exact_arithmetic_for_archived_dyadic_inputs=True,new_samples=0,new_point_integrals=0,
                statistical_confidence_claim=False,all_pairs_are_not_independent_observations=True,
                target='Post hoc R8 floating represented score/energy diagnostic with R20-style anchors; not R13 exact C0',
                no_C0_probability_certificate=True,no_center_bias_causal_attribution=True,
                source_hashes={str((source/name).relative_to(root)):sha(source/name) for name in names},
                wall_seconds=time.monotonic()-start)
    save(args.out/'RESULT.json',output);print('Exact algebra checks',output['exact_algebra_checks'],'wall',output['wall_seconds'],flush=True)


if __name__=='__main__':main()
