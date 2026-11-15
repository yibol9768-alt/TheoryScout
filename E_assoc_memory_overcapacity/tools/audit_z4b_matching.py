"""Post hoc propagation of existing Z4B matching intervals; no new samples.

Exact rational vertex enumeration propagates the registered joint intervals.
The coverage remains nominal Student-t inference and the slopes are a local
model diagnostic, not certified derivatives of actual exactly matched recall.
"""
from fractions import Fraction as F
from itertools import combinations,product
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys

METHODS=['GD','Adam2','Adam.1','Sign']
PAIRS=[('GD','Sign'),('Adam2','Adam.1'),('Adam.1','Sign')]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x') as h:json.dump(value,h,indent=2);h.write('\n')
def exact(value):return F.from_float(float(value))


def solve(rows):
    matrix=[[F(v) for v in row] for row in rows]
    for col in range(4):
        pivot=next((r for r in range(col,4) if matrix[r][col]),None)
        if pivot is None:return None
        matrix[col],matrix[pivot]=matrix[pivot],matrix[col]
        scale=matrix[col][col];matrix[col]=[v/scale for v in matrix[col]]
        for row in range(4):
            if row!=col:
                scale=matrix[row][col];matrix[row]=[a-scale*b for a,b in zip(matrix[row],matrix[col])]
    return tuple(matrix[i][-1] for i in range(4))


def vertices(constraints):
    result=set()
    for chosen in combinations(constraints,4):
        point=solve([list(row)+[bound] for row,bound in chosen])
        if point is not None and all(sum(a*x for a,x in zip(row,point))<=bound for row,bound in constraints):result.add(point)
    assert result
    return sorted(result)


def constraints(rows,gate=False):
    result=[]
    for record in rows:
        row=[F(0)]*4
        if record['type']=='absolute':row[METHODS.index(record['label'])]=1
        else:
            a,b=record['label'].split('-');row[METHODS.index(a)]=1;row[METHODS.index(b)]=-1
        lower,upper=(-F(1,100),F(1,100)) if gate else (exact(record['lower']),exact(record['upper']))
        result.extend([(row,upper),([-a for a in row],-lower)])
    return result


def linear_range(points,coefficients):
    values=[sum(a*x for a,x in zip(coefficients,point)) for point in points]
    low,high=min(values),max(values)
    return dict(lower=str(low),upper=str(high),pp=[100*float(low),100*float(high)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--replay',type=Path);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    # Exact references include a lower-dimensional polytope.
    box=[(list(row),F(1)) for i in range(4) for sign in [-1,1] for row in [[sign if j==i else 0 for j in range(4)]]]
    points=vertices(box);assert len(points)==16
    assert linear_range(points,[1,-2,3,-4])['lower']=='-10'
    equality=box+[([1,-1,0,0],F(0)),([-1,1,0,0],F(0))]
    assert len(vertices(equality))==8 and linear_range(vertices(equality),[1,-1,0,0])['upper']=='0'
    if args.replay:
        data=json.loads((args.replay/'INPUTS.json').read_text())
    else:
        import numpy as np
        from scipy.stats import t as student
        root=Path(__file__).resolve().parents[1];formal=root/'z4b/formal'
        report=json.loads((formal/'analysis/report.json').read_text())
        assert report['data_errors']==[]
        match=[r for r in csv.DictReader((formal/'analysis/matching.csv').open()) if r['role']=='primary']
        old=[r for r in csv.DictReader((formal/'analysis/old_pairs.csv').open()) if r['role']=='primary']
        lock=json.loads((formal/'lock.json').read_text())
        sys.path.insert(0,str(root/'z4b'))
        spec=importlib.util.spec_from_file_location('z4b_frozen_model',root/'z4b/model.py')
        model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
        slopes={}
        for method in METHODS:
            alpha=next(c['empirical']['alpha'] for c in lock['cells'] if c['method']==method and c['target']==.425)
            ratios=[]
            for h in [1e-5,5e-6]:
                derivative=lambda task:(model.predict(method,alpha+h,128,task)['finite']-model.predict(method,alpha-h,128,task)['finite'])/(2*h)
                ratios.append(derivative('old')/derivative('new'))
            assert abs(ratios[0]-ratios[1])<1e-6
            slopes[method]=dict(alpha=alpha,k=ratios[0],half_step_k=ratios[1])
        counts=[];hashes={}
        for seed in range(1280):
            path=formal/f'eval_128_{seed}.npz';digest=sha(path);assert digest==report['raw_hashes'][path.name];hashes[path.name]=digest
            with np.load(path,allow_pickle=False) as raw:
                assert raw['scheduled'][1] and raw['valid'][0,1].all()
                ranks=raw['ranks'][0,1];assert ranks.shape==(4,2,1024) and (ranks>=1).all() and (ranks<=4096).all()
                assert len(set(raw['ids'].tolist()))==1024
                assert np.array_equal(raw['alphas'][0,1],np.array([slopes[m]['alpha'] for m in METHODS]))
                counts.append((ranks<=410).sum(axis=-1).tolist())
        means=np.asarray(counts,dtype=float)/1024
        for r in match:
            if r['type']=='absolute':values=means[:,METHODS.index(r['label']),1]-.425
            else:
                a,b=r['label'].split('-');values=means[:,METHODS.index(a),1]-means[:,METHODS.index(b),1]
            critical=float(student.ppf(1-.025/14,1279));se=float(values.std(ddof=1)/math.sqrt(1280));mean=float(values.mean())
            assert abs(mean-float(r['mean']))<1e-14 and abs(se-float(r['se']))<1e-14
            assert abs(mean-critical*se-float(r['lower']))<1e-14 and abs(mean+critical*se-float(r['upper']))<1e-14
        for r in old:
            a,b=r['pair'].split('-');values=means[:,METHODS.index(a),0]-means[:,METHODS.index(b),0]
            critical=float(student.ppf(1-.025/6,1279));se=float(values.std(ddof=1)/math.sqrt(1280));mean=float(values.mean())
            assert abs(mean-float(r['mean']))<1e-14 and abs(mean-critical*se-float(r['lower']))<1e-14 and abs(mean+critical*se-float(r['upper']))<1e-14
        data=dict(methods=METHODS,pairs=PAIRS,matching=match,old=old,slopes=slopes,dataset_success_counts=counts,
                  raw_hashes=hashes,decisions=report['decisions'],source_hashes={str(p.relative_to(root)):sha(p) for p in [root/'Z4B_PREREG.md',formal/'lock.json',formal/'analysis/matching.csv',formal/'analysis/old_pairs.csv',formal/'analysis/report.json',root/'z4b/model.py']})
    save(args.out/'INPUTS.json',data)
    # Compact counts suffice to independently reproduce every registered CI.
    # Replay uses the archived t critical values, with their approximation scope.
    for record in data['matching']+data['old']:
        pair=record.get('pair')
        if pair is not None or record['type']=='pairwise':
            a,b=(pair or record['label']).split('-');task=0 if pair else 1
            values=[(row[METHODS.index(a)][task]-row[METHODS.index(b)][task])/1024 for row in data['dataset_success_counts']]
        else:
            values=[row[METHODS.index(record['label'])][1]/1024-.425 for row in data['dataset_success_counts']]
        assert len(values)==1280
        mean=math.fsum(values)/len(values)
        se=math.sqrt(math.fsum((v-mean)**2 for v in values)/(len(values)-1)/len(values))
        critical=float(record['critical'])
        assert abs(mean-float(record['mean']))<1e-14 and abs(se-float(record['se']))<1e-14
        assert abs(mean-critical*se-float(record['lower']))<1e-14 and abs(mean+critical*se-float(record['upper']))<1e-14
    region=constraints(data['matching']);actual=vertices(region);gate=vertices(constraints(data['matching'],True))
    results=[]
    for a,b in PAIRS:
        ka,kb=exact(data['slopes'][a]['k']),exact(data['slopes'][b]['k']);coefficients=[F(0)]*4
        coefficients[METHODS.index(a)]=-ka;coefficients[METHODS.index(b)]=kb
        correction=linear_range(actual,coefficients);gate_range=linear_range(gate,coefficients)
        ra=next(r for r in data['matching'] if r['type']=='absolute' and r['label']==a)
        rb=next(r for r in data['matching'] if r['type']=='absolute' and r['label']==b)
        point=-ka*exact(ra['mean'])+kb*exact(rb['mean'])
        old=next(r for r in data['old'] if r['pair']==a+'-'+b)
        proxy_low=exact(old['lower'])+F(correction['lower']);proxy_high=exact(old['upper'])+F(correction['upper'])
        results.append(dict(pair=a+'-'+b,point_correction_pp=100*float(point),correction_range=correction,full_gate_range=gate_range,
                     linearized_old_proxy_range=dict(lower=str(proxy_low),upper=str(proxy_high),pp=[100*float(proxy_low),100*float(proxy_high)])))
    output=dict(results=results,matching_polytope_vertices=len(actual),full_gate_vertices=len(gate),exact_polytope_reference_cases=2,
                compact_count_statistics_recomputed=True,
                inference='Post hoc local linear model diagnostic; joint nominal95% t region; not actual exactly matched recall',
                formula='chi_ab=-k_a theta_a+k_b theta_b; theta true mean-new minus target at fixed locks',
                statistical_unit='1280 datasets, never 1310720 independent queries',coverage_error_budgets=dict(matching=.025,old=.025),
                unchanged_registered_decisions=data['decisions'])
    save(args.out/'RESULTS.json',output)
    save(args.out/'VERTICES.json',dict(matching=[[str(x) for x in p] for p in actual],full_gate=[[str(x) for x in p] for p in gate]))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
