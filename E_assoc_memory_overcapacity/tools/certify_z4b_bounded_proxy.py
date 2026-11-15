"""Exact paired Z4B proxy moments and simultaneous bounded-iid confidence.

Ten fixed targets: seven matching errors and three fixed-slope proxies.
The statistical theorem needs iid datasets conditional on independent locks.
It does not require normality, independent queries, or correct model slopes.
It does not cover actual nonlinear rematching or alter registered decisions.
"""
from fractions import Fraction as F
from pathlib import Path
from itertools import combinations
import argparse
import math

import certify_weighted_chisquare as integer

I=integer.I
S=integer.SCALE
METHODS=['GD','Adam2','Adam.1','Sign']
PAIRS=[('GD','Sign'),('Adam2','Adam.1'),('Adam.1','Sign')]
TARGET=F(17,40)

def read(p):
    import json
    return json.loads(p.read_text())
def sha(p):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    import json
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')

def moments(counts):
    # Column order: old GD,Adam2,Adam.1,Sign; new in the same order.
    vectors=[[r[m][task] for task in [0,1] for m in range(4)] for r in counts]
    n=len(vectors);assert n==1280
    assert all(len(v)==8 and all(isinstance(x,int) and 0<=x<=1024 for x in v) for v in vectors)
    sums=[sum(v[j] for v in vectors) for j in range(8)]
    means=[F(v,n*1024) for v in sums]
    cov=[[F(n*sum(v[j]*v[k] for v in vectors)-sums[j]*sums[k],n*(n-1)*1024**2) for k in range(8)] for j in range(8)]
    return vectors,means,cov

def targets(data):
    result=[]
    for r in data['matching']:
        c=[F(0)]*8;constant=F(0)
        if r['type']=='absolute':c[4+METHODS.index(r['label'])]=1;constant=-TARGET
        else:
            a,b=r['label'].split('-');c[4+METHODS.index(a)]=1;c[4+METHODS.index(b)]=-1
        result.append(dict(label=r['label'],kind='matching',coefficients=c,constant=constant))
    for a,b in PAIRS:
        ka=F.from_float(data['slopes'][a]['k']);kb=F.from_float(data['slopes'][b]['k']);c=[F(0)]*8
        c[METHODS.index(a)]=1;c[METHODS.index(b)]=-1
        c[4+METHODS.index(a)]=-ka;c[4+METHODS.index(b)]=kb
        result.append(dict(label=a+'-'+b,kind='linear_proxy',coefficients=c,constant=TARGET*(ka-kb)))
    assert len(result)==10
    return result

def record(target,vectors,means,cov):
    n=len(vectors);c=target['coefficients'];constant=target['constant']
    mean=constant+sum(a*b for a,b in zip(c,means))
    variance=sum(c[j]*c[k]*cov[j][k] for j in range(8) for k in range(8));assert variance>=0
    # Independent direct recomputation from every paired dataset variable.
    values=[constant+sum(a*F(v,1024) for a,v in zip(c,row)) for row in vectors]
    assert sum(values)/n==mean
    assert sum((v-mean)**2 for v in values)/(n-1)==variance
    lower=constant+sum(min(F(0),a) for a in c);upper=constant+sum(max(F(0),a) for a in c);span=upper-lower
    assert span>0 and all(lower<=v<=upper for v in values)
    # Theorem 4, scaled to a deterministic support interval.  Twenty tails,
    # each delta=1/400, use log(800)<7.  No floating logarithm is used.
    radius=(I(14*variance/n)).sqrt()+I(F(49,3*(n-1))*span)
    rad=F(radius.hi,S);interval=[max(lower,mean-rad),min(upper,mean+rad)]
    diagonal=sum(c[j]**2*cov[j][j] for j in range(8))
    old_part=sum(c[j]*c[k]*cov[j][k] for j in range(4) for k in range(4))
    new_part=sum(c[j]*c[k]*cov[j][k] for j in range(4,8) for k in range(4,8))
    cross=2*sum(c[j]*c[k]*cov[j][k] for j in range(4) for k in range(4,8))
    assert old_part+new_part+cross==variance
    out=dict(label=target['label'],kind=target['kind'],coefficients=list(map(str,c)),constant=str(constant),
        mean=str(mean),sample_variance=str(variance),variance_of_mean=str(variance/n),support=list(map(str,[lower,upper])),span=str(span),
        radius=radius.record(),confidence_interval=list(map(str,interval)),approximate=list(map(float,interval)),
        paired_covariance_decomposition=dict(old=str(old_part),new_adjustment=str(new_part),old_new_cross=str(cross),
            diagonal_only_variance=str(diagonal),all_covariances_contribution=str(variance-diagonal)),
        direct_dataset_mean_and_variance_verified=True)
    if target['kind']=='matching':out['finite_sample_bound_certifies_one_pp_gate']=interval[0]>-F(1,100) and interval[1]<F(1,100)
    else:out['finite_sample_proxy_lower_strictly_positive']=interval[0]>0
    return out

def compute(data):
    assert data['methods']==METHODS and [tuple(p) for p in data['pairs']]==PAIRS
    vectors,means,cov=moments(data['dataset_success_counts'])
    lower_exp=sum(F(7)**j/math.factorial(j) for j in range(13));assert lower_exp>800
    rows=[record(t,vectors,means,cov) for t in targets(data)]
    return dict(n=len(vectors),queries_per_dataset=1024,statistical_unit='dataset; within-dataset dependencies unrestricted',target=str(TARGET),
        columns=['old_'+m for m in METHODS]+['new_'+m for m in METHODS],means=list(map(str,means)),unbiased_sample_covariance=[[str(v) for v in r] for r in cov],
        simultaneous_targets=10,tail_events=20,per_tail_failure='1/400',joint_failure_upper='1/20',lambda_rational='7',
        exp7_positive_Taylor_terms=13,exp7_lower_rational=str(lower_exp),log800_below7_exactly_verified=True,
        records=rows,unchanged_registered_decisions=data['decisions'],
        guarantee='At least95% simultaneous coverage of these ten fixed linear targets under conditionally iid evaluation datasets and independently fixed locks/coefficients',
        requires_Student_t_or_normality=False,actual_exact_matching_claim=False,conditional_on_gate_pass_coverage_claim=False,
        family_does_not_cover_all_historical_or_selected_analyses=True,independent_statistical_review=False)

def source_hashes(root):
    name='revisions/r14_z4b_matching/INPUTS.json';data=read(root/name)
    receipt=read(root/'revisions/R14_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
    assert sha(root/name)==manifest['files'][name]
    for p,h in data['source_hashes'].items():assert sha(root/p)==h,p
    lock=read(root/'z4b/formal/lock.json')
    for method in METHODS:
        alpha=next(c['empirical']['alpha'] for c in lock['cells'] if c['method']==method and c['target']==.425)
        assert alpha==data['slopes'][method]['alpha']
    names=[name,'tools/certify_z4b_bounded_proxy.py','tools/certify_weighted_chisquare.py']+list(data['source_hashes'])
    names+=['revisions/r47_primary_source/maurer_pontil_2009_v1.pdf','revisions/r47_primary_source/SOURCE.json']
    return data,{p:sha(root/p) for p in names}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];data,hashes=source_hashes(root);result=compute(data);result['source_hashes']=hashes
    save(a.out,result)
    for r in result['records']:print(r['kind'],r['label'],r['approximate'],flush=True)

if __name__=='__main__':main()
