"""Pool R20 C0 point certificates across list sizes; no new integration."""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_left
import argparse
import hashlib
import json

from audit_threshold_cdf_transfer import envelopes,npy_thresholds


def capped_lower(samples,points,epsilon):
    """Exact optimum among g_c=min(g,c), 0<=c<=max(g)."""
    ordered=sorted(points);xs=[p[0] for p in ordered];heights=[p[1] for p in ordered]
    for j in range(len(heights)-2,-1,-1):heights[j]=max(heights[j],heights[j+1])
    counts=[0]*(len(points)+1)
    for t in samples:counts[bisect_left(xs,t)]+=1
    heights.append(F(0));best=(F(0),F(0),F(0))
    for cap in sorted(set(heights)):
        mean=sum(c*min(h,cap) for c,h in zip(counts,heights))/len(samples)
        candidate=(mean-epsilon*cap,cap,mean)
        if candidate>best:best=candidate
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    result_path=args.input/'RESULT.json';result=json.loads(result_path.read_text())
    design=json.loads((args.input/'DESIGN.json').read_text())
    assert result['fixed_node_lower_numerical_inclusion'] and not result['full_random_threshold_numerical_certification']
    bank_path=root/'revisions/r8_random_thresholds/thresholds.npy'
    digest=hashlib.sha256(bank_path.read_bytes()).hexdigest()
    assert digest==design['source_hashes']['revisions/r8_random_thresholds/thresholds.npy']
    count,banks=npy_thresholds(bank_path);assert count==design['sample_count']
    eps=F(design['epsilon']);rows=[]
    for column,length in enumerate([1,5]):
        levels=[];samples=list(map(F.from_float,banks[column]))
        for groups in [1,2,4,8]:
            indices={0,*range(128//groups,129,128//groups)}
            nodes=[c for c in result['certified_nodes'] if c['index'] in indices]
            assert len({c['t'] for c in nodes})==len(nodes)
            points=[(F(c['t']),F(c['target_probability_lower']),F(1)) for c in nodes]
            mean,_,osc,_=envelopes(samples,points)
            lower=max(F(0),mean-eps*osc)
            original=next(r for r in result['random_threshold_model_lower'] if r['L']==length)['levels'][[1,2,4,8].index(groups)]
            separate=F(original['ideal_iid_model_lower']);chosen=max(lower,separate)
            capped,cap,capped_mean=capped_lower(samples,points,eps)
            assert capped>=chosen
            # The common oscillation can change upon pooling. Verify the actual
            # penalized bound rather than assuming improvement from more nodes.
            levels.append(dict(groups=groups,pooled_nodes=len(nodes),empirical_mean_lower=str(mean),
                               oscillation=str(osc),ideal_iid_model_lower=str(lower),approximate=float(lower),
                               gain_vs_separate_lists=str(lower-separate),
                               chosen_cap=str(cap),capped_empirical_mean=str(capped_mean),
                               chosen_model_lower=str(capped),chosen_approximate=float(capped)))
        rows.append(dict(L=length,levels=levels))
    output=dict(interpretation='Same C0 probability p(t) for both lists; all certified nodes may be pooled',
                model_lower=rows,source_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                threshold_bank_sha256=digest,new_integrations=0,new_samples=0,
                full_random_threshold_numerical_certification=False,
                caveat='Ideal IID CDF transfer only; archived threshold generation/normalization uncertified')
    with (args.out/'POOLED.json').open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps([dict(L=r['L'],lower=r['levels'][-1]['chosen_approximate']) for r in rows],indent=2))


if __name__=='__main__':main()
