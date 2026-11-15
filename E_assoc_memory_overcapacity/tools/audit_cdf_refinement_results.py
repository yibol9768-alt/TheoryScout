"""Audit finite refinement traces and exact staircase expectation sums."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import hashlib
import json


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    reference=None;initial={};resolution=[];trace_checks=0;jump_checks=0
    for path in args.inputs:
        data=read(path/'RESULT.json')
        for name,digest in data['source_hashes'].items():assert sha(root/name)==digest
        for row in data['rows']:
            length=row['L'];ts=list(map(F,row['t']));jumps=list(map(F,row['jumps']))
            assert ts==sorted(set(ts)) and all(w>=0 for w in jumps)
            nonzero={str(t):str(w) for t,w in zip(ts,jumps) if w}
            if reference is None:reference={}
            if length not in reference:reference[length]=nonzero
            else:assert nonzero==reference[length]
            previous=None
            for sweep in row['sweeps']:
                low=list(map(F,sweep['lower_CDF']));high=list(map(F,sweep['upper_CDF']))
                assert len(low)==len(high)==len(ts)
                assert low==sorted(low) and high==sorted(high)
                assert all(0<=a<=b<=1 for a,b in zip(low,high))
                if previous:
                    assert all(a>=b for a,b in zip(low,previous[0]))
                    assert all(a<=b for a,b in zip(high,previous[1]))
                bottom=sum(w*v for w,v in zip(jumps,low));top=sum(w*v for w,v in zip(jumps,high))
                assert bottom==F(sweep['staircase_expectation_lower']) and top==F(sweep['staircase_expectation_upper'])
                assert 0<=bottom<=top<=sum(jumps)<=1
                previous=low,high;trace_checks+=1;jump_checks+=2
            first=row['sweeps'][0];last=row['sweeps'][-1]
            a=F(first['staircase_expectation_lower']);b=F(first['staircase_expectation_upper'])
            if length in initial:assert (a,b)==initial[length]
            else:initial[length]=a,b
            lo=F(last['staircase_expectation_lower']);hi=F(last['staircase_expectation_upper'])
            floor=F(lo.numerator*10**8//lo.denominator,10**8)
            resolution.append(dict(L=length,cdf_nodes=len(ts),lower=str(lo),staircase_upper=str(hi),
                target_conditional_recall_lower_floor8=f'{float(floor):.8f}',lower_approximate=float(lo),
                staircase_upper_approximate=float(hi),threshold_width_pp=float((hi-lo)*100),
                width_reduction_fraction_approximate=float(1-(hi-lo)/(b-a))))
    for length in initial:
        rows=sorted([r for r in resolution if r['L']==length],key=lambda r:r['cdf_nodes'])
        assert all(F(a['lower'])<=F(b['lower']) and F(a['staircase_upper'])>=F(b['staircase_upper']) for a,b in zip(rows,rows[1:]))
    result=dict(exact_trace_enclosure_checks=trace_checks,exact_jump_sum_reconstructions=jump_checks,
        all_grids_same_staircase=True,all_initial_expectation_intervals_identical=True,
        tested_resolution_order_nested=True,resolution=resolution,
        source_results={str(path):sha(path/'RESULT.json') for path in args.inputs},
        caveat='Staircase upper endpoints do NOT bound target recall; no independent mathematical review')
    with (args.out/'CHECKS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(trace_checks=trace_checks,jump_sums=jump_checks,resolution=resolution),indent=2))


if __name__=='__main__':main()
