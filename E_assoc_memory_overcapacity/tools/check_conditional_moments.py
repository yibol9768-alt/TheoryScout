"""Audit stored moment propagation; optional exact replay comparison.

This is an arithmetic/consistency audit, not independent mathematical review.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse
import json
from certify_conditional_moments import I, S, read, sha, save, pad, moments, exact_basic_traces
from check_anchor_certificate import construction_arrays


def restore(record):
    assert record['denominator_power2']==160
    return I.raw(int(record['lower_integer']),int(record['upper_integer']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('certificate',type=Path)
    p.add_argument('--compare',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    inputs=read(args.certificate/'INPUTS.json');traces=read(args.certificate/'TRACES.json')
    result=read(args.certificate/'RESULT.json')
    for name,digest in inputs['source_hashes'].items():assert sha(root/name)==digest
    q,b,_,_=construction_arrays(root/'revisions/r11_anchor_certificate_v2/actual_construction.bin')
    mu,nn,t2=exact_basic_traces(b,q)
    assert dict(mu=str(mu),trN=str(nn),t2=str(t2))==traces['exact_Bhat_basic']
    assert q==inputs['q']==result['q'] and inputs['n']==result['n']
    for group in ['Bhat','C0']:
        restored={k:restore(v) for k,v in traces[group].items()}
        assert {k:v.record() for k,v in moments(restored,inputs['n']).items()}==result[group+'_moments']
    for name,record in traces['Bhat'].items():
        assert pad(restore(record),F(inputs['trace_perturbation_bounds'][name])).record()==traces['C0'][name]
    scope=read(root/'revisions/r11_anchor_certificate_v2/SCOPE.json')
    assert (restore(traces['C0']['mu'])-I(F(scope['center_anchor']))).record()==result['center_anchor_error']
    assert restore(result['C0_moments']['mean_V']).lo>S
    assert restore(result['C0_moments']['covariance_DV']).lo>0
    assert result['source_hashes']==inputs['source_hashes']
    if args.compare:
        for filename in ['INPUTS.json','TRACES.json','RESULT.json']:
            a=read(args.certificate/filename);b=read(args.compare/filename)
            a.pop('wall_seconds',None);b.pop('wall_seconds',None)
            assert a==b,filename
    checks=dict(source_hashes=len(inputs['source_hashes']),exact_basic_traces=3,
                trace_perturbations=8,moment_records_recomputed=22,
                center_anchor_error_recomputed=True,positive_signs_verified=2,
                comparison_mathematics_identical=bool(args.compare),independent_review=False)
    save(args.out,checks);print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
