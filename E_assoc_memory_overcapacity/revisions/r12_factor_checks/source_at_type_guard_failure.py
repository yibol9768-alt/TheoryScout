"""Standard-library checks and eta transfer for exact-factor Adam2.

The exact conditioning point C*=F F^T differs from the original R8 scatter.
--integrate certifies one fixed-threshold LOWER event, not full recall.
"""
from array import array
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import struct
import subprocess
import sys

import certify_weighted_chisquare as integer
from check_spectral_certificate import read,sha
from check_anchor_certificate import construction_arrays


def inputs(path):
    raw=path.read_bytes();d,n=struct.unpack('<QQ',raw[:16]);q=2*d
    values=array('d');values.frombytes(raw[16:])
    if sys.byteorder!='little':values.byteswap()
    assert len(values)==2*q*q+d*d+q*d+1
    i=0;result=[]
    for size in [q*q,d*d,q*d,q*q,1]:result.append(values[i:i+size]);i+=size
    return d,n,*result


def endpoint_intervals(path):
    raw=path.read_bytes();d,=struct.unpack('<Q',raw[:8]);v=array('d');v.frombytes(raw[8:])
    if sys.byteorder!='little':v.byteswap()
    assert len(v)==2*d*d
    assert all(v[2*i]<=v[2*i+1] for i in range(d*d))
    return d,v


def square_sum(values):
    # Every binary64 input is dyadic; accumulate integer squares exactly.
    exponent=max(x.as_integer_ratio()[1].bit_length()-1 for x in values)
    numerator=0
    for value in values:
        p,q=value.as_integer_ratio();e=q.bit_length()-1
        numerator+=(p*p) << (2*(exponent-e))
    return F(numerator,1 << (2*exponent))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--anchor',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeat-kernel',action='store_true')
    parser.add_argument('--integrate',action='store_true');args=parser.parse_args()
    directory=args.input.resolve();anchor=args.anchor.resolve();args.out.mkdir(parents=True,exist_ok=False)
    build=read(directory/'BUILD.json')
    assert sha(directory/'rounding_adam2_endpoint')==build['executable_sha256']
    assert sha(Path(__file__).with_name('rounding_adam2_endpoint.cpp'))==build['source_sha256']
    # Independent exact rational/160-bit radical reference for the nonzero toy.
    d,n,f,u,left,b,scale=inputs(directory/'toy.bin');q=2*d
    saved=read(directory/'toy_run.json');assert sha(directory/'toy.bin')==saved['input_sha256']
    dd,intervals=endpoint_intervals(directory/'toy_endpoint_intervals.bin');assert dd==d==2 and n==16
    ff=list(map(F,f));S=[[sum(ff[i*q+k]*ff[j*q+k] for k in range(q)) for j in range(d)] for i in range(d)]
    Z=[[4*sum(ff[(i+d)*q+k]*ff[j*q+k] for k in range(q)) for j in range(d)] for i in range(d)]
    u1=[[F(3,10)*Z[i][j]/(abs(Z[i][j])+1) for j in range(d)] for i in range(d)]
    reference=[];I=integer.I
    for i in range(d):
        for j in range(d):
            g=sum(u1[i][k]*S[k][j] for k in range(d))-Z[i][j]
            mean=(g-F(9,10)*Z[i][j])/F(19,10)
            variance=(F(99,100)*Z[i][j]**2+g*g)/F(199,100)
            target=I(u1[i][j])-I(F(3,10)*mean)/(I(variance).sqrt()+1)
            index=i*d+j
            assert F(intervals[2*index])<=F(target.lo,integer.SCALE)<=F(target.hi,integer.SCALE)<=F(intervals[2*index+1])
            reference.append(target.record())
    d,n,f,u,left,b,scale=inputs(directory/'actual.bin');q=2*d
    scope=read(directory/'SCOPE.json');audit=read(directory/'AUDIT.json')
    assert sha(directory/'actual.bin')==scope['input_sha256']==audit['input_sha256']
    assert 0<d<=1024 and 2*d<n<=2**32
    assert all(abs(v)<=2.**40 for group in (f,u,left,b,scale) for v in group) and scale[0]>0
    assert all(f[i*q+i]>0 and all(f[i*q+j]==0 for j in range(i+1,q)) for i in range(q))
    dd,intervals=endpoint_intervals(directory/'actual_endpoint_intervals.bin');assert dd==d
    fields=audit['rounding_output'];get=lambda key:F.from_float(float.fromhex(fields[key]))
    endpoint=get('max_endpoint_error_hex');left_error=get('max_left_residual_hex');block_error=get('max_block_residual_hex')
    for i,v in enumerate(u):
        assert abs(F(intervals[2*i])-F(v))<=endpoint and abs(F(intervals[2*i+1])-F(v))<=endpoint
    if args.repeat_kernel:
        path=args.out/'endpoint_repeat.bin'
        result=subprocess.check_output([str(directory/'rounding_adam2_endpoint'),str(directory/'actual.bin'),str(path)],text=True)
        (args.out/'kernel_repeat.txt').write_text(result)
        assert dict(line.split(maxsplit=1) for line in result.splitlines())==fields
        assert sha(path)==sha(directory/'actual_endpoint_intervals.bin')
    # Same B input as the already-certified anchored event is essential.
    oq,oldb,_,_=construction_arrays(anchor/'actual_construction.bin');assert oq==q and oldb==b
    ur=I(square_sum(u)).sqrt();rlo=F(ur.lo,integer.SCALE)
    du=d*endpoint;assert rlo>du
    fx2=square_sum(f[:d*q]);fy2=square_sum(f[d*q:]);b2=square_sum(b)
    norm_difference=I(scale[0])-ur
    norm_error=F(max(abs(norm_difference.lo),abs(norm_difference.hi)),integer.SCALE)
    optimizer=I(2*du)*I(fx2*fy2).sqrt()/I(rlo)
    multiplication=(I(q*d).sqrt()*I(left_error)*I(fx2).sqrt()+I(q*block_error))/I(rlo)
    normalization=I(norm_error)*I(b2).sqrt()/I(rlo)
    eta_interval=optimizer+multiplication+normalization
    eta=F(eta_interval.hi,integer.SCALE)
    anchor_scope=read(anchor/'SCOPE.json');t=F(anchor_scope['threshold']);gamma=F(anchor_scope['positive_scaling'])
    gap_interval=I(n)*I(eta)*(1+I(abs(t))/I(n-1).sqrt())
    gap=F(gap_interval.hi,integer.SCALE);shift=gamma*gap
    evidence=dict(toy_radical_reference_cases=len(reference),toy_reference=reference,
         input_hashes_checked=True,exact_positive_triangular_factor=True,original_r11_B_identical=True,
         repeated_kernel=args.repeat_kernel,endpoint_frobenius_error=str(du),norm_U_interval=ur.record(),
         optimizer_component=optimizer.record(),multiplication_component=multiplication.record(),normalization_component=normalization.record(),
         eta=eta_interval.record(),eta_upper=str(eta),score_gap=gap_interval.record(),score_gap_upper=str(gap),
         scaled_constant_shift=str(shift),conditioning='NEW exact Cstar=F F^T; original R8 scatter excluded')
    (args.out/'CHECKS.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('eta <=',float(eta),'score gap <=',float(gap),flush=True)
    if args.integrate:
        spectrum=read(anchor/'actual_spectrum.json');epsilon=F(read(anchor/'ERRORS.json')['total'])
        constant=F(anchor_scope['exact_constant'])-shift
        counts=[1]*q+[n-q];pi=integer.pi_interval();nodes=integer.gauss8();bounds=[]
        for label,sign in [('lower',-1),('upper',1)]:
            coefficients=[F(v)+constant/n+sign*epsilon for v in spectrum['eigenvalues']]+[constant/n]
            record,interval=integer.certify(coefficients,counts,pi,nodes)
            (args.out/f'probability_{label}.json').write_text(json.dumps(record,indent=2)+'\n');bounds.append(interval)
            print('Completed shifted',label,'integral',flush=True)
        enclosure=I.raw(bounds[0].lo,bounds[1].hi)
        result=dict(shifted_lower_event_probability=enclosure.record(),
                    ideal_fixed_threshold_success_lower_bound=str(F(enclosure.lo,integer.SCALE)),
                    target='Ideal exact Adam2 at NEW Cstar=F F^T; fixed t; lower bound only',
                    caveat='Not original R8 scatter, full random-threshold recall, single-dataset guarantee or independent review')
        (args.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
