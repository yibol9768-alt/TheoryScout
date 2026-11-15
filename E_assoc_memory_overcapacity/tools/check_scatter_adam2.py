"""Standard-library residual-to-probability transfer for represented C0.

Uses the frozen R11/R12 inputs; C0 is the exact symmetric dyadic replay input,
not the unrounded Gaussian sample. Optional integration is a fixed-t lower bound.
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
from check_factor_adam2 import inputs as factor_inputs, endpoint_intervals, square_sum
from check_anchor_certificate import construction_arrays


def inputs(path):
    raw=path.read_bytes();d,n=struct.unpack('<QQ',raw[:16]);q=2*d
    values=array('d');values.frombytes(raw[16:])
    if sys.byteorder!='little':values.byteswap()
    assert len(values)==3*q*q+d*d
    return d,n,values[:q*q],values[q*q:2*q*q],values[2*q*q:3*q*q],values[3*q*q:]


def abs_sum(values):
    exponent=max(v.as_integer_ratio()[1].bit_length()-1 for v in values)
    total=0
    for v in values:
        p,q=v.as_integer_ratio();total+=abs(p) << (exponent-(q.bit_length()-1))
    return F(total,1<<exponent)


def operator_bound(matrix,q):
    rows=max(abs_sum(matrix[i*q:(i+1)*q]) for i in range(q))
    columns=max(abs_sum(matrix[j::q]) for j in range(q))
    bound=integer.I(rows*columns).sqrt()
    return F(bound.hi,integer.SCALE),dict(max_row_sum=str(rows),max_column_sum=str(columns),bound=bound.record())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--factor',type=Path,required=True)
    parser.add_argument('--anchor',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeat-kernel',action='store_true')
    parser.add_argument('--integrate',action='store_true');args=parser.parse_args()
    directory=args.input.resolve();factor=args.factor.resolve();anchor=args.anchor.resolve()
    args.out.mkdir(parents=True,exist_ok=False);I=integer.I
    build=read(directory/'BUILD.json')
    assert sha(directory/'rounding_scatter_adam2')==build['executable_sha256']
    for name,digest in build['source_sha256'].items():assert sha(Path(__file__).with_name(name))==digest
    norm_tests=[[0.,0.],[.125,-.75,2.],[2.**-1074,-2.**-1022],[2.**100,2.**-100,-.375]]
    for values in norm_tests:assert abs_sum(values)==sum(abs(F(v)) for v in values)
    # Tiny exact matrix and radical references, independent of native rounding.
    d,n,c,inverse,f,u=inputs(directory/'toy.bin');q=2*d
    audit=read(directory/'toy_AUDIT.json');assert sha(directory/'toy.bin')==audit['input_sha256']
    fields=audit['rounding_output'];get=lambda key:F.from_float(float.fromhex(fields[key]))
    epsC=get('scatter_operator_error_hex');epsR=get('inverse_max_residual_hex')
    cf,rf,ff=list(map(F,c)),list(map(F,inverse)),list(map(F,f))
    for i in range(q):
        residual=sum(abs(cf[i*q+j]-sum(ff[i*q+k]*ff[j*q+k] for k in range(q))) for j in range(q))
        assert residual<=epsC
        for j in range(q):assert abs(sum(ff[i*q+k]*rf[k*q+j] for k in range(q))-int(i==j))<=epsR
    dd,intervals=endpoint_intervals(directory/'toy_endpoint_intervals.bin');assert dd==d==2 and n==16
    S=[[cf[i*q+j] for j in range(d)] for i in range(d)]
    Z=[[4*cf[(i+d)*q+j] for j in range(d)] for i in range(d)]
    u1=[[F(3,10)*Z[i][j]/(abs(Z[i][j])+1) for j in range(d)] for i in range(d)]
    references=[]
    for i in range(d):
        for j in range(d):
            g=sum(u1[i][k]*S[k][j] for k in range(d))-Z[i][j]
            mean=(g-F(9,10)*Z[i][j])/F(19,10);variance=(F(99,100)*Z[i][j]**2+g*g)/F(199,100)
            ref=I(u1[i][j])-I(F(3,10)*mean)/(I(variance).sqrt()+1);index=i*d+j
            assert F(intervals[2*index])<=F(ref.lo,integer.SCALE)<=F(ref.hi,integer.SCALE)<=F(intervals[2*index+1])
            references.append(ref.record())
    d,n,c,inverse,f,u=inputs(directory/'actual.bin');q=2*d
    audit=read(directory/'actual_AUDIT.json');assert sha(directory/'actual.bin')==audit['input_sha256']
    assert 0<d<=1024 and 2*d<n<=2**32
    assert all(abs(v)<=2.**40 for group in (c,inverse,f,u) for v in group)
    assert all(c[i*q+j]==c[j*q+i] for i in range(q) for j in range(q))
    assert all(f[i*q+i]>0 and all(f[i*q+j]==0 for j in range(i+1,q)) for i in range(q))
    pd,pn,pf,pu,_,b,scale=factor_inputs(factor/'actual.bin')
    assert (pd,pn,pf,pu)==(d,n,f,u)
    oq,oldb,_,_=construction_arrays(anchor/'actual_construction.bin');assert oq==q and oldb==b
    fields=audit['rounding_output'];get=lambda key:F.from_float(float.fromhex(fields[key]))
    epsC=get('scatter_operator_error_hex');deltaR=q*get('inverse_max_residual_hex');assert 0<=deltaR<1
    endpoint=get('max_endpoint_error_hex');dd,intervals=endpoint_intervals(directory/'actual_endpoint_intervals.bin');assert dd==d
    for i,v in enumerate(u):assert max(abs(F(intervals[2*i])-F(v)),abs(F(intervals[2*i+1])-F(v)))<=endpoint
    if args.repeat_kernel:
        path=args.out/'endpoint_repeat.bin'
        text=subprocess.check_output([str(directory/'rounding_scatter_adam2'),str(directory/'actual.bin'),str(path)],text=True)
        (args.out/'kernel_repeat.txt').write_text(text)
        assert dict(line.split(maxsplit=1) for line in text.splitlines())==fields
        assert sha(path)==sha(directory/'actual_endpoint_intervals.bin')
    betaF,normF=operator_bound(f,q);betaR,normR=operator_bound(inverse,q)
    inverse_interval=I(betaR)/(1-I(deltaR));inverse_upper=F(inverse_interval.hi,integer.SCALE)
    white_interval=I(inverse_upper).square()*I(epsC);white=F(white_interval.hi,integer.SCALE);assert white<1
    factor_interval=I(betaF)*I(white)/(1+(1-I(white)).sqrt());zeta=F(factor_interval.hi,integer.SCALE)
    factor_component=I(zeta)*(2*I(betaF)+I(zeta))
    ur=I(square_sum(u)).sqrt();rlo=F(ur.lo,integer.SCALE);du=d*endpoint;assert rlo>du
    optimizer_component=I(betaF).square()*I(2*du)/I(rlo)
    previous=read(factor/'AUDIT.json')['rounding_output']
    el=F.from_float(float.fromhex(previous['max_left_residual_hex']));eb=F.from_float(float.fromhex(previous['max_block_residual_hex']))
    multiplication=(I(q*d).sqrt()*I(el)*I(betaF)+I(q*eb))/I(rlo)
    norm_difference=I(F(scale[0]))-ur;norm_error=F(max(abs(norm_difference.lo),abs(norm_difference.hi)),integer.SCALE)
    normalization=I(norm_error)*I(square_sum(b)).sqrt()/I(rlo)
    eta_interval=factor_component+optimizer_component+multiplication+normalization
    eta=F(eta_interval.hi,integer.SCALE)
    anchor_scope=read(anchor/'SCOPE.json');t=F(anchor_scope['threshold']);gamma=F(anchor_scope['positive_scaling'])
    gap_interval=I(n)*I(eta)*(1+I(abs(t))/I(n-1).sqrt());gap=F(gap_interval.hi,integer.SCALE);shift=gamma*gap
    evidence=dict(exact_abs_sum_cases=len(norm_tests),toy_radical_reference_cases=len(references),toy_reference=references,
         exact_toy_scatter_and_inverse_residual_checks=True,source_factor_and_B_identical=True,repeated_kernel=args.repeat_kernel,
         scatter_operator_error=str(epsC),inverse_residual_operator_error=str(deltaR),factor_norm=normF,inverse_approx_norm=normR,
         true_inverse_norm_upper=str(inverse_upper),whitened_residual=white_interval.record(),positive_definiteness_verified=True,
         nearby_factor_error=factor_interval.record(),factor_component=factor_component.record(),optimizer_component=optimizer_component.record(),
         multiplication_component=multiplication.record(),normalization_component=normalization.record(),eta=eta_interval.record(),eta_upper=str(eta),
         score_gap=gap_interval.record(),score_gap_upper=str(gap),scaled_constant_shift=str(shift),
         conditioning='Exact represented C0 from R8 replay lower triangle; not unrounded Gaussian input')
    (args.out/'CHECKS.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('scatter error <=',float(epsC),'whitened residual <=',float(white),'factor error <=',float(zeta),flush=True)
    print('eta <=',float(eta),'score gap <=',float(gap),flush=True)
    if args.integrate:
        spectrum=read(anchor/'actual_spectrum.json');epsilon=F(read(anchor/'ERRORS.json')['total']);constant=F(anchor_scope['exact_constant'])-shift
        counts=[1]*q+[n-q];pi=integer.pi_interval();nodes=integer.gauss8();bounds=[]
        for label,sign in [('lower',-1),('upper',1)]:
            coefficients=[F(v)+constant/n+sign*epsilon for v in spectrum['eigenvalues']]+[constant/n]
            record,enclosure=integer.certify(coefficients,counts,pi,nodes)
            (args.out/f'probability_{label}.json').write_text(json.dumps(record,indent=2)+'\n');bounds.append(enclosure)
            print('Completed shifted',label,'integral',flush=True)
        final=I.raw(bounds[0].lo,bounds[1].hi)
        result=dict(shifted_lower_event_probability=final.record(),ideal_fixed_threshold_success_lower_bound=str(F(final.lo,integer.SCALE)),
              target='Ideal exact Adam2 at represented C0; fixed t; LOWER bound only',
              caveat='Not unrounded Gaussian data, full random-threshold recall, single-dataset guarantee or independent review')
        (args.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
