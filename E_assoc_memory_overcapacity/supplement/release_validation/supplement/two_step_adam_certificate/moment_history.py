"""Isolate the two carried state variables at fixed e=1, matched c1=2/5."""
from adam_integrals import *
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--points',default='two_step_adam_certificate/matched_chain.json');parser.add_argument('--output',default='two_step_adam_certificate/moment_history.json');args=parser.parse_args()
    start=time.monotonic();out=[]
    for bs,lo,hi in [(('0','0'),'0.388542613','0.388542635'),(('9/10','0'),'0.371480025','0.371480047'),(('0','99/100'),'0.398617421','0.398617443')]:
        betas=tuple(F(b) for b in bs);e=O;der=[]
        for k in range(16):
            a=hull(rat(k,40),rat(k+1,40));vals,err=integ(a,e,64,4,betas);s=stats(a,vals)
            assert s['N'][0]>0
            der.append(dict(a=[str(Fraction(k,40)),str(Fraction(k+1,40))],integrals=[show(v) for v in vals],N=show(s['N'])))
        ends=[]
        for x,sg in [(lo,-1),(hi,1)]:
            vals,err=integ(F(x),e,192,10,betas);s=stats(F(x),vals)
            assert s['H'][1]<0 if sg<0 else s['H'][0]>0
            ends.append(dict(a=x,H=show(s['H']),integrals=[show(v) for v in vals],remainders=[show(v) for v in err]))
        a=hull(F(lo),F(hi));vals,err=integ(a,e,192,10,betas);s=stats(a,vals);c0=div(sub(O,s['B']),sqrt(s['q']))
        out.append(dict(beta=list(bs),e='1',a=[lo,hi],derivative_slabs=der,endpoints=ends,integrals=[show(v) for v in vals],remainders=[show(v) for v in err],c0=show(c0)))
        print(bs,lo,hi,show(c0)['display_only'],flush=True)
    with open(args.points) as f:pt=json.load(f)
    def dec(v):return int(v['lo']),int(v['hi'])
    actual=dec(pt['points'][2]['c0']);stateless=dec(out[0]['c0']);momonly=dec(out[1]['c0']);rmsonly=dec(out[2]['c0'])
    gaps=dict(adding_momentum_with_rms=sub(rmsonly,actual),adding_momentum_without_rms=sub(stateless,momonly),adding_rms_with_momentum=sub(actual,momonly),adding_rms_without_momentum=sub(rmsonly,stateless),stateless_minus_actual=sub(stateless,actual))
    assert all(v[0]>0 for v in gaps.values())
    result=dict(status='PASS',r='0',u='2/5',e='1',models=out,gaps={k:show(v) for k,v in gaps.items()},source_hashes=sources(),seconds=time.monotonic()-start,resources=resources())
    with open(args.output,'x') as f:json.dump(result,f,indent=2)
    print('PASS',{k:show(v)['display_only'] for k,v in gaps.items()},'seconds',result['seconds'],'resources',result['resources'],flush=True)
