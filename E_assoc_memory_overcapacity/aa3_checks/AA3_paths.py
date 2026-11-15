"""Certify whole history paths and all 0<delta<=.3, not just one beta."""
from AA3_certify import *

def path(e_text,kind):
    e=F(e_text);r=F('1/4');amax=Fraction('29/100' if e_text=='1/2' else '53/100')
    bmax=Fraction('9/10' if kind==1 else '99/100')
    records=[];checks=0
    stack=[(amax*i/8,amax*(i+1)/8,bmax*j/4,bmax*(j+1)/4) for i in range(8) for j in range(4)]
    while stack:
        al,ah,bl,bh=stack.pop();bi=box(bl,bh)
        b1,b2=(bi,F('99/100')) if kind==1 else (F('9/10'),bi)
        try:
            st=state(box(al,ah),e,b1,b2,r)
            signok=st['j1'][1]<0 if kind==1 else (st['j2'][0]>0 if e_text=='1/2' else st['j2'][1]<0)
            good=signok and st['lam'][0]>0
        except AssertionError:
            good=False
        checks+=1
        if good:
            records.append(dict(a=[str(al),str(ah)],beta=[str(bl),str(bh)],lam=enc(st['lam']),J_over_a=enc(st['j'+str(kind)])))
        else:
            assert max((ah-al)/amax,(bh-bl)/bmax)>Fraction(1,2**17),(e_text,kind,al,ah,bl,bh)
            if (ah-al)/amax >= (bh-bl)/bmax:
                mid=(al+ah)/2;stack.extend([(al,mid,bl,bh),(mid,ah,bl,bh)])
            else:
                mid=(bl+bh)/2;stack.extend([(al,ah,bl,mid),(al,ah,mid,bh)])
        if checks%100==0:print('path',e_text,kind,'checks',checks,'accepted',len(records),'pending',len(stack),flush=True)
    # A separate full beta cover certifies existence by c1(amax)>r+.3.
    ends=[];stack=[(bmax*j/16,bmax*(j+1)/16) for j in range(16)]
    while stack:
        bl,bh=stack.pop();bi=box(bl,bh)
        b1,b2=(bi,F('99/100')) if kind==1 else (F('9/10'),bi)
        st=state(F(amax),e,b1,b2,r,96,8)
        if st['c1'][0]>F('11/20')[1]:
            ends.append(dict(beta=[str(bl),str(bh)],c1=enc(st['c1'])))
        else:
            assert bh-bl>bmax/2**16
            mid=(bl+bh)/2;stack.extend([(bl,mid),(mid,bh)])
    return dict(e=e_text,kind=kind,r='1/4',amax=str(amax),beta_max=str(bmax),other_beta='99/100' if kind==1 else '9/10',
                delta_certified='3/10',cells=records,endpoint_cover=ends,checks=checks)

if __name__=='__main__':
    resources();started=time.monotonic()
    sources=['aa3_checks/AA3_paths.py','aa3_checks/AA3_certify.py','aa3_checks/AA3_interval_core.py','paper/appendix/L_multistep.tex','Z3_ADAM_HISTORY.md','PROOF_REVIEW17.md']
    hashes=lambda:{s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources}
    before=hashes();result=dict(paths=[path(e,k) for e in ['1/2','2'] for k in [1,2]])
    assert before==hashes()
    result.update(status='PASS',source_hashes=before,hashes_unchanged=True,resources=resources(),seconds=time.monotonic()-started)
    with (OUT/'AA3_paths_certificate.json').open('x') as f:json.dump(result,f,indent=2)
    print('PASS',result['seconds'],result['resources'],flush=True)
