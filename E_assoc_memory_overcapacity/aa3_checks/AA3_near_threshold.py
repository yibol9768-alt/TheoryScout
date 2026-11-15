"""Strict finite-delta illustration: the small-target threshold can move."""
from AA3_certify import *

if __name__=='__main__':
    resources();started=time.monotonic();numeric=json.loads((OUT/'AA3_numeric.json').read_text());cases=[]
    source=['aa3_checks/AA3_near_threshold.py','aa3_checks/AA3_certify.py','aa3_checks/AA3_interval_core.py','aa3_checks/AA3_numeric.json']
    hashes=lambda:{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source}
    before=hashes()
    for ee,dd,positive in [('21/20','1/100',True),('11/10','1/100',False),('11/10','1/4',True)]:
        rows=[]
        for b2 in ['0','99/100']:
            hint=next(x for x in numeric['rows'] if x['e']==float(Fraction(ee)) and x['delta']==float(Fraction(dd)) and x['beta1']==.9 and x['beta2']==float(Fraction(b2)))
            center=Fraction(round(hint['a']*10**15),10**15);rad=Fraction(2,10**13);lo,hi=center-rad,center+rad
            st=[state(F(a),F(ee),F('9/10'),F(b2),F('1/4'),128,10) for a in [lo,hi]]
            u=F(Fraction('1/4')+Fraction(dd))
            assert st[0]['c1'][1]<u[0] and st[1]['c1'][0]>u[1]
            mono=[]
            for j in range(16):
                v=state(box(hi*j/16,hi*(j+1)/16),F(ee),F('9/10'),F(b2),F('1/4'),48,4)
                assert v['lam'][0]>0
                mono.append(dict(a=[str(hi*j/16),str(hi*(j+1)/16)],lam=enc(v['lam'])))
            c0=state(box(lo,hi),F(ee),F('9/10'),F(b2),F('1/4'),128,10)['c0']
            rows.append(dict(beta2=b2,root=[str(lo),str(hi)],c0=enc(c0),endpoints=[enc(v['c1']) for v in st],monotonic=mono))
        va,vb=[(int(v['c0']['lo']),int(v['c0']['hi'])) for v in rows];gap=sub(vb,va)
        assert gap[0]>0 if positive else gap[1]<0
        cases.append(dict(e=ee,delta=dd,rows=rows,gap=enc(gap),positive=positive))
        print(ee,dd,enc(gap)['display'],flush=True)
    assert before==hashes()
    result=dict(status='PASS',cases=cases,source_hashes=before,seconds=time.monotonic()-started,resources=resources())
    with (OUT/'AA3_near_threshold_certificate.json').open('x') as f:json.dump(result,f,indent=2)
    print('PASS',result['seconds'],result['resources'],flush=True)
