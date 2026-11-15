"""Selected-point independent directed-interval reintegration.
Uses review17's DIFFERENT arithmetic/series backend, never the AA3 integer jets.
No finite-dimensional experiment is run.
"""
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'review17_checks'))
import interval_audit as other
sys.path.insert(0,str(ROOT/'aa3_checks'))
from AA3_numeric import resources

if __name__=='__main__':
    resources();started=time.monotonic()
    other.iv.dps=40
    cert=json.loads((ROOT/'aa3_checks/AA3_points_certificate.json').read_text())
    sources=['aa3_checks/AA3_independent.py','review17_checks/interval_audit.py','review17_checks/numerical.py','aa3_checks/AA3_points_certificate.json']
    hashes=lambda:{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    before=hashes();out=[]
    for row in cert['points']:
        if row['delta']!='3/10':continue
        e=other.I(row['e']);r=other.I('1/4');u=other.I('11/20');arms=[];values=[]
        for arm in row['rows']:
            b1=other.I(arm['beta1']);b2=other.I(arm['beta2']);lo,hi=arm['root']
            endpoints=[]
            for a in [other.I(lo),other.I(hi)]:
                vals=other.integrate(lambda z,n:other.endpoint(z,n,a,e,b1,b2),128,8)
                st=other.derived(vals,a,r)
                endpoints.append(st['new']/other.iv.sqrt(st['q'])-u)
            assert other.positive(-endpoints[0]) and other.positive(endpoints[1])
            a=other.box(lo,hi)
            vals=other.integrate(lambda z,n:other.endpoint(z,n,a,e,b1,b2),128,8)
            c0=other.derived(vals,a,r)['c0'];values.append(c0)
            arms.append(dict(beta1=arm['beta1'],beta2=arm['beta2'],root=arm['root'],c0=other.enc(c0),endpoint_offsets=[other.enc(x) for x in endpoints]))
            print('independent',row['e'],arm['beta1'],arm['beta2'],other.enc(c0)['decimal'],flush=True)
        gap1,gap2=values[2]-values[0],values[2]-values[1]
        assert other.positive(-gap1)
        assert other.positive(gap2) if row['e']=='1/2' else other.positive(-gap2)
        out.append(dict(e=row['e'],delta=row['delta'],arms=arms,gap1=other.enc(gap1),gap2=other.enc(gap2)))
    assert before==hashes()
    result=dict(status='PASS',selected_points=out,source_hashes=before,resources=resources(),seconds=time.monotonic()-started,
                scope='Reintegrates six root boxes and twelve root endpoints; does NOT independently reintegrate all continuous/history path cells.')
    with (ROOT/'aa3_checks/AA3_independent.json').open('x') as f:json.dump(result,f,indent=2)
    print('PASS',result['seconds'],result['resources'],flush=True)
