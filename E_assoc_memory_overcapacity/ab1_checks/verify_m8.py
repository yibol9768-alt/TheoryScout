#!/usr/bin/env python3
"""AB1: exact saved-record audit and complete directed-Decimal reintegration.
Default re-integrates all 173+64 cells. No existing file is written.
"""
import argparse, csv, hashlib, io, json, math, sys, time
from fractions import Fraction as F
from pathlib import Path
import AB1_decimal_backend as dec
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'ab1_checks'
EXPECTED = [('1/2',1,'29/100','9/10','99/100',32,-1),
            ('1/2',2,'29/100','99/100','9/10',49,1),
            ('2',1,'53/100','9/10','99/100',32,-1),
            ('2',2,'53/100','99/100','9/10',60,-1)]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def must(ok, msg):
    if not ok: raise AssertionError(msg)
def interval(x):
    if isinstance(x,dict): return tuple(F(int(x[k]),2**x['power']) for k in ['lo','hi'])
    return tuple(map(F,x))
def mul(x,y):
    vals=[a*b for a in x for b in y]; return min(vals),max(vals)
def div(x,y):
    must(y[0]>0,'positive denominator'); return mul(x,(1/y[1],1/y[0]))
def root(x):
    scale=2**192
    def floor(v): return math.isqrt((v.numerator*scale*scale)//v.denominator)
    lo,hi=floor(x[0]),floor(x[1]); return F(lo,scale),F(hi+(F(hi*hi,scale*scale)<x[1]),scale)
def encode(x): return [str(v) for v in x]
def overlap(x,y): return max(x[0],y[0])<=min(x[1],y[1])
def signed(x,sign): return x[0]>0 if sign>0 else x[1]<0
def cover(cells,amax,bmax):
    edges={F(0),amax}
    for row in cells:
        al,ah=map(F,row['a']); bl,bh=map(F,row['beta'])
        must(0<=al<ah<=amax and 0<=bl<bh<=bmax,'cell outside rectangle')
        edges.update((al,ah))
    edges=sorted(edges)
    for l,h in zip(edges,edges[1:]):
        bands=sorted(tuple(map(F,v['beta'])) for v in cells if F(v['a'][0])<h and F(v['a'][1])>l)
        cursor=F(0)
        for bl,bh in bands: must(bl==cursor,'gap or internal overlap'); cursor=bh
        must(cursor==bmax,'incomplete rectangle')
def ends_cover(rows,bmax):
    cursor=F(0)
    for l,h in sorted(tuple(map(F,r['beta'])) for r in rows):
        must(l==cursor and h>l,'endpoint gap/overlap'); cursor=h
    must(cursor==bmax,'endpoint incomplete')
def derived(cell,st,kind):
    j=interval(st['Jnormalized']); q=interval(st['q']); lam=interval(st['lambda'])
    must(q[0]>0 and lam[0]>0,'q/lambda positive')
    # derivative = a^3 p_j^2 * H, H=(15/16)Jnormalized/(sqrt(q)*Lambda).
    H=div(mul((F(15,16),)*2,j),mul(root(q),lam))
    al,ah=map(F,cell['a']); bl,bh=map(F,cell['beta'])
    p2=(1/(1+bh)**2,1/(1+bl)**2)
    actual=mul(mul((al**3,ah**3),p2),H)
    return dict(Jnormalized=encode(j),matched_derivative_over_a3p2=encode(H),
                matched_derivative=encode(actual),lambda_bound=encode(lam),q=encode(q),
                strict_for_a_positive=True,zero_at_a_zero=(al==0))
def saved_checks(verbose=True):
    integer=json.loads((BASE/'AB1_m8_integer_certificate.json').read_text())
    decimal=json.loads((BASE/'AB1_m8_decimal_certificate.json').read_text())
    must(len(integer['paths'])==len(decimal['paths'])==4,'four paths')
    rows=[]; endpoints=[]
    for pi,(spec,ip,dp) in enumerate(zip(EXPECTED,integer['paths'],decimal['paths'])):
        e,kind,amax,bmax,other,count,sign=spec
        must((ip['e'],ip['kind'],ip['amax'],ip['beta_max'],ip['other_beta'])==spec[:5],'path specification')
        must(F(ip['r'])==F(1,4) and F(ip['delta_certified'])==F(3,10),'path quantifiers')
        must(ip['stored_J_normalization']=='J_beta/(a*p_j^2), centered determinant','normalization')
        must(dp['e']==e and dp['kind']==kind,'Decimal path identity')
        must(len(ip['cells'])==len(dp['cells'])==count,'region cell count')
        must(len(ip['endpoint_cover'])==len(dp['endpoints'])==16,'endpoint count')
        cover(ip['cells'],F(amax),F(bmax));cover(dp['cells'],F(amax),F(bmax))
        ends_cover(ip['endpoint_cover'],F(bmax));ends_cover(dp['endpoints'],F(bmax))
        for ci,(ic,dc) in enumerate(zip(ip['cells'],dp['cells'])):
            must(dc['parent']==ci and dc['a']==ic['a'] and dc['beta']==ic['beta'],'cell identity')
            ij=interval(ic['J_over_a']); dj=interval(dc['Jnormalized'])
            il=interval(ic['lam']);dl=interval(dc['lambda'])
            must(signed(ij,sign) and signed(dj,sign) and il[0]>0 and dl[0]>0,'saved sign')
            must(overlap(ij,dj) and overlap(il,dl),'independent intervals disagree')
            row=dict(id=f'P{pi+1}-R{ci+1:03d}',path=pi+1,e=e,kind=kind,a=ic['a'],beta=ic['beta'],
                     sign=sign,integer_Jnormalized=encode(ij),**derived(ic,dc,kind))
            rows.append(row)
            if verbose: print(row['id'],'PASS saved records; derivative sign',sign,flush=True)
        for ei,(ic,dc) in enumerate(zip(ip['endpoint_cover'],dp['endpoints'])):
            must(ic['beta']==dc['beta'],'endpoint identity')
            bound=interval(ic['c1']);ind=interval(dc['c1'])
            must(bound[0]>F(11,20) and ind[0]>F(11,20) and overlap(bound,ind),'endpoint > .55')
            row=dict(id=f'P{pi+1}-E{ei+1:03d}',path=pi+1,e=e,kind=kind,a=amax,beta=dc['beta'],
                     integer_c1=encode(bound),decimal_c1=encode(ind))
            endpoints.append(row)
            if verbose: print(row['id'],'PASS saved endpoint > .55',flush=True)
    return integer,rows,endpoints

def export_tables(rows,ends):
    # Exact rational strings determine all decisions; no display float is read.
    for name,objects in [('AB1_m8_directed_envelopes.json',rows),('AB1_m8_endpoint_envelopes.json',ends)]:
        with (BASE/name).open('x') as f: json.dump(objects,f,indent=2)
    stream=io.StringIO();writer=csv.writer(stream)
    writer.writerow(['id','e','kind','a_lo','a_hi','beta_lo','beta_hi','J_lo','J_hi','D_over_a3p2_lo','D_over_a3p2_hi','D_lo','D_hi','sign_for_a_positive'])
    for r in rows: writer.writerow([r['id'],r['e'],r['kind'],*r['a'],*r['beta'],*r['Jnormalized'],*r['matched_derivative_over_a3p2'],*r['matched_derivative'],r['sign']])
    with (BASE/'AB1_m8_directed_envelopes.csv').open('x') as f: f.write(stream.getvalue())

def bindings():
    manifest=json.loads((BASE/'AB1_manifest.json').read_text())
    for rel,digest in manifest['sha256'].items(): must(sha(ROOT/rel)==digest,'SHA256 mismatch: '+rel)
    return sha(BASE/'AB1_manifest.json'),manifest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records-only',action='store_true')
    p.add_argument('--output',help='new JSON report below ab1_checks/; existing files never overwritten')
    args=p.parse_args();start=time.monotonic();res=dec.resources();mh,manifest=bindings()
    cert,rows,ends=saved_checks()
    must(json.loads((BASE/'AB1_m8_directed_envelopes.json').read_text())==rows,'derived envelope table')
    must(json.loads((BASE/'AB1_m8_endpoint_envelopes.json').read_text())==ends,'derived endpoint table')
    reintegrated=[];reends=[]
    if not args.records_only:
        for pi,(spec,path) in enumerate(zip(EXPECTED,cert['paths'])):
            e,kind,amax,bmax,other,count,sign=spec
            for ci,cell in enumerate(path['cells']):
                a=dec.I(*cell['a']);beta=dec.I(*cell['beta'])
                b1,b2=(beta,dec.I(other)) if kind==1 else (dec.I(other),beta)
                st=dec.state(a,dec.I(e),b1,b2,kind)
                enc=dec.encode(st);j=interval(enc['Jnormalized']);lam=interval(enc['lambda'])
                must(signed(j,sign) and lam[0]>0,'reintegration sign')
                must(overlap(j,interval(cell['J_over_a'])),'reintegration/integer disagreement')
                row=dict(id=f'P{pi+1}-R{ci+1:03d}',a=cell['a'],beta=cell['beta'],**derived(cell,enc,kind))
                reintegrated.append(row)
                print(row['id'],'PASS reintegrated; J=',enc['Jnormalized'],flush=True)
            for ei,cell in enumerate(path['endpoint_cover']):
                beta=dec.I(*cell['beta']);b1,b2=(beta,dec.I(other)) if kind==1 else (dec.I(other),beta)
                st=dec.state(dec.I(amax),dec.I(e),b1,b2,kind,mode='value',cells=96,order=8)
                must(st['c1'].l>dec.D('.55'),'reintegration endpoint')
                row=dict(id=f'P{pi+1}-E{ei+1:03d}',beta=cell['beta'],**dec.encode(st));reends.append(row)
                print(row['id'],'PASS reintegrated endpoint',row['c1'],flush=True)
    must(bindings()[0]==mh,'bindings changed during run')
    results=dict(status='PASS',mode='records_only' if args.records_only else 'full_reintegration',
                 region_count=len(rows),endpoint_count=len(ends),reintegrated_region_count=len(reintegrated),
                 reintegrated_endpoint_count=len(reends),manifest_sha256=mh,
                 paper_sha256={k:v for k,v in manifest['sha256'].items() if k.startswith('paper/')},
                 saved_envelopes_sha256=hashlib.sha256(canonical(dict(regions=rows,endpoints=ends))).hexdigest(),
                 regions=reintegrated,endpoints=reends,seconds=time.monotonic()-start,resources=dec.resources())
    results['result_sha256']=hashlib.sha256(canonical(dict(regions=reintegrated,endpoints=reends))).hexdigest()
    if args.output:
        out=(ROOT/args.output).resolve();must(out.is_relative_to(BASE),'output must stay in ab1_checks')
        with out.open('x') as f:json.dump(results,f,indent=2)
    print('ALL 173 REGION + 64 ENDPOINT CELLS PASS',results['mode'])
    print(json.dumps({k:v for k,v in results.items() if k not in ['regions','endpoints']},indent=2))
if __name__=='__main__':main()
