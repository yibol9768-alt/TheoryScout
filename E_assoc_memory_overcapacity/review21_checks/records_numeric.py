"""Review 21: independent record verification and raw two-step Adam calculation.
No import or execution of audited code. All outputs use exclusive create.
"""
import json, hashlib, math, os, time
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy.special import roots_legendre, ndtr, ndtri
from scipy.optimize import brentq
from scipy.stats import nct, t

BASE = Path.cwd()
OUT = BASE / 'review21_checks'
def save(name, obj):
    with (OUT/name).open('x') as f: json.dump(obj,f,indent=2)
def load(name): return json.loads((BASE/name).read_text())
def resources():
    cg=Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().split('0::')[1].strip().lstrip('/')
    r={k:(cg/k).read_text().strip() for k in ['memory.max','memory.swap.max','memory.peak','memory.events']}
    r['threads']=len(list(Path('/proc/self/task').iterdir()))
    assert r['memory.max']=='3221225472' and r['memory.swap.max']=='0' and r['threads']<=2
    return r
def iv(x):
    assert x['power']==160
    l,h=F(x['lo'])/2**160,F(x['hi'])/2**160
    assert l<=h
    return l,h
def diff(a,b): return a[0]-b[1],a[1]-b[0]
def cover(seq,key,top):
    s=sorted(tuple(map(F,row[key])) for row in seq)
    assert s and s[0][0]==0 and s[-1][1]==top
    assert all(0<=l<h<=top for l,h in s)
    assert all(x[1]==y[0] for x,y in zip(s,s[1:]))
def records():
    results={}; hashes=[]
    for name in ['continuous','points','centered_paths','near_threshold']:
        c=load('aa3_checks/AA3_'+name+'_certificate.json')
        assert c['status']=='PASS'
        for p,h in c['source_hashes'].items():
            hashes.append(dict(certificate=name,path=p,match=hashlib.sha256((BASE/p).read_bytes()).hexdigest()==h))
        if name=='continuous':
            for row in c['continuous']:
                cover(row['cells'],'a',F(row['amax']))
                assert iv(row['c1_at_amax'])[0]>F('11/20')
                for cell in row['cells']:
                    assert iv(cell['lam'])[0]>0 and iv(cell['j1'])[1]<0
                    assert (iv(cell['j2'])[0]>0 if row['e']=='1/2' else iv(cell['j2'])[1]<0)
            results[name]=[len(row['cells']) for row in c['continuous']]
        if name=='centered_paths':
            rows=[]
            for row in c['paths']:
                cells=row['cells']; am,bm=F(row['amax']),F(row['beta_max'])
                assert am==F('29/100' if row['e']=='1/2' else '53/100')
                assert bm==F('9/10' if row['kind']==1 else '99/100')
                knots=sorted({F(x) for cell in cells for x in cell['a']})
                assert knots[0]==0 and knots[-1]==am
                for l,h in zip(knots,knots[1:]):
                    cover([cell for cell in cells if F(cell['a'][0])<(l+h)/2<F(cell['a'][1])],'beta',bm)
                for cell in cells:
                    assert 0<=F(cell['a'][0])<F(cell['a'][1])<=am
                    assert iv(cell['lam'])[0]>0
                    assert iv(cell['J_over_a'])[0]>0 if row['kind']==2 and row['e']=='1/2' else iv(cell['J_over_a'])[1]<0
                cover(row['endpoint_cover'],'beta',bm)
                assert all(iv(cell['c1'])[0]>F('11/20') for cell in row['endpoint_cover'])
                rows.append(dict(e=row['e'],kind=row['kind'],cells=len(cells),ends=len(row['endpoint_cover']),
                                 lambda_min=float(min(iv(x['lam'])[0] for x in cells)),c1_endpoint_min=float(min(iv(x['c1'])[0] for x in row['endpoint_cover']))))
            results[name]=rows
        if name in ['points','near_threshold']:
            groups=c['points'] if name=='points' else c['cases']
            for group in groups:
                target=F('1/4')+F(group['delta'])
                for row in group['rows']:
                    l,h=map(F,row['root']); assert 0<l<h
                    cover(row['monotonic'],'a',h)
                    assert all(iv(cell['lam'])[0]>0 for cell in row['monotonic'])
                    end=row['endpoint_c1'] if name=='points' else row['endpoints']
                    assert iv(end[0])[1]<target<iv(end[1])[0]
                vals=[iv(row['c0']) for row in group['rows']]
                for key,gap in ([('gap1',diff(vals[2],vals[0])),('gap2',diff(vals[2],vals[1]))] if name=='points' else [('gap',diff(vals[1],vals[0]))]):
                    saved=iv(group[key]); assert saved[0]<=gap[0]<=gap[1]<=saved[1]
                if name=='points':
                    assert iv(group['gap1'])[1]<0
                    assert iv(group['gap2'])[0]>0 if group['e']=='1/2' else iv(group['gap2'])[1]<0
                else: assert iv(group['gap'])[0]>0 if group['positive'] else iv(group['gap'])[1]<0
            results[name]=dict(groups=len(groups),roots=sum(len(g['rows']) for g in groups))
    delivery=load('aa3_checks/AA3_delivery.json')
    for key in ['source_hashes','new_file_hashes']:
        for p,h in delivery[key].items(): hashes.append(dict(certificate='delivery/'+key,path=p,match=hashlib.sha256((BASE/p).read_bytes()).hexdigest()==h))
    results['hash_checks']=len(hashes); results['hash_mismatches']=[x for x in hashes if not x['match']]
    return results

# A complex-step derivative of the ORIGINAL bias-corrected moment recursion.
# Neither endpoint formula nor its derivative is imported from the subject.
nodes,weights=roots_legendre(800)
z=7*(nodes+1); weights=weights*7*np.sqrt(2/np.pi)*np.exp(-z*z/2)
def endpoint(a,e,b1,b2):
    displacement=np.zeros_like(z,dtype=np.result_type(a,b1,b2,float)); m=displacement.copy(); v=displacement.copy()
    for step in [1,2]:
        g=displacement-z
        m=b1*m+(1-b1)*g
        v=b2*v+(1-b2)*g*g
        displacement=displacement-a*(m/(1-b1**step))/(np.sqrt(v/(1-b2**step))+e)
    return displacement
def signals(a,e,b1,b2,derivative=False):
    d=endpoint(a,e,b1,b2); B=np.dot(weights,z*d); V=np.dot(weights,d*d)
    q=1+1.5*(V-B); c=np.array([1-.75*B,.25+.75*B])/np.sqrt(q)
    if not derivative:return c
    pars=np.array([a,b1,b2],complex); deriv=[]
    for j in range(3):
        pp=pars.copy(); pp[j]+=1e-25j
        deriv.append(np.imag(signals(pp[0],e,pp[1],pp[2]))/1e-25)
    dc=np.array(deriv).T
    return c,dc,dc[0,1:]-dc[0,0]*dc[1,1:]/dc[1,0]
def match(e,delta,b1,b2):
    # Increasing bracket from zero, never jump over an initial crossing.
    grid=np.linspace(0,.29 if e<=.5 else .53,101)
    vals=[signals(a,e,b1,b2)[1] for a in grid]
    assert all(h>l for l,h in zip(vals,vals[1:]))
    k=next(k for k in range(1,len(grid)) if vals[k]>.25+delta)
    a=brentq(lambda a:signals(a,e,b1,b2)[1]-.25-delta,grid[k-1],grid[k],xtol=5e-16)
    c,dc,md=signals(a,e,b1,b2,True)
    return dict(a=a,c0=c[0],c1=c[1],matched_derivatives=md.tolist(),slope=dc[0,0]/dc[1,0],
                R0=ndtr(c[0]/np.sqrt(.5)-ndtri(.9)))
def numerics():
    out=[]
    for e,delta in [(e,d) for e in [.5,2] for d in [.01,.1,.25,.3]]+[(1.05,.01),(1.1,.01),(1.1,.25)]:
        arms=[match(e,delta,*b) for b in [(0,.99),(.9,0),(.9,.99)]]
        out.append(dict(e=e,delta=delta,arms=arms,gap1=arms[2]['c0']-arms[0]['c0'],gap2=arms[2]['c0']-arms[1]['c0'],
                        Rgap1=arms[2]['R0']-arms[0]['R0'],Rgap2=arms[2]['R0']-arms[1]['R0']))
    fixed=[dict(e=e,delta=delta,**match(e,delta,.4,.7)) for e in [.5,2] for delta in [.001,.01,.1,.3]]
    # A first-order matched projection does NOT remove arm-dependent finite-d drift.
    zcrit=ndtri(.9875); weakest=abs(out[7]['Rgap2']); sigma=.0002
    power=dict(zcrit=zcrit,M_normal=float(ndtr(np.sqrt(128)/2-zcrit)),M_half=float(ndtr(np.sqrt(128)/4-zcrit)),
               M_noncentral_t=float(nct.sf(t.ppf(.9875,127),127,np.sqrt(128)/2)),
               R_normal=float(ndtr(np.sqrt(1024)*weakest/sigma-zcrit)),R_half=float(ndtr(np.sqrt(1024)*weakest/(2*sigma)-zcrit)),
               R_noncentral_t=float(nct.sf(t.ppf(.9875,1023),1023,np.sqrt(1024)*weakest/sigma)),
               R_B90=(zcrit+ndtri(.9))**2*sigma**2/weakest**2,
               hoeffding_B90=2*(np.sqrt(np.log(1/.0125))+np.sqrt(np.log(10)))**2/weakest**2,
               bernstein_additive_B1024=14*np.log(2/.00625)/(3*(1024-1)),
               bernstein_additive_B_needed=1+14*np.log(2/.00625)/(3*weakest),
               finite_d_coefficient_difference_to_cancel_weakest_R=weakest*2048,
               finite_d_coefficient_difference_to_cancel_weakest_c0=abs(out[7]['gap2'])*2048)
    return dict(contrasts=out,fixed_history=fixed,power=power)
if __name__=='__main__':
    started=time.monotonic(); resources()
    paths=[BASE/'AA3_ADAM_PREDICTIONS.md',BASE/'AA3_PREREG_DRAFT.md',BASE/'Z3_ADAM_HISTORY.md',BASE/'PROOF_REVIEW17.md',BASE/'Z2B_BR_PROOF.md']+list((BASE/'aa3_checks').glob('*'))
    before={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    save('input_hashes.json',before)
    result=dict(records=records(),numeric=numerics(),resources=resources(),seconds=time.monotonic()-started)
    assert before=={p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in before}
    result['inputs_unchanged']=True
    save('records_numeric.json',result)
    print(json.dumps(result,indent=2))
