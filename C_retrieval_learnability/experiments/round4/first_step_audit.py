"""Reconstruct actual first effective score shifts including temperature changes.
Reporting audit only: no new training trajectory, no frozen file modifications.
"""
import experiment as c
from experiment import np,json
from run import identifier

def main():
    rows=[]
    for path in sorted((c.OUT/'raw').glob('*.json')):
        r=json.loads(path.read_text()); cfg=r['config']; d,s,m,pa=(cfg[k] for k in ('d','seed','mode','schedule'))
        A,p=c.data(d,s); b=None
        if m in ('b','theta','solve'):
            b,_=c.root_bias(16*(p['U']@p['V'].T),A)
            if m=='b': p['b']=b.copy()
            if m=='theta': p['theta']=-b/16
        if m=='b_legacy': p['b']=np.zeros(560)
        if pa=='train': p['a']=np.array([np.log(16.)])
        _,shift0=c.effective(p,m,16.,b)
        _,g=c.obj(p,A,m,16.,b)
        q={k:v.copy() for k,v in p.items()}; c.Adam(q,r['rates']).update(q,g)
        beta1=float(np.exp(q['a'][0])) if pa=='train' else 16*256**(1/c.T)
        qb=b
        if m=='solve': qb,_=c.root_bias(beta1*(q['U']@q['V'].T),A,b)
        _,shift1=c.effective(q,m,beta1,qb)
        # A coordinate-only audit keeps beta fixed and, for solve, reuses old b.
        _,coordinate_shift=c.effective(q,m,16.,b)
        real=float(np.sqrt(np.mean((shift1-shift0)**2)))
        coord=float(np.sqrt(np.mean((coordinate_shift-shift0)**2)))
        assert abs(coord-r['calibration']['initial_actual_bias_effective_move'])<1e-12
        rows.append(dict(id=path.stem,**cfg,beta0=16.,beta1=beta1,
                         reference_target=r['calibration']['bias_reference_effective_move'],
                         coordinate_only_rms=coord,actual_effective_shift_rms=real))
    c.save(c.OUT/'first_step_audit.json',rows)
    print(json.dumps(dict(runs=len(rows),max_actual_shift=max(r['actual_effective_shift_rms'] for r in rows))))
if __name__=='__main__': main()
