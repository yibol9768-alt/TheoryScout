"""Freeze-checked numerical readout tables and independent fixed selectors."""
import json
import time
import numpy as np
from design import *
from common import OUT,begin,finish,prior_elapsed

def main():
    state=begin('prepare')
    assert not (OUT/'readout_tables.npz').exists() and not (OUT/'forecasts.json').exists()
    tables={}; audit=[]
    pairs=sorted({(d*l,value_dim(d*l,d,r)) for d in DIMS for l in LOADS for r in READOUTS})
    for n,p in pairs:
        v=iid_integral(RHO_GRID,n,p,96)
        check_ids=np.arange(0,len(RHO_GRID),8)
        other=iid_integral(RHO_GRID[check_ids],n,p,64)
        mid=(RHO_GRID[check_ids[:-1]]+RHO_GRID[check_ids[:-1]+1])/2
        exact=iid_integral(mid,n,p,96)
        interpolation=np.interp(mid,RHO_GRID,v)
        qdiff=float(np.max(np.abs(v[check_ids]-other)))
        idiff=float(np.max(np.abs(exact-interpolation)))
        audit.append({'n':n,'p':p,'quadrature_max':qdiff,'interpolation_max':idiff})
        tables[f'n{n}_p{p}']=v
        print(f'table n={n} p={p} quadrature={qdiff:.7g} interpolation={idiff:.7g}',flush=True)
    np.savez(OUT/'readout_tables.npz',**tables)
    # Fail the numerical gate; never silently change quadrature/grid.
    assert max(r['quadrature_max'] for r in audit)<=.002
    assert max(r['interpolation_max'] for r in audit)<=.0005
    readout=Readout(OUT/'readout_tables.npz'); forecasts=[]; moments=[]
    for family in FAMILIES:
        for d in DIMS:
            for a in ALPHAS:
                for load in LOADS:
                    n=d*load; ids=query_ids(n); mask=[bins(n,ids)==b for b in range(4)]
                    Zs=[seed_rng(seed,d,load,a,family,2).standard_normal((len(ids),d))*
                        np.sqrt(spectrum(d,a,family)/d) for seed in FORECAST_SEEDS]
                    for method in METHODS:
                        ker=Kernel(d,load,a,family,method); um,uq=ker.unconditional()
                        moments.append({'family':family,'d':d,'alpha':a,'load':load,'method':method,
                            'm_bins':[float(um[x].mean()) for x in mask],
                            'Q_bins':[float(uq[x].mean()) for x in mask]})
                        cond=[ker.conditional(Z)[:2] for Z in Zs]
                        for r in READOUTS:
                            p=value_dim(n,d,r)
                            vals=[float(readout(m/np.sqrt(Q),n,p).mean()) for m,Q in cond]
                            forecasts.append({'family':family,'d':d,'alpha':a,'load':load,
                                'method':method,'readout':r,'p':p,'predicted':float(np.mean(vals)),
                                'integration_se':float(np.std(vals,ddof=1)/np.sqrt(len(vals)))})
                        del ker
                    print(f'forecast {family} d={d} a={a} load={load} elapsed={time.monotonic()-state[0]:.1f}s',flush=True)
                    assert time.monotonic()-state[0]+prior_elapsed('prepare')<3150
    (OUT/'forecasts.json').write_text(json.dumps(forecasts))
    (OUT/'unconditional_moments.json').write_text(json.dumps(moments))
    (OUT/'numerical_audit.json').write_text(json.dumps(audit,indent=2))
    finish(state,forecast_rows=len(forecasts),moment_rows=len(moments),table_pairs=len(pairs),
           max_quadrature_difference=max(r['quadrature_max'] for r in audit),
           max_interpolation_difference=max(r['interpolation_max'] for r in audit))

if __name__=='__main__': main()
