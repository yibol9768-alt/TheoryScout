import hashlib
import json
from pathlib import Path
import sympy as sp
from environment import resources


def main():
    B,V,Ba,Va,Be,Ve=sp.symbols('B V Ba Va Be Ve',real=True)
    q=1-2*B+2*V
    c0=(1-B)/sp.sqrt(q); c1=B/sp.sqrt(q)
    jac=sp.Matrix([[sp.diff(c0,x) for x in (B,V)],[sp.diff(c1,x) for x in (B,V)]])
    assert sp.simplify(jac.det()-1/q**2)==0
    N=Ba*(1-B+2*V)-B*Va
    ca=sp.diff(c1,B)*Ba+sp.diff(c1,V)*Va
    assert sp.simplify(ca-N/q**sp.Rational(3,2))==0
    ce=sp.diff(c1,B)*Be+sp.diff(c1,V)*Ve
    matched=(sp.diff(c0,B)*Be+sp.diff(c0,V)*Ve)-(sp.diff(c0,B)*Ba+sp.diff(c0,V)*Va)*ce/ca
    assert sp.simplify(matched-(Be*Va-Ba*Ve)/(sp.sqrt(q)*N))==0
    a,A,Q,Aa,Qa,Ae,Qe=sp.symbols('a A Q Aa Qa Ae Qe')
    wedge=(a*Ae)*(2*a*Q+a*a*Qa)-(A+a*Aa)*(a*a*Qe)
    J=2*Ae*Q-A*Qe+a*(Ae*Qa-Aa*Qe)
    assert sp.expand(wedge-a*a*J)==0
    results={p:json.loads(Path('review12_checks/'+p+'.json').read_text()) for p in ('audit','interval_recheck','extra_checks','monte_carlo')}
    assert all(results[p]['status']=='PASS' for p in ('audit','interval_recheck','extra_checks'))
    recheck=results['interval_recheck']
    manifest=json.loads(Path('review12_checks/input_sha256.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    out=dict(symbolic_identities='PASS',reviewed_files_unchanged=True,original_rectangle_count=results['audit']['rectangle_count'],library_recomputed_original_rectangles=len(recheck['selected_original_indices']),library_recomputed_subrectangles=len(recheck['results']),library_recomputed_edge_intervals=len(recheck['edges']),resources_by_check={k:v['resources'] for k,v in results.items()},mc_summary=results['monte_carlo']['summaries'],resources=resources())
    with open('review12_checks/final_summary.json','x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
