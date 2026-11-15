"""Artifact audit and exact finite-support check of the new non-Gaussian row formula."""
import hashlib, itertools, json, os, ast, re
from fractions import Fraction as F
from pathlib import Path

P=Path(__file__).resolve().parent
ROOT=P.parent
expected={
    'THEOREM_M.md':'c3ba84568df754f0f5428940a6827cc97fe38f7b20f61546e5d212f17efd91ea',
    'ICML_REVIEW_SIM2.md':'e9ec4a694aed37485d683c129b899b5ae971154ff26239b58b22613c22b47307'}
for name,value in expected.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,name
for name in ['run.py','report.py','focus.py','audit.py']:
    ast.parse((P/name).read_text(),filename=name)
doc=(ROOT/'W9_EXTENSION.md').read_text()
assert all(doc.count('\\tag{W9.'+str(i)+'}')==1 for i in range(1,27))
assert doc.count('\\[')==doc.count('\\]')
# A TeX matrix row separator immediately followed by "(" is not inline math.
assert len(re.findall(r'(?<!\\)\\\(',doc))==len(re.findall(r'(?<!\\)\\\)',doc))
rows=[json.loads(s) for s in (P/'formal_rows.jsonl').read_text().splitlines()]
focus=[json.loads(s) for s in (P/'focus_rows.jsonl').read_text().splitlines()]
assert len(rows)==600 and len(focus)==72
assert len({(r['case'],r['d'],r['seed'],r['method']) for r in rows})==600
assert len({(r['case'],r['seed']) for r in focus})==72
checks=json.loads((P/'closed_form_checks.json').read_text())
assert len(checks)==9 and max(r['error'] for r in checks)<3e-5
assert json.loads((P/'formal_metadata.json').read_text())['seconds']+json.loads((P/'focus_summary.json').read_text())['metadata']['seconds']<1200
assert os.environ['OPENBLAS_NUM_THREADS']=='2'

coeff=[F(1,8),F(-1,7),F(2,5),F(1,3),F(-1,9)]
certificates=[]
for name,support,prob,product_scale,kurtosis in [
    ('rademacher',[-1,1],[F(1,2),F(1,2)],F(1),F(-2)),
    # epsilon=zeta/sqrt(2); epsilon_j sum t epsilon = zeta_j sum t zeta /2.
    ('asymmetric_two_point',[-1,2],[F(2,3),F(1,3)],F(1,2),F(-3,2))]:
    means=[F(0)]*3
    second=[[F(0) for _ in range(3)] for _ in range(3)]
    for indices in itertools.product(range(2),repeat=len(coeff)):
        weight=F(1)
        vals=[]
        for i in indices:
            weight*=prob[i];vals.append(F(support[i]))
        f=sum(a*b for a,b in zip(coeff,vals))
        products=[product_scale*vals[j]*f for j in range(3)]
        for j in range(3):
            means[j]+=weight*products[j]
            for k in range(3):
                second[j][k]+=weight*products[j]*products[k]
    assert means==coeff[:3]
    norm=sum(v*v for v in coeff)
    for j in range(3):
        for k in range(3):
            obs=second[j][k]-means[j]*means[k]
            target=(norm if j==k else 0)+coeff[j]*coeff[k]+(kurtosis*coeff[j]**2 if j==k else 0)
            assert obs==target,(name,j,k,obs,target)
    certificates.append(dict(distribution=name,states=2**len(coeff),exact_rational_covariance_identity=True))
out=dict(inputs_unchanged=expected,formal_rows=len(rows),focus_rows=len(focus),
         formula_W9_26=certificates,closed_form_checks=9,
         note='Numerical/artifact audit; not a formal proof verification.',
         source_hashes={name:hashlib.sha256((P/name).read_bytes()).hexdigest()
                        for name in ['run.py','report.py','focus.py','audit.py']})
(P/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
