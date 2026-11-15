"""逐字节/逐AST核对科学实现、统计规范、计划、预算及冻结条目。"""
import ast, difflib, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit_y3c_checks'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False)
identical={}
for name in ['config.json','parameters.json','predictions.csv','streams.json','seed_audit.json','rng_inventory.json',
             'power_summary.json','primary_power.json','planning_covariances.json','power_grid.csv','environment.json',
             'environment_packages.json','input_hashes.json','predict.py']:
    a=ROOT/'y3b'/name;b=ROOT/'y3c'/name
    identical[name]={'byte_identical':a.read_bytes()==b.read_bytes(),'y3b_sha256':sha(a),'y3c_sha256':sha(b)}
identical['PREREG']={'byte_identical':(ROOT/'Y3B_PREREG.md').read_bytes()==(ROOT/'Y3C_PREREG.md').read_bytes()}
assert all(v['byte_identical'] for v in identical.values())
asts={};diffs=[]
expected={'run.py':{'main'},'analyze.py':{'main'},'common.py':{'verify'},'predict.py':set()}
constants={}
for name,allowed in expected.items():
    a=(ROOT/'y3b'/name).read_text();b=(ROOT/'y3c'/name).read_text()
    diffs.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='y3b/'+name,tofile='y3c/'+name))
    def functions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    aa=functions(a);bb=functions(b)
    changed=sorted(k for k in set(aa)|set(bb) if aa.get(k)!=bb.get(k))
    assert set(changed)==allowed,(name,changed)
    asts[name]={'identical_functions':sorted(k for k in aa if k in bb and aa[k]==bb[k]),'changed_functions':changed}
    def assignments(s):
        return {ast.unparse(n.targets[0]):ast.dump(n.value,include_attributes=False)
                for n in ast.parse(s).body if isinstance(n,ast.Assign) and len(n.targets)==1}
    ca=assignments(a);cb=assignments(b)
    changed_constants=sorted(k for k in set(ca)|set(cb) if ca.get(k)!=cb.get(k))
    constants[name]=changed_constants
    assert changed_constants==(['OUT'] if name=='common.py' else [])
with (OUT/'scientific_sources.diff').open('x') as f:f.writelines(diffs)
manifest=[]
for line in (ROOT/'Y3C_FROZEN.sha256').read_text().splitlines():
    h,rel=line.split('  ',1);p=ROOT/rel
    assert p.resolve().is_relative_to(ROOT)
    manifest.append({'path':rel,'matches':sha(p)==h})
assert all(v['matches'] for v in manifest)
sys.path.insert(0,str(ROOT/'y3c'))
import common
cfg=common.config();plan=list(common.jobs(cfg['blocks']))
assert len(plan)==288 and len(set(plan))==288
streams=json.loads((ROOT/'y3c/streams.json').read_text())
dump(OUT/'design_check.json',dict(byte_comparisons=identical,functions=asts,changed_top_level_assignments=constants,
    manifest_items=len(manifest),manifest=manifest,config=cfg,plan=[list(k) for k in plan],
    no_formal_y3c=not (ROOT/'y3c/formal').exists(),no_formal_y3b=not (ROOT/'y3b/formal').exists()))
print(json.dumps(dict(byte_identical_files=len(identical),ast_files=len(asts),frozen_entries=len(manifest),plan_cells=len(plan))))
