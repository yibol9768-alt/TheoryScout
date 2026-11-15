"""Offline quadrature, power, cost, visible seed audit. NEVER creates a Generator."""
import ast
import itertools
import math
import subprocess
import time
import numpy as np
from scipy.stats import nct, norm, t
from common import ROOT, OUT, METHODS, PAIRS, ARMS, dump, sha, environment, entropy, usage
from model import analytic_alpha, predict

def main():
    started = usage()
    inputs = {}
    def source(rel):
        p = ROOT / rel; inputs[rel] = sha(p); return p
    for rel in ['ICML_REVIEW_SIM5.md','Z2B_BR_PROOF.md','PROOF_REVIEW18.md',
                'Y3D_PREREG.md','y3d/common.py','y3d/launch.py','y3d/integrity.py','y3d/publication.py',
                'review13_checks/kernel_budget.json']:
        source(rel)
    cfg = dict(dims=[128,256], targets=[.35,.425,.50], n_cal=128, n_eval=224,
               q_cal=512, q_eval=1024, grid_count=17, r=0., kappa=.25, rho=.1,
               nonmonotonic_limit=.015, match_band=.01, agreement_band=.015,
               match_family=42, old_family=18, agreement_family=18,
               preparation_cpu_reserve=600, runner_cpu_soft=20200, runner_cpu_hard=20400,
               analysis_cpu_hard=500, supervisor_cpu_hard=100, total_cpu_hard=21600,
               runner_wall_soft=11900, runner_wall_hard=12000, analysis_wall_hard=550,
               total_wall_hard=12600, planned_only=True)
    predictions = []
    for d, target, method in itertools.product(cfg['dims'],cfg['targets'],METHODS):
        aa = analytic_alpha(method,d,target,True); ai = analytic_alpha(method,d,target,False)
        if aa is None or ai is None: raise ValueError('analytic target unbracketed')
        predictions.append(dict(d=d,target=target,method=method,analytic_alpha=aa,limit_alpha=ai,
            analytic_old=predict(method,aa,d,'old'),limit_old=predict(method,ai,d,'old')))
    # No old endpoint data used: explicit conservative planning assumptions.
    # Dataset-level SD includes finite Q and shared-data dependence; it is NOT sqrt(p(1-p)/NQ).
    sd_new=.025; sd_pair=.030; sd_cal=.030
    rows=[]
    critical=float(t.ppf(1-.025/(2*18),cfg['n_eval']-1))
    matchcrit=float(t.ppf(1-.025/(2*42),cfg['n_eval']-1))
    for d,target,(a,b) in itertools.product(cfg['dims'],cfg['targets'],PAIRS):
        pa=next(p for p in predictions if (p['d'],p['target'],p['method'])==(d,target,a))
        pb=next(p for p in predictions if (p['d'],p['target'],p['method'])==(d,target,b))
        effect=pa['analytic_old']['finite']-pb['analytic_old']['finite']
        shift=(pa['analytic_old']['B']-pb['analytic_old']['B'])/d
        nc=effect*math.sqrt(cfg['n_eval'])/sd_pair
        power=float(nct.sf(critical,cfg['n_eval']-1,nc))
        # Approximate two-sided equivalence, with t cutoff and normal planning error.
        half=critical*sd_pair/math.sqrt(cfg['n_eval'])
        agree=float(max(0,2*norm.cdf((cfg['agreement_band']-half)*math.sqrt(cfg['n_eval'])/sd_pair)-1))
        coeff=float(nct.sf(critical,cfg['n_eval']-1,abs(shift)*math.sqrt(cfg['n_eval'])/sd_pair))
        rows.append(dict(d=d,target=target,pair=a+'-'+b,effect=effect,finite_shift=shift,
                         direction_power=power,agreement_power_approx=agree,
                         correction_direction_power=coeff))
    mh=matchcrit*sd_new/math.sqrt(cfg['n_eval'])
    joint_sd=math.sqrt(sd_cal**2/cfg['n_cal']+sd_new**2/cfg['n_eval'])
    match_power=float(max(0,2*norm.cdf((cfg['match_band']-mh)/joint_sd)-1))
    # Conservative charge every alpha as a separate two-projection pass despite shared affine GEMMs.
    import json
    kernel=json.loads(source('review13_checks/kernel_budget.json').read_text())['rows']
    costs=[]
    for row in kernel:
        if row['d'] not in cfg['dims']: continue
        train=row['training_3_GEMM_cpu']; scan=row['candidate_10passes_cpu']/10
        gen=.035*row['d']*row['n']/1e6
        cal=cfg['n_cal']*4*(2*(train+scan*cfg['grid_count']*2*cfg['q_cal']/512)+gen+1)
        ev=cfg['n_eval']*(2*(train+scan*2*3*4*2*2*cfg['q_eval']/512)+gen+1)
        costs.append(dict(d=row['d'],cal_cpu=cal,eval_cpu=ev))
    projected=sum(x['cal_cpu']+x['eval_cpu'] for x in costs)+500
    if projected>cfg['runner_cpu_soft']: raise ValueError('cost design not feasible')
    power=dict(assumptions=dict(sd_new=sd_new,sd_old_pair=sd_pair,sd_cal=sd_cal,
        no_new_pilot=True,variance_inflation='SD .025/.030 are assumptions, not data-derived bounds'),
        rows=rows,match_gate_marginal_power_approx=match_power,
        all42_gate_power_union_lower=max(0,1-42*(1-match_power)),
        direction_family_power_union_lower=max(0,1-sum(1-r['direction_power'] for r in rows)),
        min_direction_power=min(r['direction_power'] for r in rows),
        coefficient_confirmatory=False,reason='small 1/d shifts underpowered; no coefficient acceptance claim',
        cost_rows=costs,projected_cpu=projected,projected_wall_at_1_cpu_per_second=projected,
        wall_comment='wall forecast uses 1.8 CPU/wall below; hard timeout overrides forecast',
        projected_wall=projected/1.8,critical_old=critical,critical_match=matchcrit)
    if power['min_direction_power'] < .8: raise ValueError('direction power below prereg threshold')
    # Complete new entropy plan. Same evaluation stream reused across all methods/arms/targets.
    streams=[]
    for d,m,s,k in itertools.product(cfg['dims'],range(4),range(cfg['n_cal']),range(4)):
        streams.append(entropy(0,d,m,s,k))
    for d,s,k in itertools.product(cfg['dims'],range(cfg['n_eval']),range(4)):
        streams.append(entropy(1,d,-1,s,k))
    # Scan only this project, skip symlinks; include JSON entropy records + AST literal initializers.
    cached=OUT/'planning_snapshot'
    files=[] if cached.exists() else subprocess.check_output(['rg','--files','--hidden','-g','!.git','-g','!z4/**','-g','!Z4_*'],cwd=ROOT,text=True).splitlines()
    old=set(); inventory=[]; unresolved=[]
    if cached.exists():
        import json
        old={tuple(e) for e in json.loads(source('z4/planning_snapshot/old_entropies.json').read_text())}
        inventory=json.loads(source('z4/planning_snapshot/rng_inventory.json').read_text())
        for rel in sorted({r['file'] for r in inventory}): source(rel)
        # Complete the initial size-limited scan by streaming all oversized JSONL files.
        skipped=json.loads(source('z4/planning_snapshot/seed_audit.json').read_text())['unresolved']
    else: skipped=[]
    def add(v):
        if type(v) is int and v>=0: old.add((v,))
        elif isinstance(v,(list,tuple)) and v and all(type(x) is int and x>=0 for x in v): old.add(tuple(v))
    def walk(o):
        if isinstance(o,dict):
            for k,v in o.items():
                if 'entropy' in k.lower():
                    add(v)
                    if isinstance(v,list) and all(type(x) is int for x in v):
                        for j in range(6): add(v+[j])
                walk(v)
        elif isinstance(o,list):
            for x in o:
                if isinstance(x,(dict,list)): walk(x)
    for row in skipped:
        p=ROOT/row['file']
        if p.suffix!='.jsonl': raise ValueError('unresolved seed input')
        with p.open() as f:
            for line in f:
                if 'entropy' in line: walk(json.loads(line))
        source(row['file'])
    for rel in files:
        p=ROOT/rel
        if p.is_symlink() or p.suffix not in ['.py','.json','.jsonl']: continue
        if p.stat().st_size>25000000:
            unresolved.append(dict(file=rel,reason='over 25MB')); continue
        txt=p.read_text(errors='replace')
        if p.suffix in ['.json','.jsonl'] and 'entropy' in txt:
            source(rel)
            try:
                if p.suffix=='.json': walk(json.loads(txt))
                else:
                    for line in txt.splitlines():
                        if 'entropy' in line: walk(json.loads(line))
            except (ValueError,TypeError): unresolved.append(dict(file=rel,reason='malformed record'))
        if p.suffix=='.py':
            for i,line in enumerate(txt.splitlines(),1):
                if any(k in line for k in ['SeedSequence','default_rng','RandomState','random.seed','manual_seed','Sobol(']):
                    inventory.append(dict(file=rel,line=i,text=line.strip())); source(rel)
            try: tree=ast.parse(txt)
            except SyntaxError: unresolved.append(dict(file=rel,reason='syntax')); continue
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['SeedSequence','default_rng','RandomState'] and node.args:
                    try: add(ast.literal_eval(node.args[0]))
                    except (ValueError,TypeError): pass
    # Offline copy of established conservative dynamic domains (never import old code).
    for ph,d,ir,s,k in itertools.product([11,22,33],[16,128,256,512],range(3),range(192),range(4)):
        add([20261106,913752,ph,d,ir,s,k])
    for prefix in [[20261106,30917401,71],[20261106,87410329,83]]:
        for d,ir,s,k in itertools.product([128,256,512],range(3),range(64),range(4)): add(prefix+[d,ir,s,k])
    # Reuse exactly the conservative family definitions as inert syntax, not executable imports.
    tree=ast.parse(source('y3b/prepare.py').read_text())
    fam=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='families' for t in n.targets))
    safe={'itertools':itertools,'range':range}
    families=eval(compile(ast.Expression(fam),'<frozen conservative domains>','eval'),{'__builtins__':{},**safe})
    for seqs in families.values():
        for ent in seqs: add(ent)
    def state(e): return tuple(map(int,np.random.SeedSequence(e).generate_state(8)))
    new={state(e):e for e in streams}
    if len(new)!=len(streams): raise ValueError('internal seed collision')
    collisions=[dict(old=list(e),new=new[state(e)]) for e in old if state(e) in new]
    if collisions: raise ValueError('old seed collision')
    # A new namespace literal must not occur in any old initializer/record.
    namespace_hits=[r for r in inventory if '4097162831' in r['text']]
    if namespace_hits: raise ValueError('namespace reused')
    dump(OUT/'config.json',cfg); dump(OUT/'predictions.json',predictions); dump(OUT/'power.json',power)
    dump(OUT/'streams.json',streams); dump(OUT/'rng_inventory.json',inventory)
    dump(OUT/'old_entropies.json',[list(e) for e in sorted(old)])
    dump(OUT/'seed_audit.json',dict(planned=len(streams),old_unique=len(old),collisions=collisions,
        state_bits=256,namespace_hits=namespace_hits,unresolved=unresolved,
        limitation='Visible recorded/static/conservative domains only; unrecorded executions cannot be excluded; state separation is not mathematical independence.',
        training_or_query_draws=0))
    dump(OUT/'environment.json',environment()); dump(OUT/'input_hashes.json',inputs)
    dump(OUT/'planning_usage.json',dict(start=started,end=usage(),formal_sampling=False))
    print(json.dumps(dict(projected_cpu=projected,projected_wall=projected/1.8,
        min_direction_power=power['min_direction_power'],match_power=match_power,streams=len(streams),old=len(old))))

if __name__=='__main__': main()
