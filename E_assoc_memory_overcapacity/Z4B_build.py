"""一次性建立修订包；仅写 z4b，禁止科学抽样。"""
import hashlib
import json
from pathlib import Path
import resource
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'z4b'
assert not OUT.exists() or not list(OUT.iterdir())
OUT.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
protected=list((ROOT/'z4').rglob('*'))+list(ROOT.glob('Z4_*'))+[ROOT/'AUDIT_Z4.md',ROOT/'audit_z4_checks/repair_refined.json',ROOT/'audit_z4_checks/complete_power_scenario.json']
baseline={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file() and not p.is_symlink()}
save('baseline.json',baseline)
names=['common_v2.py','model.py','run_v2.py','analyze_v2.py','protocol_v2.py','launch_v2.py','entry.py','test_resume.py','test_synthetic.py','test_credentials.py','test_final.py','resume_extra.py']
for name in names:
    s=(ROOT/'z4'/name).read_text()
    s=s.replace("'z4'","'z4b'").replace('z4/','z4b/').replace('Z4_','Z4B_')
    s=s.replace('all18_directions','primary_all3_directions').replace('all18_finite_agreement','primary_all3_finite_agreement').replace('all_empirical_matching','primary_all7_matching')
    if name=='model.py': s=s.replace('from common import','from common_v2 import')
    if name=='common_v2.py':
        s=s.replace('[20261106, 4097162831, 4,','[20261106, 3187362049, 45,')
        s=s[:s.index('    required =')]+'''    required = {'Z4B_PREREG.md','Z4B_READY.md'} | {'z4b/' + n for n in
        ['common_v2.py','run_v2.py','analyze_v2.py','protocol_v2.py','launch_v2.py','entry.py',
         'model.py','config.json','environment.json','streams.json','seed_audit.json',
         'power.json','resources.json','port_synthetic_tests.json','resume_tests.json',
         'credential_tests.json','final_tests.json','revision_tests.json','volume_result.json',
         'input_hashes.json','baseline_check.json','freeze.py']}
    if not required <= seen: raise ValueError('incomplete manifest')
    if environment() != read(OUT / 'environment.json'): raise ValueError('environment mismatch')
    for rel,h in read(OUT/'input_hashes.json').items():
        if sha(ROOT/rel)!=h: raise ValueError('input hash: '+rel)
    if not read(OUT/'power.json')['design_qualified']: raise ValueError('power design failed')
    return sha(manifest)
'''
        s+='''
def scheduled(cfg, seed, target):
    return target == cfg['primary_target'] or seed < cfg['side_n_eval']

def expected_n(cfg, target):
    return cfg['n_eval'] if target == cfg['primary_target'] else min(cfg['side_n_eval'],cfg['n_eval'])
'''
    if name=='run_v2.py':
        s=s.replace('commit, usage','commit, usage, scheduled')
        s=s.replace("            for t,target in enumerate(cfg['targets']):\n                cell=", "            for t,target in enumerate(cfg['targets']):\n                if not scheduled(cfg,s,target): continue\n                cell=")
        s=s.replace('return dict(ids=ids,alphas=alphas,valid=valid,ranks=ranks,',"return dict(ids=ids,alphas=alphas,valid=valid,ranks=ranks,\n        scheduled=np.array([scheduled(cfg,s,t) for t in cfg['targets']],dtype=bool),")
    if name=='analyze_v2.py':
        s=s.replace('plan, jobname','plan, jobname, scheduled, expected_n')
        s=s.replace("{'ids','ranks','metadata','alphas','valid'}","{'ids','ranks','metadata','alphas','valid','scheduled'}")
        s=s.replace("        for a,arm in enumerate(ARMS):", "        mask=np.array([scheduled(cfg,job[2],t) for t in cfg['targets']],dtype=bool)\n        if arrays['scheduled'].dtype!=bool or not np.array_equal(arrays['scheduled'],mask): raise ValueError('schedule mask')\n        for a,arm in enumerate(ARMS):",1)
        s=s.replace('idx=(a,ti,mi);ok=v is not None',"idx=(a,ti,mi);ok=v is not None and bool(mask[ti]);v=v if ok else None")
        s=s.replace("valid=arr['valid'],alphas=arr['alphas'])","valid=arr['valid'],alphas=arr['alphas'],scheduled=arr['scheduled'])")
        start=s.index("                    if ds['valid'][a,ti,mi]:")
        end=s.index('    matching=[]',start)
        s=s[:start]+'''                    ok=bool(ds['valid'][a,ti,mi])
                    completed.append(dict(d=d,seed=seed,arm=arm,target=target,method=method,
                        scheduled=bool(ds['scheduled'][ti]),valid=ok,
                        alpha=float(ds['alphas'][a,ti,mi]) if ok else None,
                        old=float(ds['recall'][a,ti,mi,0]) if ok else None,
                        new=float(ds['recall'][a,ti,mi,1]) if ok else None))
'''+s[end:]
        s=s.replace("            for ti,target in enumerate(cfg['targets']):\n                abs_gates={}","            for ti,target in enumerate(cfg['targets']):\n                primary=arm=='empirical' and d==128 and target==cfg['primary_target']\n                N=expected_n(cfg,target)\n                mf=cfg['match_family'] if primary else 21\n                of=cfg['old_family'] if primary else 9\n                role='primary' if primary else 'descriptive'\n                abs_gates={}")
        s=s.replace("interval(vals[:,1]-target,cfg['match_family'])","interval(vals[:,1]-target,mf)")
        s=s.replace("len(vals)==cfg['n_eval']","len(vals)==N").replace("len(old)==cfg['n_eval']","len(old)==N")
        s=s.replace("interval(new,cfg['match_family'])","interval(new,mf)").replace("interval(old,cfg['old_family'])","interval(old,of)")
        s=s.replace('interval(vals[:,task],48,.05)','interval(vals[:,task],24,.05)')
        s=s.replace('dict(arm=arm,d=d,target=target,', 'dict(arm=arm,d=d,target=target,role=role,expected_n=N,')
        start=s.index("    emp=[r for r in old_pairs")
        end=s.index('    return dict(manifest=',start)
        s=s[:start]+'''    emp=[r for r in old_pairs if r['role']=='primary']
    matches=[r for r in matching if r['role']=='primary']
    plan_complete=not data_errors and all(r['state']=='complete' for r in available)
    decisions=dict(primary_plan_complete=plan_complete,primary_pair_count=len(emp),
        primary_match_count=len(matches),
        primary_all7_matching=plan_complete and len(matches)==7 and all(r['matched'] for r in matches),
        primary_all3_directions=plan_complete and len(emp)==3 and all(r['direction_supported'] for r in emp),
        primary_all3_finite_agreement=plan_complete and len(emp)==3 and all(r['finite_agreement'] for r in emp),
        coefficient_confirmation=False,descriptive_confirmation=False)
'''+s[end:]
    if name=='entry.py': s=s.replace('ITIMER_PROF, 9.','ITIMER_PROF, 39.')
    if name=='launch_v2.py': s=s.replace('ITIMER_PROF,89.','ITIMER_PROF,59.')
    if name=='protocol_v2.py': s=s.replace("observed['self_cpu'] < 10","observed['self_cpu'] < cfg['observer_cpu_hard']")
    if name in ['test_resume.py','test_synthetic.py']:
        s=s.replace('from common import','from common_v2 import').replace('from analyze import','from analyze_v2 import').replace('from run import','from run_v2 import').replace('from protocol import','from protocol_v2 import')
        s=s.replace('resource.RLIMIT_CPU,(100,110)','resource.RLIMIT_CPU,(180,190)')
        s=s.replace("                        v=c['empirical']['alpha'] if a==0 else c['analytic']", "                        v=(c['empirical']['alpha'] if a==0 else c['analytic']) if scheduled(cfg,s,target) else None")
        s=s.replace('usage, config, environment','usage, config, environment, scheduled')
        s=s.replace('valid=valid,\n                metadata=',"valid=valid,\n                scheduled=np.array([scheduled(cfg,s,t) for t in cfg['targets']],dtype=bool),\n                metadata=")
        if name=='test_synthetic.py':
            s=s.replace("('availability.csv','availability')]:","('availability.csv','availability'),('all_completed.csv','all_completed')]:")
            s=s.replace("files=files,confirmation_valid=False","files=files,usage=usage(),confirmation_valid=False")
            s=s.replace("    require(verify_accepted(data,mh,cfg)['confirmation_valid'])","    observer_receipt(data,mh)\n    require(verify_accepted(data,mh,cfg)['confirmation_valid'])",1)
            s=s.replace("    report=dict(synthetic_only=True", "    report=dict(synthetic_only=True")
            s=s.replace("OUT/'synthetic_tests.json'","OUT/'port_synthetic_tests.json'")
            s=s.replace("if __name__=='__main__': main()","from resume_extra import observer_receipt\nif __name__=='__main__': main()")
    if name=='test_final.py': s=s.replace('family==18','family==3')
    (OUT/name).write_text(s)
cfg=json.loads((ROOT/'z4/config.json').read_text())
cfg.update(dims=[128],n_cal=1024,n_eval=1280,primary_target=.425,side_n_eval=64,
           match_family=7,old_family=3,agreement_family=3,observer_cpu_hard=40,supervisor_cpu_hard=60,
           supervisor_observer_cpu_reserve=100,independent_calibration_check=False)
save('config.json',cfg)
save('build_receipt.json',dict(copied_sources={n:sha(ROOT/'z4'/n) for n in names},no_sampling=True,
                             usage_cpu=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime))
print(dict(created=str(OUT),baseline_files=len(baseline)))
