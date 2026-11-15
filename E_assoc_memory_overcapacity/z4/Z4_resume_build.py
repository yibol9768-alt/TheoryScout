"""续作构建器：只排他新建文件，保留中断时全部字节。无抽样。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU, (80, 90))
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'z4'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def write(p, s):
    with p.open('x') as f: f.write(s)

baseline = {str(p.relative_to(ROOT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
baseline.update({p.name: sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and not p.name.endswith('.log')})
write(OUT / 'resume_baseline.json', json.dumps(baseline, indent=2) + '\n')

s = (OUT / 'common.py').read_text()
s = s[:s.index('def verify():')] + '''def verify():
    manifest = ROOT / 'Z4_FROZEN.sha256'
    seen = set()
    for line in manifest.read_text().splitlines():
        h, rel = line.split('  ', 1)
        p = ROOT / rel
        if rel in seen or p.is_symlink() or not p.resolve().is_relative_to(ROOT): raise ValueError('manifest path')
        if not (rel.startswith('z4/') or rel.startswith('Z4_')): raise ValueError('manifest scope')
        if sha(p) != h: raise ValueError('freeze hash: ' + rel)
        seen.add(rel)
    required = {'Z4_PREREG.md','Z4_READY.md'} | {'z4/' + n for n in
        ['common_v2.py','run_v2.py','analyze_v2.py','protocol_v2.py','launch_v2.py','entry.py',
         'model.py','common.py','plan.py','freeze.py','test_resume.py','resume_tests.json',
         'config.json','environment.json','streams.json','seed_audit.json','power.json',
         'power_resume.json','predictions.json','synthetic_tests.json','input_hashes.json']}
    if not required <= seen: raise ValueError('incomplete manifest')
    if environment() != read(OUT / 'environment.json'): raise ValueError('environment mismatch')
    return sha(manifest)
'''
write(OUT / 'common_v2.py', s)

s = (OUT / 'run.py').read_text().replace('from common import ', 'from common_v2 import ').replace('from protocol import ', 'from protocol_v2 import ')
write(OUT / 'run_v2.py', s)

s = (OUT / 'analyze.py').read_text().replace('from common import ', 'from common_v2 import ').replace('from run import ', 'from run_v2 import ').replace('from protocol import ', 'from protocol_v2 import ')
s = s.replace("csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['empty'])", "csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['empty'],lineterminator='\\n')")
s = s.replace("cc=interval(vals[:,task],24,.05)", "cc=interval(vals[:,task],48,.05)")
s = s.replace("direction=gate and ci['lower'] is not None and ci['lower']>0", "direction=gate and finite is not None and ((finite>0 and ci['lower'] is not None and ci['lower']>0) or (finite<0 and ci['upper'] is not None and ci['upper']<0))")
s = s.replace("correction_direction=correction,**ci)", "correction_direction=correction,**ci)")
# Keep every dataset, including unpaired readable data. Predictions condition on the locked alpha.
s = s.replace("matching=[];curves=[];old_pairs=[];scaled=[];gates={}", """completed=[]
    for (d,seed),ds in sorted(datasets.items()):
        for a,arm in enumerate(ARMS):
            for ti,target in enumerate(cfg['targets']):
                for mi,method in enumerate(METHODS):
                    if ds['valid'][a,ti,mi]:
                        completed.append(dict(d=d,seed=seed,arm=arm,target=target,method=method,
                            alpha=float(ds['alphas'][a,ti,mi]),old=float(ds['recall'][a,ti,mi,0]),
                            new=float(ds['recall'][a,ti,mi,1])))
    matching=[];curves=[];old_pairs=[];scaled=[];gates={} """)
s = s.replace("availability=available,curves=curves,matching=matching,old_pairs=old_pairs,", "all_completed=completed,availability=available,curves=curves,matching=matching,old_pairs=old_pairs,")
s = s.replace("('availability.csv','availability')", "('availability.csv','availability'),('all_completed.csv','all_completed')")
s = s.replace("credential=cred,files=files,confirmation_valid=False", "credential=cred,files=files,usage=__import__('common_v2').usage(),confirmation_valid=False")
write(OUT / 'analyze_v2.py', s)

s = (OUT / 'protocol.py').read_text().replace('from common import ', 'from common_v2 import ').replace('from analyze import ', 'from analyze_v2 import ')
s = s.replace("'scaled_residuals.csv','availability.csv'", "'scaled_residuals.csv','availability.csv','all_completed.csv'")
s = s.replace("('availability.csv','availability')", "('availability.csv','availability'),('all_completed.csv','all_completed')")
s = s.replace("if launch['manifest']!=manifest or start['manifest']!=manifest:", "if launch['config']!=cfg or launch['manifest']!=manifest or start['manifest']!=manifest:")
s = s.replace("        for name,h in pending['files'].items():", """        if pending['confirmation_valid'] is not False: errors.append('PENDING cannot confirm')
        if not 0 <= pending['usage']['cpu'] <= cfg['analysis_cpu_hard']: errors.append('analysis CPU')
        for name,h in pending['files'].items():""")
s = s.replace("def verify_accepted(data,manifest,cfg):", "def verify_accepted(data,manifest,cfg):")
s = s.replace("    accepted=read(data/'ACCEPTED.json');status=read(data/'launch_status.json')", """    observed=read(data/'EXIT_OBSERVED.json')
    if observed['returncode']!=0 or observed['killed'] is not False or observed['manifest']!=manifest: raise ValueError('supervisor did not exit normally')
    if observed['launch_status_sha256']!=sha(data/'launch_status.json'): raise ValueError('observer status hash')
    receipt=read(data/'SUPERVISOR_EXIT.json')
    if observed['receipt_sha256']!=sha(data/'SUPERVISOR_EXIT.json') or receipt['normal'] is not True or receipt['manifest']!=manifest: raise ValueError('exit receipt')
    if receipt['launch_status_sha256']!=sha(data/'launch_status.json'): raise ValueError('receipt status hash')
    if not 0 <= observed['wall'] <= cfg['total_wall_hard'] or not 0 <= observed['cpu'] <= cfg['total_cpu_hard']-cfg['preparation_cpu_reserve']: raise ValueError('observer total budget')
    if not 0 <= observed['self_cpu'] < 10: raise ValueError('observer CPU')
    accepted=read(data/'ACCEPTED.json');status=read(data/'launch_status.json')""")
# Require all reported resources to be nonnegative finite numbers, including tampered NaNs.
s = s.replace("    status=dict(manifest=manifest,stages=stages", """    if any(type(v) not in (int,float) or not 0 <= v < float('inf') for v in [wall,children_cpu,self_cpu]):
        errors.append('invalid budget number')
    status=dict(manifest=manifest,stages=stages""")
write(OUT / 'protocol_v2.py', s)

s = (OUT / 'launch.py').read_text().replace('from common import ', 'from common_v2 import ').replace('from protocol import ', 'from protocol_v2 import ')
s = s.replace("signal.setitimer(signal.ITIMER_PROF,99.)", "signal.setitimer(signal.ITIMER_PROF,89.)")
s = s.replace("('runner','run.py',", "('runner','run_v2.py',").replace("('analysis','analyze.py',", "('analysis','analyze_v2.py',")
s = s.replace("    start=time.monotonic()", "    if os.environ.get('Z4_OBSERVER_PID')!=str(os.getppid()): raise ValueError('observer required')\n    start=time.monotonic()")
write(OUT / 'launch_v2.py', s)

s = (OUT / 'test_synthetic.py').read_text().replace('from common import ', 'from common_v2 import ').replace('from protocol import ', 'from protocol_v2 import ').replace('from analyze import ', 'from analyze_v2 import ').replace('from run import ', 'from run_v2 import ')
s = s.replace("BASE=OUT/'synthetic'", "BASE=OUT/'synthetic_resume'")
s = s.replace("credential=creds['analysis'],files=files,confirmation_valid=False", "credential=creds['analysis'],files=files,usage=usage(),confirmation_valid=False")
s = s.replace("('availability.csv','availability')", "('availability.csv','availability'),('all_completed.csv','all_completed')")
s = s.replace("    require(verify_accepted(data,mh,cfg)['confirmation_valid'])", "    observer_receipt(data,mh)\n    require(verify_accepted(data,mh,cfg)['confirmation_valid'])")
s = s.replace("    with (data/'analysis/old_pairs.csv').open('a')", "    observer_receipt(data,mh)\n    with (data/'analysis/old_pairs.csv').open('a')")
s = s.replace("dump(OUT/'synthetic_tests.json',report)", "dump(OUT/'resume_tests.json',report)")
s = s.replace("    check('kernel_soft_CPU_hard_CPU_wall_kill',kernel_limits)", "    extra_checks()\n    check('kernel_soft_CPU_hard_CPU_wall_kill',kernel_limits)")
s = s.replace("if __name__=='__main__': main()", "from resume_extra import observer_receipt, extra_checks\nif __name__=='__main__': main()")
# Extra checks receive the running module, also when executed as a script.
s = s.replace("    extra_checks()", "    extra_checks(sys.modules[__name__])")
write(OUT / 'test_resume.py', s)

write(OUT / 'resume_build.json', json.dumps(dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    previous_synthetic_passed=False, previous_failed=['supervisor_recomputes_full_fixture_and_accepts','accepted_evidence_mutation_rejected'],
    baseline_files=len(baseline), scientific_design_unchanged=True, draws=0,
    usage_cpu=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime), indent=2)+'\n')
print('New resume modules created; existing files preserved; no draws.')
