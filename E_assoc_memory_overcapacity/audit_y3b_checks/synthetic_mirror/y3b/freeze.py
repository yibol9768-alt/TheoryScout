"""最终数据前冻结；排他创建主清单，不覆盖任何既有输入。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(60,70))
import ast
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import time
from common import ROOT,OUT,config,environment,entropy,jobs,dump,sha,usage

def main():
    manifest=ROOT/'Y3B_FROZEN.sha256'
    assert not manifest.exists() and not (OUT/'formal').exists()
    assert environment()==json.loads((OUT/'environment.json').read_text())
    cfg=config()
    assert (cfg['blocks'],cfg['query_default'],cfg['query_target'])==(32,512,8192)
    assert cfg['runner_cpu_hard']+cfg['analysis_cpu_hard']+10+cfg['preparation_cpu_reserve']==21600
    assert cfg['runner_cpu_soft']<cfg['runner_cpu_hard'] and cfg['confirm_family']==3
    tests=json.loads((OUT/'synthetic_v4_tests.json').read_text())
    assert tests['passed'] and tests['no_random_training_or_queries']
    sources=sorted(OUT.glob('*.py'))
    for p in sources: ast.parse(p.read_text(),filename=str(p))
    baseline=json.loads((OUT/'existing_baseline.json').read_text())
    changed=[rel for rel,h in baseline.items() if not (ROOT/rel).is_file() or sha(ROOT/rel)!=h]
    concurrent=json.loads((OUT/'concurrent_change.json').read_text())
    assert changed==[concurrent['path']],changed
    assert baseline[concurrent['path']]==concurrent['baseline_sha256']
    assert sha(ROOT/concurrent['path'])==concurrent['observed_sha256']
    inputs=json.loads((OUT/'input_hashes.json').read_text())
    assert concurrent['path'] not in inputs
    changed_inputs=[rel for rel,h in inputs.items() if sha(ROOT/rel)!=h]
    assert not changed_inputs,changed_inputs
    assert sha(OUT/'predictions.csv')==sha(ROOT/'y3_checks/Y3_frozen_predictions.csv')
    params=json.loads((OUT/'parameters.json').read_text())
    pop=json.loads((ROOT/'experiments/x1/X1_population.json').read_text())
    for p in params:
        original=next(v for v in pop['rows'] if v['ir']==p['ir'] and v['method']==p['method'])
        assert p['amplitude']==original['amplitude']
    streams=json.loads((OUT/'streams.json').read_text())
    expected=[dict(d=d,ir=ir,seed=s,stream=k,entropy=entropy(d,ir,s,k))
              for d,ir,s in jobs(cfg['blocks']) for k in range(4)]
    assert streams==expected
    seed=json.loads((OUT/'seed_audit.json').read_text())
    assert seed['planned_streams']==len(streams) and not seed['collisions']
    packages=sorted([dict(name=d.metadata['Name'],version=d.version) for d in importlib.metadata.distributions()],
                    key=lambda p:(p['name'].lower(),p['version']))
    dump(OUT/'environment_packages.json',packages)
    # 离线实际CPU（含合成OS限额测试子进程）+本次冻结+9秒仅校验启动上界+5秒收尾储备。
    prep=json.loads((OUT/'prepare_status.json').read_text())['end']['cpu_seconds']
    prior=prep+json.loads((OUT/'primary_power.json').read_text())['usage']['cpu_seconds']
    for name in ['synthetic_tests.json','analyze.py_tests.json','synthetic_v3_tests.json','synthetic_v4_tests.json']:
        s=json.loads((OUT/name).read_text())
        prior+=s['usage']['cpu_seconds']+s['cpu_seconds_children']
    freeze_cpu=usage()['cpu_seconds']
    prior+=concurrent['first_attempt_cpu_upper_seconds']
    assert prior+freeze_cpu+9+5<cfg['preparation_cpu_reserve']
    audit=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        new_formal_sampling=False,formal_directory_exists=False,
        existing_nonlog_files_checked=len(baseline),existing_nonlog_files_changed=changed,
        unrelated_concurrent_change=concurrent,
        old_input_hashes_checked=len(inputs),old_input_hashes_changed=changed_inputs,
        programs=[str(p.relative_to(ROOT)) for p in sources],synthetic_checks=len(tests['checks']),
        predictions_identical_to_Y3=True,population_amplitudes_identical=True,
        planned_streams=len(streams),old_entropy_count=seed['old_entropy_count'],rng_collisions=[],
        prior_cpu_seconds=prior,freeze_usage=usage(),
        preparation_cpu_with_verify_and_tail_upper=prior+freeze_cpu+14,
        cpu_reserve=cfg['preparation_cpu_reserve'],local_freeze_only=True,
        limitation='本地内容哈希及排他创建，不声称管理员不可更改的WORM或可信时间戳。')
    dump(ROOT/'Y3B_FREEZE_AUDIT.json',audit)
    paths=sorted([p for p in OUT.rglob('*') if p.is_file()]+[p for p in ROOT.glob('Y3B_*') if p.is_file()])
    assert all(not p.is_symlink() for p in paths)
    with manifest.open('x') as f:
        for p in paths: f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
        f.flush();os.fsync(f.fileno())
    os.chmod(manifest,0o444)
    for p in paths: os.chmod(p,0o444)
    print(json.dumps(dict(frozen_files=len(paths),manifest_sha256=sha(manifest),
        existing_checked=len(baseline),checks=len(tests['checks']),usage=usage()),ensure_ascii=False))

if __name__=='__main__': main()
