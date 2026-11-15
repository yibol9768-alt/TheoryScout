"""只读复核；--freeze仅排他新建补充散列清单，绝不回写旧文件。"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import argparse, ast, csv, hashlib, json, math
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def manifest_check(p):
    n=0
    for line in p.read_text().splitlines():
        h,name=line.split('  ',1);f=(R/name).resolve()
        assert f.is_relative_to(R) and sha(f)==h,name;n+=1
    return n

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--check-only',action='store_true');z=ap.parse_args()
    original=manifest_check(R/'HOLDOUT_D_PREREG.sha256')
    a=json.loads((R/'HOLDOUT_D_SEED_AUDIT.json').read_text())
    assert all(sha(R/p)==h for p,h in a['source_hashes'].items())
    r=json.loads((R/'HOLDOUT_D_AMEND1_RESULTS.json').read_text())
    d=json.loads((R/'HOLDOUT_D_DESIGN.json').read_text())
    assert r['N']==d['N']==144 and r['successful_cap']==9000 and r['attempt_cap']==12000
    assert d['comparison']['cutoff']['critical']==.541 and len(r['scenarios'])==6
    assert all(sha(R/p)==h for p,h in r['inputs'].items())
    sums=[[0]*5 for _ in range(2)]
    for counts in r['batch_counts'].values():
        for j in range(2):
            n,death,e,m,s=counts[j]
            assert math.isclose((e/n)*(m/e)*(1-s/m),death/n)
            for k,v in enumerate(counts[j]):sums[j][k]+=v
    assert sums==[[1463,34,333,72,38],[1339,14,289,56,42]]
    for x in r['scenarios'].values():
        n=x['simulations'];assert n==10000
        for k,count in x['counts'].items():assert math.isclose(x['rates'][k],count/n)
        assert x['counts']['budget_complete']==n-x['exhaustions']
        assert x['rates']['joint']<=min(x['rates']['primary'],x['rates']['calibration'])
        assert x['rates']['all_gates_normal']<=min(x['rates']['joint'],x['rates']['mechanisms_normal'])
        b=x['actual_bootstrap'];assert b['outer']==300 and b['inner']==10000 and b['bootstrap_seed_each_experiment']==116990005
        assert b['valid_min']>=9900
        for k,count in b['counts'].items():assert math.isclose(b['rates'][k],count/300)
    for p in list(R.glob('HOLDOUT_D_AMEND1_*.py'))+list((R/'HOLDOUT_D_runner').glob('*.py')):ast.parse(p.read_text())
    log=(R/'HOLDOUT_D_runner/HOLDOUT_D_MOCK_TEST_RELEASE.log').read_text()
    test=json.loads(log.strip().splitlines()[-1]);assert test['passed'] and test['tests']==22 and test['failures']==test['errors']==0
    artifacts=R/test['artifacts'];assert artifacts.is_relative_to(R/'HOLDOUT_D_runner')
    assert any((artifacts/'response_crash').glob('HOLDOUT_D_RECOVERY_*.jsonl'))
    envs=list((R/'HOLDOUT_D_runner/HOLDOUT_D_LAUNCH_TEST_FINAL').glob('HOLDOUT_D_LAUNCH_ENV.json'))
    env=json.loads(envs[0].read_text())
    assert env['sid']!=env['original_sid'] and env['ppid']==1 and not env['tty'] and env['stdin']=='/dev/null'
    assert env['memory_max']=='2147483648' and env['swap_max']=='0' and env['flock_held']
    final=R/'HOLDOUT_D_runner/HOLDOUT_D_DETACHED_MOCK_FINAL'
    summary=json.loads((final/'HOLDOUT_D_SUMMARY.json').read_text())
    assert summary['terminal']==144 and summary['successful']==summary['attempts']==1584 and not summary['halt']
    ev=[json.loads(x) for x in (final/'HOLDOUT_D_WAL.jsonl').read_text().splitlines()]
    head='0'*64;pending=set();success=attempts=0;responses=set();actions=set()
    for i,event in enumerate(ev,1):
        e=event.copy();h=e.pop('digest');assert e['seq']==i and e['prev']==head
        raw=json.dumps(e,ensure_ascii=False,separators=(',',':')).encode()
        assert hashlib.sha256(raw).hexdigest()==h;head=h
        if e['kind']=='reserve':
            attempts+=1;assert e['call']==attempts;pending.add(e['call']);assert success+len(pending)<=9000 and attempts<=12000
        if e['kind']=='response':
            assert e['call'] in pending;pending.remove(e['call']);responses.add(e['call']);success+=200<=e['status']<300
        if e['kind']=='action':assert e['call'] in responses and e['call'] not in actions;actions.add(e['call'])
    assert not pending and success==attempts==len(actions)==1584
    assert any(e['kind']=='launch' and e['environment']['ppid']==1 and not e['live'] for e in ev)
    manifest=R/'HOLDOUT_D_AMEND1.sha256'
    if z.freeze:
        paths=sorted([p for p in R.glob('HOLDOUT_D_AMEND1_*') if p.is_file() and p.name!='HOLDOUT_D_AMEND1_VALIDATION.json']+
                     [p for p in (R/'HOLDOUT_D_runner').rglob('*') if p.is_file()]+[R/'experiments/real/reliability_core.py'])
        assert not any(p.is_symlink() or p.suffix in ('.tmp','.pyc') for p in paths)
        with manifest.open('x') as f:
            for p in paths:f.write(sha(p)+'  '+str(p.relative_to(R))+'\n')
    frozen=manifest_check(manifest)
    result=dict(status='PASS',verified_utc=datetime.now(timezone.utc).isoformat(),original_frozen_files=original,
                original_inputs_unchanged=len(a['source_hashes']),amendment_manifest_entries=frozen,
                scenarios=6,simulations_each=10000,actual_bootstrap_outer_each=300,actual_bootstrap_inner=10000,
                mock_tests=22,detached_mock_terminal=144,detached_mock_returns=1584,
                independent_WAL_chain_order_budget_check=True,launch_verified=True,real_llm_calls=0,real_quota_probes=0,
                execution_ready=False,note='本验证结果在冻结之后生成，故不列入被验证清单；清单不含自身。')
    if not z.check_only:
        with (R/'HOLDOUT_D_AMEND1_VALIDATION.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
