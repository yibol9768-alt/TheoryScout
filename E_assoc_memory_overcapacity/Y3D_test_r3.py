"""R3 分支隔离检查；仅复用本轮已分析的确定性夹具。"""
import copy,csv,json,resource,shutil,sys
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(15,20))
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'y3d'))
import common,publication
common.environment()
TEST=ROOT/'y3d/r3_rerun';TEST.mkdir()
cfg=common.config();mh=common.sha(ROOT/'Y3C_FROZEN.sha256')
BASE=ROOT/'y3d/audit_rerun/independent_faults/control/formal'
FALSE=ROOT/'y3d/audit_rerun/false_confirmation/formal'
U=dict(cpu_seconds=1.,peak_rss_mib=100.,monotonic=1.)
STATUS=dict(stages=[dict(program=p,returncode=0,wall_hard_killed=False) for p in ['run.py','analyze.py']],
            wall_seconds=100.,cpu_seconds_children=3.,supervisor_usage=U)
checks=[]
def check(label,ok):
    assert ok,label
    checks.append(label)
def save(p,x):p.write_text(json.dumps(x,indent=2))
def clone(label,base=BASE):
    data=TEST/label/'formal'
    shutil.copytree(base,data,ignore=shutil.ignore_patterns('FINAL_ACCEPTED*','launch_status.json'))
    if base==FALSE:
        shutil.copyfile(FALSE.parent/'PENDING_before_mutation.json',data/'analysis/PENDING.json')
    return data
def reject(label,data):
    check(label,not publication.finalize(data,mh,cfg,copy.deepcopy(STATUS)) and not (data/'analysis/FINAL_ACCEPTED.json').exists())
def rebind_inventory(data):
    p=data/'analysis/PENDING.json';x=json.loads(p.read_text());x['report_inventory']=publication.report_inventory(data/'analysis');save(p,x)
def rewrite_csv(path,mutate):
    with path.open(newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    mutate(rows)
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

data=clone('failed_statistics_control',FALSE)
check('failed_statistics_execution_can_publish',publication.finalize(data,mh,cfg,copy.deepcopy(STATUS)))
accepted=publication.verify_accepted(data,mh,cfg)
check('failed_statistics_no_false_confirmation',all(accepted[k] is False for k in ['primary_both_passed','d512_passed','full72_equivalent']))
for key in ['primary_both_passed','d512_passed','full72_equivalent']:
    data=clone('flip_only_'+key,FALSE);p=data/'analysis/PENDING.json';x=json.loads(p.read_text())
    x['candidate_decisions'][key]=True;save(p,x);reject('single_candidate_flip_'+key,data)

# 重新绑定摘要仅用于分支隔离：确保覆盖检查本身有效，而非只靠哈希拒绝。
for label,mutate in [('missing_row',lambda r:r.pop()),('duplicate_row',lambda r:r.__setitem__(-1,r[0].copy()))]:
    data=clone(label);rewrite_csv(data/'analysis/all_completed.csv',mutate);rebind_inventory(data);reject('coverage_'+label,data)
for label,file,mutate in [
    ('endpoint_n','endpoints.csv',lambda r:r[0].update(n='23')),
    ('endpoint_eligibility','endpoints.csv',lambda r:r[0].update(eligible='False')),
    ('diagnostic_n','diagnostics72.csv',lambda r:r[0].update(n='31')),
    ('diagnostic_equivalence','diagnostics72.csv',lambda r:r[0].update(equivalent='False' if r[0]['equivalent']=='True' else 'True')),
]:
    data=clone(label);rewrite_csv(data/'analysis'/file,mutate);rebind_inventory(data);reject('independent_'+label,data)
data=clone('summary_count');p=data/'analysis/PENDING.json';x=json.loads(p.read_text());x['complete_cells']=287;save(p,x);reject('candidate_sample_summary',data)

# 同步重签测试凭据里的相关哈希，以单独走 verify 的独立重算分支。
# 这是测试隔离，不作为单故障攻击模型或真实监管签发。
for label,mode in [('verify_candidate_decisions','candidate'),('verify_coverage','coverage'),('verify_sample_count','sample')]:
    data=clone(label,FALSE if mode=='candidate' else BASE)
    check(label+'_control_publishes',publication.finalize(data,mh,cfg,copy.deepcopy(STATUS)))
    cp=data/'analysis/PENDING.json';candidate=json.loads(cp.read_text())
    ap=data/'analysis/FINAL_ACCEPTED.json';accepted=json.loads(ap.read_text())
    if mode=='candidate':
        for k in candidate['candidate_decisions']:candidate['candidate_decisions'][k]=True;accepted[k]=True
    elif mode=='coverage':
        rewrite_csv(data/'analysis/all_completed.csv',lambda r:r.pop())
        candidate['report_inventory']=publication.report_inventory(data/'analysis')
    else:candidate['complete_blocks']=23
    save(cp,candidate)
    for rel in accepted['evidence_sha256']:accepted['evidence_sha256'][rel]=common.sha(data/rel)
    save(ap,accepted)
    try:publication.verify_accepted(data,mh,cfg)
    except AssertionError as exc:check(label+'_independent_rejection',True)
    else:raise AssertionError(label+' accepted coherent but false evidence')

children=resource.getrusage(resource.RUSAGE_CHILDREN)
common.dump(TEST/'results.json',dict(passed=True,checks=checks,usage=common.usage(),
    cpu_seconds_children=children.ru_utime+children.ru_stime,no_formal_sampling=True,
    source_sha256={n:common.sha(ROOT/'y3d'/n) for n in ['publication.py','analyze.py']}))
print(json.dumps(dict(passed=True,checks=len(checks),usage=common.usage())))
