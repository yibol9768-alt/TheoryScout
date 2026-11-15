"""独立反例：原分析三个终点均失败，只改候选判定可否变成可验证确认。"""
import copy,csv,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];AUDIT=ROOT/'audit_y3c_checks'
sys.path.insert(0,str(ROOT/'y3c'))
import common,publication,predict
common.environment()
import numpy as np
def forbidden(*a,**k):raise RuntimeError('禁止随机抽样')
np.random.default_rng=forbidden;np.random.Generator=forbidden;np.random.PCG64=forbidden
cfg=common.config();mh=common.sha(ROOT/'Y3C_FROZEN.sha256');pred,_=predict.load()
CASE=AUDIT/'false_confirmation';DATA=CASE/'formal';DATA.mkdir(parents=True)
BASE=AUDIT/'independent_faults/control/formal'
for p in BASE.iterdir():
    if p.is_file() and p.name in publication.INPUTS:shutil.copyfile(p,DATA/p.name)
for d,ir,s in common.jobs(32):
    name=f'd{d}_r{ir}_s{s}.npz'
    with np.load(BASE/name,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    Q=len(a['ids']);n=d*d//4
    for m in common.METHODS:
        for task in common.TASKS:
            count=round(Q*pred[d,ir,m,task]['recall_limit'])
            if m=='GD' and task=='old':count+=s%3-1
            ok=np.arange(Q)<count
            a[m+'_'+task+'_success']=ok
            a[m+'_'+task+'_ranks']=np.where(ok,1,n).astype(np.int64)
    with (DATA/name).open('xb') as f:np.savez_compressed(f,**a)
events=[json.loads(line) for line in (DATA/'events.jsonl').read_text().splitlines()]
for e in events:
    if e['state']=='complete':
        d,ir,s=e['cell'];e['sha256']=common.sha(DATA/f'd{d}_r{ir}_s{s}.npz')
(DATA/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
worker=r'''
import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import analyze
data=Path(sys.argv[2]);mh=sys.argv[3]
analyze.OUT=data.parent;analyze.verify=lambda:mh
sys.argv=['analyze.py','--formal-via-launcher'];os.environ['Y3C_LAUNCHER']='1'
analyze.main()
'''
with (CASE/'actual_analyzer.log').open('x') as f:
    child=subprocess.run([sys.executable,'-B','-c',worker,str(ROOT/'y3c'),str(DATA),mh],stdout=f,stderr=subprocess.STDOUT,timeout=30)
assert child.returncode==0
p=DATA/'analysis/PENDING.json';candidate=json.loads(p.read_text())
assert candidate['candidate_confirmation_valid']
original=copy.deepcopy(candidate['candidate_decisions'])
assert not original['primary_both_passed'] and not original['d512_passed']
with (CASE/'PENDING_before_mutation.json').open('x') as f:json.dump(candidate,f,indent=2)
for k in candidate['candidate_decisions']:candidate['candidate_decisions'][k]=True
p.write_text(json.dumps(candidate,indent=2))
u=dict(cpu_seconds=1.,peak_rss_mib=100.,monotonic=1.)
state=dict(stages=[dict(program='run.py',returncode=0,wall_hard_killed=False),
                   dict(program='analyze.py',returncode=child.returncode,wall_hard_killed=False)],
           wall_seconds=100.,cpu_seconds_children=3.,supervisor_usage=u)
accepted=publication.finalize(DATA,mh,cfg,state)
credential=publication.verify_accepted(DATA,mh,cfg)
with (DATA/'analysis/endpoints.csv').open() as f:tables=list(csv.DictReader(f))
result=dict(original_decisions=original,analyzer_returncode=child.returncode,
            accepted=accepted,verification_succeeded=True,confirmation_valid=credential['confirmation_valid'],
            published_decisions={k:credential[k] for k in original},endpoints_untouched=tables,
            modified_only='PENDING.json candidate_decisions after normal analysis exit; NPZ and reports unchanged',
            no_formal_sampling=True,usage=common.usage())
with (CASE/'results.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps(result,ensure_ascii=False))
