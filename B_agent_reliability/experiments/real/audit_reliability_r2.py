"""Supplementary offline audit, registered before calibration; no model calls.

Does not change frozen protocols, executors, selection, or admission thresholds.
Adds cluster intervals for already preregistered estimands and checks timing.
"""
import argparse
import collections
from datetime import datetime
import hashlib
import json
import random
import reliability_r2 as r
import report_reliability_r2 as report

METRICS = {'自然错误':('e','n'), '首次错前风险':('first_e','first_n'),
           '首错后风险':('after_e','after_n'),
           '反馈首复查detect':('first_detect','first_reviews'),
           '首复查修好':('first_fixed','first_reviews'),
           '二复查修好':('second_fixed','second_reviews'),
           '复查仍错':('repair_reerrors','repair_attempts')}

def stamp(x):
    return datetime.fromisoformat(x).timestamp()

def main(phase):
    name='DIFFICULTY_R2' if phase=='r2' else 'CALIB1'
    out=r.ROOT/('difficulty_r2_real' if phase=='r2' else 'calib1_real')
    a=json.loads((r.ROOT/f'ANALYSIS_{name}.json').read_text())
    budget=[json.loads(s) for s in (out/'budget.jsonl').read_text().splitlines()]
    checks=0
    streak=0
    must_cool=False
    forbidden_until=0
    for x in budget:
        if x['event']=='outcome':
            streak=streak+1 if x['failed'] else 0
            assert streak==x['streak']
            if streak>=5:
                must_cool=True
        elif x['event']=='cooldown':
            assert must_cool and x['seconds']==600
            forbidden_until=max(forbidden_until,stamp(x['utc'])+600)
            streak=0
            must_cool=False
        elif x['event']=='reserve':
            assert stamp(x['utc'])>=forbidden_until-.05
        checks+=1
    assert not must_cool
    for result in a['rows']:
        events=[json.loads(s) for s in (out/(result['id']+'.jsonl')).read_text().splitlines()]
        recent={}
        requests=[x for x in events if x['event']=='request']
        for x in requests:
            key=x['step'],x['review']
            if x['attempt']:
                prev=recent[key]
                failure=next(e for e in events if e['event']=='service_error' and e['call']==prev['call'])
                assert stamp(x['utc'])-stamp(failure['utc']) >= (10,30,60,120)[x['attempt']-1]-.05
                assert prev['messages']==x['messages']
            recent[key]=x
            checks+=1
    groups=collections.defaultdict(list)
    for row in a['rows']:
        groups[row['family'],row['level']].append(row)
    cells=[]
    for (family,level),rows in groups.items():
        rng=random.Random(752911)
        samples={name:[] for name in METRICS}
        for _ in range(10000):
            chosen=rng.choices(rows,k=len(rows))
            for label,(num,den) in METRICS.items():
                n=sum(x[den] for x in chosen)
                if n:
                    samples[label].append(sum(x[num] for x in chosen)/n)
        metrics={}
        for label,(num,den) in METRICS.items():
            boot=sorted(samples[label])
            n=sum(x[den] for x in rows)
            e=sum(x[num] for x in rows)
            metrics[label]=dict(e=e,n=n,cluster_95=[boot[int(.025*(len(boot)-1))],boot[int(.975*(len(boot)-1))]] if boot else None,
                                valid_bootstrap=len(boot),iid_reference=report.exact(e,n))
        cells.append(dict(family=family,level=level,metrics=metrics))
    if phase=='calib1':
        selection=json.loads((r.ROOT/'FREEZE_SELECTION_CALIB1.json').read_text())
        assert selection['selected']==r.selection()['selected']
        for path,h in selection['input_sha256'].items():
            assert r.sha(r.ROOT/path)==h
        for config in groups:
            boundaries=[x for x in budget if x['event']=='service_boundary' and x['config']==f'{config[0]}_d{config[1]}']
            count=len(groups[config])
            assert [x['attempted'] for x in boundaries]==[n for n in (10,25,50) if n<=count]
            for boundary in boundaries:
                recent=[x['failed'] for x in budget if x['event']=='outcome' and stamp(x['utc'])<stamp(boundary['utc'])][-20:]
                assert sum(recent)==boundary['failures'] and len(recent)==boundary['recent_count']
                checks+=1
    result=dict(utc=r.d.now(),phase=phase,checks=checks,cells=cells,
                audit_sha256=r.sha(r.ROOT/'audit_reliability_r2.py'),
                no_threshold_changes=True)
    (r.ROOT/f'AUDIT_{name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['','## 补充离线核验与分簇区间','',
           f'附加审计完成{checks}项退避间隔、冷却禁发及边界检查；执行器、预注册和所有门槛保持冻结。',
           '以下对预先登记的各率按整题簇重采样10000次（固定种子752911），零分母重采样不生成伪造估计。窄区间不证明独立，早停和服务缺失仍限制解释。','',
           '|配置|指标|e/n|整题簇95%|有效bootstrap|','|---|---|---|---|---|']
    for cell in cells:
        for label,s in cell['metrics'].items():
            lines.append(f"|{cell['family']}/{cell['level']}|{label}|{s['e']}/{s['n']}|{report.interval(s['cluster_95'])}|{s['valid_bootstrap']}|")
    lines += ['',f'机器可读结果见 [AUDIT_{name}.json](AUDIT_{name}.json)。']
    path=r.ROOT/(name+'.md')
    existing=path.read_text()
    assert '## 补充离线核验与分簇区间' not in existing
    with path.open('a') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(phase=phase,checks=checks,cells=len(cells)),ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['r2','calib1'])
    main(ap.parse_args().phase)
