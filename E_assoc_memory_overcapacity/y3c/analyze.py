"""固定统计分析；只用新数据和冻结预测，不读X1终点。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU, (45, 50))
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.stats import t
from common import ROOT, OUT, METHODS, TASKS, PAIRS, DIMS, config, jobs, verify, dump, table, sha, usage
from predict import load
from integrity import inspect

def interval(values, family):
    a = np.asarray(values, dtype=float)
    assert np.isfinite(a).all() and family >= 1
    n = len(a)
    result = dict(n=n, mean=float(a.mean()) if n else None, se=None, critical=None,
                  lower=None, upper=None, p_two_sided=None, p_bonferroni=None)
    if n < 2: return result
    mean = result['mean']; se = float(a.std(ddof=1)/math.sqrt(n))
    crit = float(t.ppf(1 - .05/(2*family), n-1))
    p = float(2*t.sf(abs(mean/se), n-1)) if se else (1. if mean == 0 else 0.)
    result.update(se=se, critical=crit, lower=mean-crit*se, upper=mean+crit*se,
                  p_two_sided=p, p_bonferroni=min(1., family*p))
    return result

def equivalent(stat, band=.005):
    return stat['lower'] is not None and stat['lower'] > -band and stat['upper'] < band

def loss(y, uncorrected, corrected):
    delta = corrected-uncorrected
    # 与两平方差严格等价，避免消减；权重固定。
    return 2*delta*(y-uncorrected)-delta*delta

def analysis(records, pred, cfg, output, started=None):
    """records[(d,ir,seed)][method,task]；合成表亦使用同一实现。"""
    count = cfg['blocks']
    complete_blocks = [s for s in range(count) if all((d, ir, s) in records for d in DIMS for ir in range(3))]
    endpoints = []
    for task in TASKS:
        values = []
        for s in complete_blocks:
            ll = []
            for d in DIMS:
                for ir in range(3):
                    rr = records[d, ir, s]
                    if task == 'old':
                        for a, b in PAIRS:
                            p, q = pred[d, ir, a, task], pred[d, ir, b, task]
                            ll.append(loss(rr[a,task]['recall']-rr[b,task]['recall'],
                                           p['recall_limit']-q['recall_limit'], p['recall_first_order']-q['recall_first_order']))
                    else:
                        for m in METHODS:
                            p = pred[d, ir, m, task]
                            ll.append(loss(rr[m,task]['recall'], p['recall_limit'], p['recall_first_order']))
            values.append(float(np.mean(ll)))
        stat = interval(values, cfg['confirm_family'])
        endpoints.append(dict(endpoint='three_dim_'+task, **stat,
                              eligible=len(values)>=cfg['min_blocks'],
                              passed=len(values)>=cfg['min_blocks'] and stat['lower'] is not None and stat['lower']>0))
    vals = []
    a, b = cfg['target_pair']
    p, q = pred[512, 2, a, 'old'], pred[512, 2, b, 'old']
    for s in range(count):
        if (512, 2, s) not in records: continue
        rr = records[512, 2, s]
        vals.append(loss(rr[a,'old']['recall']-rr[b,'old']['recall'],
                         p['recall_limit']-q['recall_limit'], p['recall_first_order']-q['recall_first_order']))
    stat = interval(vals, cfg['confirm_family'])
    eligible = cfg['d512_confirmatory'] and len(vals) == count
    endpoints.append(dict(endpoint='d512_rminus_half_GD_minus_Sign_old', **stat,
                          eligible=eligible, passed=eligible and stat['lower'] is not None and stat['lower']>0))
    table(output/'endpoints.csv', endpoints)
    diagnostics = []
    for d in DIMS:
        for ir in range(3):
            for task in TASKS:
                labels = PAIRS if task=='old' else [(m, None) for m in METHODS]
                for a, b in labels:
                    p = pred[d, ir, a, task]
                    limit = p['recall_limit']; corrected = p['recall_first_order']
                    if b:
                        q = pred[d, ir, b, task]
                        limit -= q['recall_limit']; corrected -= q['recall_first_order']
                    values = []
                    for s in range(count):
                        if (d, ir, s) in records:
                            rr = records[d, ir, s]
                            values.append(rr[a,task]['recall']-(rr[b,task]['recall'] if b else 0.))
                    stat = interval(np.array(values)-corrected, 72)
                    diagnostics.append(dict(d=d, ir=ir, task=task, label=a+'-'+b if b else a,
                        observed=float(np.mean(values)) if values else None, uncorrected=limit, corrected=corrected,
                        coefficient=d*(corrected-limit), scaled_observed_bias=d*(float(np.mean(values))-limit) if values else None,
                        **stat, equivalent=equivalent(stat, cfg['equivalence_band'])))
    assert len(diagnostics)==72
    table(output/'diagnostics72.csv', diagnostics)
    secondary = []; mechanism = []
    for d in DIMS:
        for ir in range(3):
            for m in METHODS:
                values = [records[d,ir,s][m,'old']['recall'] for s in range(count) if (d,ir,s) in records]
                p = pred[d,ir,m,'old']
                secondary.append(dict(d=d,ir=ir,method=m,uncorrected=p['recall_limit'],
                    corrected=p['recall_first_order'],**interval(values,45)))
                for task in TASKS:
                    p=pred[d,ir,m,task]
                    rr=[records[d,ir,s][m,task] for s in range(count) if (d,ir,s) in records]
                    def avg(k): return float(np.mean([r[k] for r in rr])) if rr else None
                    mechanism.append(dict(d=d,ir=ir,method=m,task=task,n=len(rr),
                        mean_D=avg('D'),mean_V=avg('V'),
                        scaled_D_bias=d*(avg('D')-p['mu']) if rr else None,
                        scaled_V_bias=d*(avg('V')-1) if rr else None,
                        proved_m=p['m'],proved_w=p['w'],
                        scaled_query_var_D=d*avg('var_D') if rr else None,
                        scaled_query_var_V=d*avg('var_V') if rr else None,
                        scaled_query_cov_DV=d*avg('cov_DV') if rr else None,
                        candidate_var_V=4.,candidate_cov_DV=4*p['mu']))
    table(output/'secondary_old45.csv',secondary)
    table(output/'mechanism90.csv',mechanism)
    ledger = []; per_dataset = []
    for d, ir, s in jobs(count):
        present = (d,ir,s) in records
        state='complete' if present else 'interrupted_uncommitted' if started is not None and (d,ir,s) in started else 'not_started'
        ledger.append(dict(d=d,ir=ir,seed=s,state=state,complete=present,complete_block=s in complete_blocks))
        if present:
            for m in METHODS:
                for task in TASKS:
                    per_dataset.append(dict(d=d,ir=ir,seed=s,method=m,task=task,
                        complete_block=s in complete_blocks,**records[d,ir,s][m,task]))
    table(output/'availability.csv',ledger)
    table(output/'all_completed.csv',per_dataset,fields=['d','ir','seed','method','task','complete_block',
                                                      'recall','D','V','var_D','var_V','cov_DV'])
    return dict(complete_cells=len(records), complete_blocks=len(complete_blocks),
        ungrouped_complete_cells=sum(1 for d,ir,s in records if s not in complete_blocks),
        primary_both_passed=all(e['passed'] for e in endpoints[:2]),d512_passed=endpoints[2]['passed'],
        full72_equivalent=all(r['equivalent'] for r in diagnostics),
        planned_complete=len(records)==9*count,diagnostic_family=72,confirm_family=cfg['confirm_family'])

def read_cell(path, mh, cfg):
    with np.load(path, allow_pickle=False) as f:
        meta=json.loads(str(f['metadata']))
        d,ir,s = meta['d'],meta['ir'],meta['seed']
        assert (d,ir,s) in set(jobs(cfg['blocks'])) and meta['manifest_sha256']==mh and meta['schema']==1
        assert path.name==f'd{d}_r{ir}_s{s}.npz'
        Q=cfg['query_target'] if (d,ir)==(512,2) else cfg['query_default']
        n=d*d//4; ids=f['ids']
        assert ids.shape==(Q,) and np.issubdtype(ids.dtype,np.integer)
        assert len(np.unique(ids))==Q and ids.min()>=0 and ids.max()<n
        required={'ids','metadata'} | {m+'_'+task+'_'+suffix for m in METHODS for task in TASKS
                                                    for suffix in ['ranks','success','D','V']}
        assert set(f.files)==required
        rr={}
        for m in METHODS:
            for task in TASKS:
                rank,ok,D,V=[f[m+'_'+task+'_'+k] for k in ['ranks','success','D','V']]
                assert all(a.shape==(Q,) for a in [rank,ok,D,V])
                assert np.issubdtype(rank.dtype,np.integer) and ok.dtype==np.bool_
                assert rank.min()>=1 and rank.max()<=n and np.array_equal(ok,rank<=math.ceil(.1*n))
                assert D.dtype==np.float64 and V.dtype==np.float64 and np.isfinite(D).all() and np.isfinite(V).all() and (V>=0).all()
                rr[m,task]=dict(recall=float(ok.mean()),D=float(D.mean()),V=float(V.mean()),
                    var_D=float(D.var(ddof=1)),var_V=float(V.var(ddof=1)),cov_DV=float(np.cov(D,V,ddof=1)[0,1]))
        return (d,ir,s),rr

def main():
    assert sys.argv[1:]==['--formal-via-launcher'] and __import__('os').environ.get('Y3C_LAUNCHER')=='1'
    mh=verify();cfg=config();pred,_=load();data=OUT/'formal';output=data/'analysis'
    output.mkdir()  # 存在即拒绝覆盖
    dump(output/'start.json',dict(manifest_sha256=mh,usage=usage()))
    records={};errors=[];data_hashes={}
    for path in sorted(data.glob('*.npz')):
        try:
            data_hashes[path.name]=sha(path)
            key,rr=read_cell(path,mh,cfg)
            assert key not in records
            records[key]=rr
        except Exception as exc:
            errors.append(dict(file=path.name,error=repr(exc)))
    dump(output/'data_hashes.json',data_hashes)
    dump(output/'invalid_files.json',errors)
    integrity=inspect(data,mh,cfg,records,data_hashes)
    dump(output/'integrity_errors.json',integrity['errors'])
    result=analysis(records,pred,cfg,output,started=integrity['started'])
    result['integrity_errors']=integrity['errors']
    result['event_parse_errors']=[e for e in integrity['errors'] if e['code']=='invalid_event']
    result['invalid_files']=errors
    result['partial_files']=[p.name for p in data.glob('*.partial')]
    result['analysis_complete']=True
    result['candidate_confirmation_valid']=not errors and integrity['valid']
    result['candidate_decisions']={k:result[k] for k in ['primary_both_passed','d512_passed','full72_equivalent']}
    # 分析进程只能交付候选；退出前任何时间被杀也不会留下有效确认。
    result['confirmation_valid']=False
    result['publication_state']='pending_supervisor'
    result['primary_both_passed']=False;result['d512_passed']=False;result['full72_equivalent']=False
    result['runner_exit']=integrity['runner_exit']
    result['readable_cells']=[list(k) for k in sorted(records)]
    result['data_hashes']=data_hashes
    result['manifest_sha256']=mh
    result['usage']=usage()
    result['runner_status_present']=(data/'runner_status.json').is_file()
    dump(output/'PENDING.json',result)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
