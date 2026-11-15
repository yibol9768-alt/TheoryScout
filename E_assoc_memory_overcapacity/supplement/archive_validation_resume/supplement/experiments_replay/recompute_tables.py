"""Recompute manuscript tables from dataset-level measurements, not saved CIs."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t
from population_model import METHODS,CORRELATIONS,PAIRS,population,predictions
HERE=Path(__file__).resolve().parent

def rows(path):
    if path.suffix=='.jsonl':return [json.loads(s) for s in path.read_text().splitlines()]
    with path.open() as f:return list(csv.DictReader(f))
def write(path,data):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def interval(values,family):
    v=np.asarray(values);n=len(v);mean=float(v.mean());se=float(v.std(ddof=1)/math.sqrt(n));crit=float(t.ppf(1-.05/(2*family),n-1))
    p=float(2*t.sf(abs(mean/se),n-1)) if se else float(mean==0)
    return dict(n=n,mean=mean,se=se,lower=mean-crit*se,upper=mean+crit*se,p_bonferroni=min(1.,family*p))
def loss(y,p0,p1):return 2*(p1-p0)*(y-p0)-(p1-p0)**2

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--matched-measurements',type=Path,default=HERE/'matched_retrieval_measurements.jsonl')
    p.add_argument('--confirmation-measurements',type=Path,default=HERE/'finite_dimension_measurements.csv')
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    params=population();pop={(r['ir'],r['method']):r for r in params};pp=predictions(params)
    pred={(int(r['d']),int(r['ir']),r['method'],r['task']):r for r in pp}
    frozen=rows(HERE/'finite_dimension_predictions.csv')
    max_prediction_error=max(abs(float(r[k])-pred[int(r['d']),int(r['ir']),r['method'],r['task']][k]) for r in frozen for k in ['mu','m','w','B_recall','recall_limit','recall_first_order'])
    assert max_prediction_error<1e-9
    write(a.output/'regenerated_predictions.csv',pp)
    mr={}
    for r in rows(a.matched_measurements):
        key=(int(r['d']),int(r['ir']),int(r['seed']),r['method'])
        if 'task' in r:mr.setdefault(key,{})[r['task']]=float(r['recall'])
        else:
            assert key not in mr;mr[key]={task:float(r[task]) for task in ['old','new']}
    seeds=sorted({s for d,ir,s,m in mr});dims=sorted({d for d,ir,s,m in mr})
    assert len(mr)==len(seeds)*len(dims)*15
    contrasts=[];matching=[]
    for d in dims:
        for ir in range(3):
            for m in METHODS:
                target=pop[ir,m]['new_recall'];stat=interval([mr[d,ir,s,m]['new']-target for s in seeds],30)
                matching.append(dict(d=d,r=CORRELATIONS[ir],method=m,practical_match=stat['lower']>=-.015 and stat['upper']<=.015,**stat))
            for m,n in PAIRS:
                prediction=pop[ir,m]['old_recall']-pop[ir,n]['old_recall']
                stat=interval([mr[d,ir,s,m]['old']-mr[d,ir,s,n]['old'] for s in seeds],18)
                contrasts.append(dict(d=d,r=CORRELATIONS[ir],contrast=m+'-'+n,prediction=prediction,
                    prediction_covered=stat['lower']<=prediction<=stat['upper'],positive=stat['lower']>0,**stat))
    write(a.output/'matched_retrieval.csv',contrasts);write(a.output/'new_task_matching.csv',matching)
    md=['|r|对比|预测（百分点）|d=128|d=256|','|---|---|---:|---:|---:|']
    for ir,r in enumerate(CORRELATIONS):
        for m,n in PAIRS:
            rr=[next(v for v in contrasts if v['d']==d and v['r']==r and v['contrast']==m+'-'+n) for d in dims]
            md.append('|'+str(r)+'|'+m+'−'+n+'|'+f"{100*rr[0]['prediction']:.2f}"+'|'+ '|'.join(f"{100*v['mean']:.2f}"+('†' if v['prediction_covered'] else '') for v in rr)+'|')
    (a.output/'matched_retrieval_table.md').write_text('\n'.join(md)+'\n')
    data={}
    for r in rows(a.confirmation_measurements):
        key=(int(r['d']),int(r['ir']),int(r['seed']),r['method'],r['task']);assert key not in data
        data[key]={k:float(r[k]) for k in ['recall','D','V','var_D','var_V','cov_DV']}
    blocks=sorted({key[2] for key in data});assert len(data)==len(blocks)*90
    def observation(d,ir,s,m,task):return data[d,ir,s,m,task]['recall']
    endpoint_values={}
    for task in ['old','new']:
        vals=[]
        for s in blocks:
            terms=[]
            for d in [128,256,512]:
                for ir in range(3):
                    pairs=PAIRS if task=='old' else [(m,None) for m in METHODS]
                    for m,n in pairs:
                        target=pred[d,ir,m,task];p0=target['recall_limit'];p1=target['recall_first_order'];y=observation(d,ir,s,m,task)
                        if n:
                            target=pred[d,ir,n,task];p0-=target['recall_limit'];p1-=target['recall_first_order'];y-=observation(d,ir,s,n,task)
                        terms.append(loss(y,p0,p1))
            vals.append(float(np.mean(terms)))
        endpoint_values['three_dim_'+task]=vals
    p0=pred[512,2,'GD','old']['recall_limit']-pred[512,2,'Sign','old']['recall_limit']
    p1=pred[512,2,'GD','old']['recall_first_order']-pred[512,2,'Sign','old']['recall_first_order']
    endpoint_values['d512_rminus_half_GD_minus_Sign_old']=[loss(observation(512,2,s,'GD','old')-observation(512,2,s,'Sign','old'),p0,p1) for s in blocks]
    endpoints=[]
    for name,v in endpoint_values.items():
        st=interval(v,3);endpoints.append(dict(endpoint=name,**st,eligible=len(blocks)>=24,passed=len(blocks)>=24 and st['lower']>0))
    write(a.output/'confirmatory_endpoints.csv',endpoints)
    diagnostics=[];secondary=[];mechanism=[]
    for d in [128,256,512]:
        for ir in range(3):
            for task in ['old','new']:
                for m,n in PAIRS if task=='old' else [(m,None) for m in METHODS]:
                    target=pred[d,ir,m,task];p0=target['recall_limit'];p1=target['recall_first_order']
                    if n:p0-=pred[d,ir,n,task]['recall_limit'];p1-=pred[d,ir,n,task]['recall_first_order']
                    vv=[observation(d,ir,s,m,task)-(observation(d,ir,s,n,task) if n else 0.) for s in blocks]
                    st=interval(np.asarray(vv)-p1,72)
                    diagnostics.append(dict(d=d,ir=ir,task=task,label=m+'-'+n if n else m,observed=float(np.mean(vv)),uncorrected=p0,corrected=p1,
                        coefficient=d*(p1-p0),scaled_observed_bias=d*(float(np.mean(vv))-p0),**st,equivalent=st['lower']>-.005 and st['upper']<.005))
            for m in METHODS:
                target=pred[d,ir,m,'old'];secondary.append(dict(d=d,ir=ir,method=m,uncorrected=target['recall_limit'],corrected=target['recall_first_order'],**interval([observation(d,ir,s,m,'old') for s in blocks],45)))
                for task in ['old','new']:
                    p=pred[d,ir,m,task];rr=[data[d,ir,s,m,task] for s in blocks]
                    avg=lambda k:float(np.mean([r[k] for r in rr]))
                    mechanism.append(dict(d=d,ir=ir,method=m,task=task,n=len(rr),mean_D=avg('D'),mean_V=avg('V'),scaled_D_bias=d*(avg('D')-p['mu']),
                        scaled_V_bias=d*(avg('V')-1),proved_m=p['m'],proved_w=p['w'],scaled_query_var_D=d*avg('var_D'),scaled_query_var_V=d*avg('var_V'),scaled_query_cov_DV=d*avg('cov_DV')))
    write(a.output/'diagnostics72.csv',diagnostics);write(a.output/'secondary_old45.csv',secondary);write(a.output/'mechanism90.csv',mechanism)
    summary=[]
    for d in [128,256,512]:
        rr=[v for v in diagnostics if v['d']==d]
        summary.append(dict(d=d,cells=24,equivalent=sum(v['equivalent'] for v in rr),nonequivalent_width_at_least_one_point=sum(not v['equivalent'] and v['upper']-v['lower']>=.01 for v in rr),
            point_within_half_point=sum(abs(v['mean'])<=.005 for v in rr),interval_excludes_zero=sum(v['lower']>0 or v['upper']<0 for v in rr),mean_abs_residual_points=100*float(np.mean([abs(v['mean']) for v in rr]))))
    write(a.output/'diagnostics_summary.csv',summary)
    result=dict(status='PASS',regenerated_prediction_max_error=max_prediction_error,matched_datasets=len(mr)//5,confirmation_datasets=len(data)//10,
        matched_positive_contrasts=sum(r['positive'] for r in contrasts),matched_population_values_covered=sum(r['prediction_covered'] for r in contrasts),
        new_task_equivalence=sum(r['practical_match'] for r in matching),confirmatory_endpoints_passed=sum(r['passed'] for r in endpoints),diagnostic_equivalence=sum(r['equivalent'] for r in diagnostics))
    if len(seeds)==192 and len(blocks)==32:
        assert result['matched_positive_contrasts']==18 and result['matched_population_values_covered']==7
        assert result['new_task_equivalence']==22 and result['confirmatory_endpoints_passed']==3 and result['diagnostic_equivalence']==10
        expected=[6.776452474841476e-5,7.462337632869399e-5,4.147854158910549e-5]
        assert max(abs(r['mean']-v) for r,v in zip(endpoints,expected))<1e-12
        result['manuscript_tables_reproduced']=True
    with (a.output/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
