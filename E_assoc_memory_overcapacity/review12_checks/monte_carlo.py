"""Actual two-step training, training candidates and queries; no proxy updates.
Fixed population parameters (no empirical retuning). Paired comparisons.
"""
import gc
import json
import math
import time
import numpy as np
from scipy.stats import norm, t as student
from environment import resources

PARAMETERS={
    'GD': 1-math.sqrt(1-2/(2+math.sqrt(21))),
    'Adam_1': .3806017005,
    'Adam_0.1': .2233914125,
    'Sign': .2049397185,
}
PREDICTED_C0={'GD':math.sqrt(21)/5,'Adam_1':.90477480,'Adam_0.1':.86752635,'Sign':.84906214}


def train(z0,z1,s,algorithm):
    u=z0.copy()
    rate=PARAMETERS[algorithm]
    m=np.zeros_like(u); v=np.zeros_like(u)
    for step in (1,2):
        residual=u@s-z1
        if algorithm=='GD':
            u-=rate*residual
        elif algorithm=='Sign':
            u-=math.sqrt(2)*rate*np.sign(residual)
        else:
            e=1 if algorithm=='Adam_1' else .1
            m=.9*m+.1*residual
            v=.99*v+.01*residual**2
            mh=m/(1-.9**step); vh=v/(1-.99**step)
            u-=math.sqrt(2)*rate*mh/(np.sqrt(vh)+math.sqrt(2)*e)
    return u


def evaluate(u,x,y,queries,rho=.1):
    successes=[]; corrects=[]; wrong_sum=0.; wrong_square=0.; count=0
    n=x.shape[1]
    for start in range(0,len(queries),128):
        labels=queries[start:start+128]
        score=y.T@(u@x[:,labels])
        correct=score[labels,np.arange(len(labels))]
        rank=1+np.sum(score>correct[None,:],axis=0)
        successes.extend((rank<=math.ceil(rho*n)).tolist())
        corrects.extend(correct.tolist())
        wrong_sum+=float(score.sum()-correct.sum())
        wrong_square+=float(np.einsum('ij,ij->',score,score)-correct@correct)
        count+=(n-1)*len(labels)
    variance=wrong_square/count-(wrong_sum/count)**2
    return dict(recall=float(np.mean(successes)),sample_mean_correct=float(np.mean(corrects)),wrong_sd=math.sqrt(variance),successes=successes)


def main():
    started=time.monotonic(); runs=[]
    for d in (256,512):
        n=d*d//4; kappa=n/d**2
        for replicate in range(3):
            seed=120000+d*10+replicate
            rng=np.random.default_rng(seed)
            x0=rng.standard_normal((d,n))/math.sqrt(d)
            y0=rng.standard_normal((d,n))/math.sqrt(d)
            x1=rng.standard_normal((d,n))/math.sqrt(d)
            y1=rng.standard_normal((d,n))/math.sqrt(d)
            z0=(d/math.sqrt(n))*(y0@x0.T)
            z1=(d/math.sqrt(n))*(y1@x1.T)
            s=(d/n)*(x1@x1.T)
            queries=rng.choice(n,size=1024,replace=False)
            run=dict(d=d,n=n,kappa=kappa,replicate=replicate,seed=seed,queries=len(queries),algorithms={})
            paired={}
            for alg in PARAMETERS:
                u=train(z0,z1,s,alg)
                q=float(np.einsum('ij,ij->',u,u))/d**2
                # Full training mean, exact identity: mean s_ii = <U,Z>/(d sqrt(n)).
                c0=float(np.einsum('ij,ij->',u,z0))/(d**2*math.sqrt(q))
                c1=float(np.einsum('ij,ij->',u,z1))/(d**2*math.sqrt(q))
                old=evaluate(u,x0,y0,queries)
                new=evaluate(u,x1,y1,queries)
                paired[alg]=np.asarray(old.pop('successes'),dtype=np.int8)
                new.pop('successes')
                run['algorithms'][alg]=dict(full_training_signal_c0=c0,full_training_signal_c1=c1,old=old,new=new)
            run['paired_old_differences']={}
            for high,low in zip(list(PARAMETERS)[:-1],list(PARAMETERS)[1:]):
                diff=paired[high]-paired[low]
                run['paired_old_differences'][high+' minus '+low]=dict(mean=float(diff.mean()),count_positive=int(np.sum(diff>0)),count_negative=int(np.sum(diff<0)))
            run['resources']=resources()
            runs.append(run)
            print('d',d,'replicate',replicate,'c0',{a:round(r['full_training_signal_c0'],7) for a,r in run['algorithms'].items()},'recall',{a:round(r['old']['recall'],7) for a,r in run['algorithms'].items()},'elapsed',time.monotonic()-started,flush=True)
            del x0,y0,x1,y1,z0,z1,s,u,paired
            gc.collect()
    summaries=[]
    for d in (256,512):
        selected=[r for r in runs if r['d']==d]
        summary=dict(d=d,algorithms={},paired_old_differences={})
        for alg in PARAMETERS:
            summary['algorithms'][alg]={field:float(np.mean([r['algorithms'][alg][field] for r in selected])) for field in ('full_training_signal_c0','full_training_signal_c1')}
            for task in ('old','new'):
                summary['algorithms'][alg][task+'_recall']=float(np.mean([r['algorithms'][alg][task]['recall'] for r in selected]))
        for high,low in zip(list(PARAMETERS)[:-1],list(PARAMETERS)[1:]):
            key=high+' minus '+low
            differences=np.array([r['paired_old_differences'][key]['mean'] for r in selected])
            mean=float(differences.mean())
            radius=float(student.ppf(.975,2)*differences.std(ddof=1)/math.sqrt(3))
            summary['paired_old_differences'][key]=dict(mean=mean,seed_cluster_t95=[mean-radius,mean+radius],direction_positive=mean>0)
        summaries.append(summary)
    result=dict(status='COMPLETED',description='Finite dimension diagnostic, not a proof. Three independent seeds per dimension; paired queries; cluster-t intervals have df=2 and are not simultaneous.',task_correlations={'rx':0,'ry':0},rho=.1,target_u=.4,parameters=PARAMETERS,predicted_c0=PREDICTED_C0,predicted_recall={a:float(norm.cdf(2*c+norm.ppf(.1))) for a,c in PREDICTED_C0.items()},predicted_new_recall=float(norm.cdf(.8+norm.ppf(.1))),runs=runs,summaries=summaries,seconds=time.monotonic()-started,resources=resources())
    with open('review12_checks/monte_carlo.json','x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'summaries':summaries,'seconds':result['seconds'],'resources':result['resources']},indent=2),flush=True)


if __name__=='__main__':main()
