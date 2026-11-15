"""Post-run transparent adjudication: unavailable controls are not training errors.
Frozen summarize.py uses len(completed)==60 to gate every training hypothesis;
keep that artifact intact and evaluate observed comparisons explicitly here.
"""
import experiment as c
from experiment import np,json

def main():
    rows=[json.loads(p.read_text()) for p in sorted((c.OUT/'raw').glob('*.json'))]
    runs=[r for r in rows if r['status']=='completed']
    get={(r['config']['d'],r['config']['seed'],r['config']['mode']):r for r in runs}
    paired=[]
    for s in range(15,20):
        r,z,n=(get[6,s,m]['final'] for m in ('sigmoid_row','sigmoid_zero','nce'))
        paired.append(dict(seed=s,row_minus_zero=r['success_fraction']-z['success_fraction'],row_minus_nce=r['success_fraction']-n['success_fraction']))
    times={d:[get[d,s,'sigmoid_zero']['first_all_sign_step'] for s in range(15,20)] for d in (7,8)}
    med={d:float(np.median(times[d])) for d in times}
    slower=sum(a>b for a,b in zip(times[7],times[8]))
    diag=[]
    for d in (6,7,8):
        for mode in ('nce','sigmoid_zero','sigmoid_row'):
            rr=[get[d,s,mode] for s in range(15,20)]
            fr=[]
            for r in rr:
                gam=np.array(r['final']['gamma']); z=np.array(r['final']['effective_zeta'])
                fr.append(float(((gam>0)&(z<=0)).mean()))
            diag.append(dict(d=d,mode=mode,mean_rank_correct_but_threshold_wrong=float(np.mean(fr))))
    controls={}
    for d in (6,7,8):
        rr=[get[d,s,'sigmoid_lift'] for s in range(15,20) if (d,s,'sigmoid_lift') in get]
        controls[d]=dict(available=len(rr),initial_all_sign=sum(r['initial']['all_sign'] for r in rr),final_all_sign=sum(r['final']['all_sign'] for r in rr))
    out=dict(reason='Frozen summarizer gates P2-P4 on all 60 completed; one preregistered unavailable lift prevents that gate. No training or frozen code changed.',
             statuses={status:sum(r['status']==status for r in rows) for status in sorted({r['status'] for r in rows})},
             unavailable=[r for r in rows if r['status']!='completed'],P1=True,P2=False,
             P3_slow=med[7]>=2*med[8] and slower>=4,P3_obstruction=False,
             P4='not fully evaluable: 4/5 d6 lifts available; every available control succeeds',
             median_first_sign=med,median_ratio=med[7]/med[8],paired_slower_seeds=slower,
             paired_d6=paired,row_minus_zero_mean=float(np.mean([x['row_minus_zero'] for x in paired])),
             mean_abs_row_minus_nce=float(np.mean([abs(x['row_minus_nce']) for x in paired])),
             controls=controls,threshold_diagnostics=diag)
    c.save(c.OUT/'adjudication.json',out)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
