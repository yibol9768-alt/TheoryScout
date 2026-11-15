"""Resolve late sparse CE brackets using only saved directions, supplemental."""
import json
import time
import numpy as np
from util import HERE, DATA, save, sha, verify_prereg, metadata, make_bundle, forward
from replay import directions, trajectory


def main():
    start=time.monotonic();verify_prereg();out=HERE/'full'
    supplement=HERE/'SUPPLEMENT_DENSE_CE.md'
    assert sha(supplement)==(HERE/'SUPPLEMENT_DENSE_CE.sha256').read_text().split()[0]
    reports=[]
    for path in sorted((DATA/'runs').glob('e1_*_base.json')):
        d=json.loads(path.read_text());bundle=make_bundle(d['seed'],d['condition'])
        weights,_=trajectory(d,directions(DATA/d['directions_file']))
        cum=np.r_[0,np.cumsum(np.sqrt(np.sum(np.diff(weights,axis=0)**2,axis=(1,2,3))))]
        replay=json.loads((out/'replays'/f'{d["run_id"]}.json').read_text())
        measured={r['step']:dict(r) for r in replay['dense_CE']}
        newsteps=set();targets={}
        sparse_ce={r['step']:r['ce'] for r in d['train']}
        endpoint_error=0.
        for target in (1.,.8):
            bracket=next(((a['step'],b['step']) for a,b in zip(d['train'][:-1],d['train'][1:]) if a['ce']>=target and b['ce']<=target),None)
            if bracket:
                newsteps.update(range(bracket[0],bracket[1]+1))
            targets[str(target)]=dict(original_sparse_bracket=bracket)
        for t in sorted(newsteps):
            if t in measured and (t<=50):
                continue
            measured[t]=dict(step=t,ce=forward(weights[t],bundle['train'])['loss'],
                             Mabs=float(np.abs(forward(weights[t],bundle['probe'])['logits']).max()),
                             cumulative_actual_length=cum[t])
            if t in sparse_ce:
                endpoint_error=max(endpoint_error,abs(measured[t]['ce']-sparse_ce[t]))
        assert endpoint_error<=1e-5,(d['run_id'],endpoint_error)
        for target in (1.,.8):
            result=targets[str(target)];bracket=result['original_sparse_bracket']
            points=[measured[t] for t in sorted(measured)]
            best=min(points,key=lambda r:(abs(r['ce']-target),r['step']))
            adjacent=next(((a,b) for a,b in zip(points[:-1],points[1:]) if b['step']-a['step']==1 and a['ce']>=target and b['ce']<=target),None)
            result.update(status='adjacent_steps_observed' if adjacent else 'no_adjacent_downcrossing_observed',closest=best,error=abs(best['ce']-target),
                          eligible=abs(best['ce']-target)<=.005,
                          adjacent_bracket=dict(before=adjacent[0],after=adjacent[1],
                          logit_endpoint_range=[min(p['Mabs'] for p in adjacent),max(p['Mabs'] for p in adjacent)]) if adjacent else None)
        reports.append(dict(run_id=d['run_id'],seed=d['seed'],condition=d['condition'],eta=d['eta'],optimizer=d['optimizer'],
                            max_sparse_CE_replay_error=endpoint_error,targets=targets))
        print(f'{d["run_id"]} supplemental CE complete',flush=True)
    lookup={(r['condition'],r['eta'],r['seed'],r['optimizer']):r for r in reports}
    pairs=[]
    for main in (r for r in reports if r['optimizer']=='muon0'):
        for other in ('muon09','adamw','signsgd'):
            control=lookup[main['condition'],main['eta'],main['seed'],other]
            for target in ('1.0','0.8'):
                a,b=main['targets'][target],control['targets'][target]
                valid=(a.get('eligible',False) and b.get('eligible',False))
                difference=abs(a['closest']['ce']-b['closest']['ce']) if a['closest'] and b['closest'] else None
                pairs.append(dict(condition=main['condition'],eta=main['eta'],seed=main['seed'],target=float(target),comparator=other,
                                  CE_difference=difference,matched=bool(valid and difference<=.01),
                                  Mabs_ratio=a['closest']['Mabs']/b['closest']['Mabs'] if valid and difference<=.01 else None))
    counts=[]
    for other in ('muon09','adamw','signsgd'):
        for target in (1.,.8):
            selected=[p for p in pairs if p['comparator']==other and p['target']==target]
            matches=[p for p in selected if p['matched']]
            counts.append(dict(comparator=other,target=target,total_pairs=len(selected),matched_pairs=len(matches),
                               median_Mabs_ratio_muon_other=float(np.median([p['Mabs_ratio'] for p in matches])) if matches else None))
    save(out/'dense_ce_supplement.json',dict(status='supplement_only_main_prereg_unchanged',runs=reports,pairs=pairs,counts=counts,
         main_CE_table_replaced=False,supplement_sha256=sha(supplement),script_sha256=sha(__file__),resources=metadata(start)))


if __name__=='__main__':
    main()
