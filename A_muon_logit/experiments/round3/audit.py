"""Audit saved evidence, source immutability, gates, calibration and CE matching."""
import json
import time
import numpy as np
from util import (HERE, ROOT, DATA, sha, save, verify_prereg, metadata, fit_beta, predict, max_acf)


def main():
    start=time.monotonic();prereg=verify_prereg();out=HERE/'full'
    replays=[json.loads(p.read_text()) for p in sorted((out/'replays').glob('*.json'))]
    assert len(replays)==72
    assert sum(r['primary'] for r in replays)==18
    seen=set()
    for r in replays:
        for filename,checksum in r['source_sha256'].items():
            assert sha(ROOT/filename)==checksum,filename
            seen.add(filename)
    old=json.loads((DATA/'v2_offline/diagnostics.json').read_text())
    spatial_count=0
    for run in old['runs']:
        for filename,checksum in run['source_sha256'].items():
            assert sha(DATA/filename)==checksum,filename
        for h in run['heads']:
            expected=(h['near_max_coverage']>=.8 and h['explained_change'] is not None and h['explained_change']>=.7 and h['delta_rms']>=.1)
            assert h['spatial_gate']==expected
            spatial_count+=expected
    assert spatial_count==38
    spatial_strata=[]
    for condition in ('isotropic','anisotropic'):
        retained=0
        for eta in (.01,.03,.1):
            rows=[r for r in old['runs'] if r['condition']==condition and r['eta']==eta]
            counts=[sum(r['heads'][h]['spatial_gate'] for r in rows) for h in range(4)]
            retained+=int(max(counts)>=2)
            spatial_strata.append(dict(condition=condition,eta=eta,same_head_seed_counts=counts))
        assert retained>=2
    stats=json.loads((out/'statistics.json').read_text())
    fits=[];maxfiterror=0.;maxmeanerror=0.;maxfrospread=0.
    for r in replays:
        for cp in r['checkpoints']:
            vals=[cp['comparisons']['matched_fro'][name]['update_fro'] for name in ('muon0','signsgd','adamw')]
            spread=float(np.ptp(vals,axis=0).max())
            maxfrospread=max(maxfrospread,spread)
        if not r['primary']:
            continue
        name=r['run_id'];arr=np.load(out/'arrays'/f'{name}.npz');fit=np.load(out/'arrays'/f'{name}_fit.npz')
        rep=json.loads((out/'feedback'/f'{name}.json').read_text())
        beta=fit_beta(arr['g'][:150],arr['xi'][:150],arr['bound'])
        err=float(np.max(np.abs(predict(arr['g'],beta,arr['bound'])-fit['m'])))
        maxfiterror=max(maxfiterror,err)
        assert arr['xi'].shape==(450,32) and arr['ab'].shape==(451,4,4,2)
        assert abs(float(max_acf(fit['innovation']))-rep['observed_max_acf'])<1e-12
        ymean=arr['xi'][150:].reshape(3,100,32).mean(1)
        pmean=fit['m'][150:].reshape(3,100,32).mean(1)
        direct=float(np.mean((ymean-pmean)**2))
        maxmeanerror=max(maxmeanerror,abs(direct-rep['all_candidates']['window_MSE_feedback']))
        fits.append(name)
    assert maxfiterror<1e-12 and maxmeanerror<1e-12 and maxfrospread<1e-10
    for pair in stats['CE']['pairs']:
        a,b=pair['muon'],pair['other']
        gate=(a['error']<=.005 and b['error']<=.005 and pair['CE_difference']<=.01)
        assert pair['matched']==gate
        for result in (a,b):
            bracket=result['first_downcrossing']
            if bracket and bracket['dense']:
                assert bracket['upper_step']['step']-bracket['lower_step']['step']==1
    supplement=json.loads((out/'dense_ce_supplement.json').read_text())
    assert supplement['main_CE_table_replaced'] is False and len(supplement['runs'])==72
    for run in supplement['runs']:
        for target, value in run['targets'].items():
            bracket=value['adjacent_bracket']
            if bracket:
                assert bracket['after']['step']-bracket['before']['step']==1
                assert bracket['before']['ce']>=float(target)>=bracket['after']['ce']
            assert value['eligible']==(value['error']<=.005)
    for pair in supplement['pairs']:
        if pair['matched']:
            assert pair['CE_difference']<=.01
    calibration=json.loads((out/'calibration_summary.json').read_text())
    assert len(calibration['runs'])==18 and calibration['full_size_auxiliary']
    for r in calibration['runs']:
        raw=np.load(out/'calibration'/f'{r["run_id"]}_statistics.npz')
        assert all(len(raw[k])==20000 for k in ('calibration','validation','rho_0.2','rho_0.4'))
        assert len(raw['permutation'])==1999 and r['channels']==32
        threshold=float(np.quantile(raw['calibration'],.95,method='higher'))
        assert threshold==r['critical_95pct']
        assert float(np.mean(raw['validation']>threshold))==r['independent_FWER']
        for rho in ('.2','.4'):
            label='rho_0'+rho
            assert float(np.mean(raw[label]>threshold))==r['power']['0'+rho]
        assert float((1+np.sum(raw['permutation']>=r['observed_T']))/2000)==r['exchangeability_sensitivity_p']
    validation={k:max(r['validation'][k] for r in replays) for k in replays[0]['validation']}
    save(out/'audit.json',dict(status='PASS',runs=72,primary_runs=18,original_files_unchanged=len(seen),
         primary_sources_match_v2=True,feedback_runs=len(fits),calibration_runs=18,
         inherited_spatial_pass=spatial_count,inherited_spatial_strata=spatial_strata,
         max_validation_errors=validation,max_refit_error=maxfiterror,max_window_MSE_error=maxmeanerror,
         max_matched_fro_spread=maxfrospread,exact_CE_pair_count=sum(p['matched'] for p in stats['CE']['pairs']),
         missing_CE_pair_count=sum(not p['matched'] for p in stats['CE']['pairs']),
         supplemental_CE_pair_count=sum(p['matched'] for p in supplement['pairs']),
         prereg_sha256=prereg,resources=metadata(start),script_sha256=sha(__file__)))
    print('Full audit PASS',flush=True)


if __name__=='__main__':
    main()
