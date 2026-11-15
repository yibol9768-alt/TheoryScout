"""Frozen-mode diagnostics with fixed windows and one-way forecasts."""
import numpy as np
from model import H


def statistics(x, alpha):
    # x: [time, head, Q/K], or [bootstrap,time,head,Q/K].
    axis = x.ndim-3
    mean = x.mean(axis=axis)
    centered = x-np.expand_dims(mean,axis)
    var = np.mean(centered**2,axis=axis)
    covariance = np.mean(centered[...,0]*centered[...,1],axis=axis)
    corr = covariance/np.sqrt(var[...,0]*var[...,1]+1e-30)
    ratio = mean**2/(var+1e-30)/(alpha/(2-alpha)) if alpha>0 else np.full_like(mean,np.nan)
    return mean,var,covariance,corr,ratio


def acf(x):
    axis = x.ndim-3
    centered = x-np.expand_dims(x.mean(axis=axis),axis)
    den = np.mean(centered**2,axis=axis)+1e-30
    results = []
    for lag in range(1,21):
        a = np.take(centered,np.arange(lag,x.shape[axis]),axis=axis)
        b = np.take(centered,np.arange(x.shape[axis]-lag),axis=axis)
        results.append(np.mean(a*b,axis=axis)/den)
    return np.stack(results,axis=-1)


def windows(xi, alpha, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for end in range(150,len(xi)+1,50):
        start = end-100
        x = xi[start:end]
        s = statistics(x,alpha)
        starts = rng.integers(0,100,size=(200,10))
        indices = ((starts[...,None]+np.arange(10))%100).reshape(200,100)
        boot = statistics(x[indices],alpha)
        ac = acf(x)
        ac_boot = acf(x[indices])
        names = ['mean','variance','cross_covariance','qk_correlation','coherent_noise_ratio']
        row = {'start':start+1,'end':end,'acf_lags':list(range(1,21)),
               'acf':ac,'acf_ci95':np.quantile(ac_boot,[.025,.975],axis=0)}
        for name,v,b in zip(names,s,boot):
            row[name] = v
            row[name+'_ci95'] = np.quantile(b,[.025,.975],axis=0)
        rows.append(row)
    return rows


def analyze_modes(xi, ab_steps, alpha, eta, seed, probes):
    window_rows = windows(xi,alpha,seed+713)
    if len(xi)<200:
        return {'windows':window_rows,'scope':'insufficient budget for forecasts'}
    early = statistics(xi[:150],alpha)
    m,var,cov,_,_ = early
    initial = np.asarray(ab_steps[149])
    c = 1-alpha
    forecasts = []
    for step in range(160,len(xi)+1,10):
        n = step-150
        scale = (1-c**n)/(1-c)
        noise_scale = (1-c**(2*n))/(1-c*c)
        mu = c**n*initial+eta*m*scale
        pred_q2k2 = mu*mu+eta**2*var*noise_scale
        pred_qk = mu[:,0]*mu[:,1]+eta**2*cov*noise_scale
        actual_ab = np.asarray(ab_steps[step-1])
        forecasts.append({'step':step,'predicted_a2_b2':pred_q2k2,
                          'predicted_ab':pred_qk,'actual_a2_b2':actual_ab**2,
                          'actual_ab':actual_ab[:,0]*actual_ab[:,1]})
    late_forecasts = [f for f in forecasts if f['step']>len(xi)-100]
    pq2k2 = np.mean([f['predicted_a2_b2'] for f in late_forecasts],axis=0)
    aq2k2 = np.mean([f['actual_a2_b2'] for f in late_forecasts],axis=0)
    pqk = np.mean([f['predicted_ab'] for f in late_forecasts],axis=0)
    aqk = np.mean([f['actual_ab'] for f in late_forecasts],axis=0)
    errs = np.maximum(np.max(np.abs(pq2k2-aq2k2)/(np.abs(aq2k2)+1e-12),axis=1),
                      np.abs(pqk-aqk)/(np.abs(aqk)+1e-12))
    early_ratio = np.asarray(window_rows[0]['coherent_noise_ratio'])
    late_ratio = np.asarray(window_rows[-1]['coherent_noise_ratio'])
    qk_correlation = statistics(xi[50:],alpha)[3]
    ac_early,ac_late = np.abs(window_rows[0]['acf']),np.abs(window_rows[-1]['acf'])
    head_rows = []
    valid_probes = [p for p in probes if p['step']>=50 and 'residual' in p]
    for h in range(H):
        peak = max(valid_probes,key=lambda p:p['head_Mabs'][h])
        short_corr = max(ac_early[h].max(),ac_late[h].max())
        crossing = bool(np.all(early_ratio[h]>1) and np.all(late_ratio[h]<1) and
                        np.all(late_ratio[h]<early_ratio[h]))
        observed_crossings = [w['end'] for w in window_rows
                              if np.all(np.asarray(w['coherent_noise_ratio'])[h]<1)]
        conditions = {'residual':bool(peak['residual'][h]<=.3),
                      'explained_change':bool(peak['explained_change'][h]>=.7),
                      'qk_correlation':bool(qk_correlation[h]>=.8),
                      'phase_crossing':crossing, 'short_correlation':bool(short_corr<=.2),
                      'forecast':bool(errs[h]<=.3)}
        head_rows.append({'head':h,'peak_step':peak['step'],
                          'peak_Mabs':peak['head_Mabs'][h], 'peak_residual':peak['residual'][h],
                          'peak_explained_change':peak['explained_change'][h],
                          'qk_correlation':qk_correlation[h],
                          'early_ratio':early_ratio[h],'late_ratio':late_ratio[h],
                          'first_update_only_crossing_window_end':observed_crossings[0] if observed_crossings else None,
                          'max_abs_acf_early_late':short_corr,'forecast_relative_error':errs[h],
                          'conditions':conditions,'C1_all_conditions':all(conditions.values())})
    return {'windows':window_rows,'fit_window':[1,150],
            'fit_mean':m,'fit_variance':var,'fit_cross_covariance':cov,
            'forecast':forecasts,'heads':head_rows,
            'note':'Forecasts condition on independent, stationary innovations and are diagnostics when assumptions fail.'}
