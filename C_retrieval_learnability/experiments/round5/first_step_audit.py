"""Reconstruct chosen first updates; distinguish trial calibration moves from actual rates."""
import experiment as c
from run import read_records,identifier

def main():
    rows=[]; max_error=0.; legacy_trial_error=0.
    for r in read_records(c.OUT/'main'/'raw')+read_records(c.OUT/'holdout'/'raw'):
        q=r['config']; A,p=c.data(q['d'],q['seed'],q['n']); beta=16.
        b=c.root_bias(beta*(p['U']@p['V'].T),A)[0] if q['init']=='star' else c.np.zeros(len(A))
        p['b' if q['mode']=='b' else 'theta']=b.copy() if q['mode']=='b' else -b/beta
        if q['schedule']=='train': p['a']=c.np.array([c.np.log(beta)])
        old={k:v.copy() for k,v in p.items()}; _,g=c.obj(p,A,q['mode'],beta)
        c.Adam(p,r['rates']).update(p,g)
        beta1=float(c.np.exp(p['a'][0])) if q['schedule']=='train' else c.beta_path(q['schedule'],1/c.T)
        _,shift0=c.effective(old,q['mode'],beta); _,coord=c.effective(p,q['mode'],beta); _,full=c.effective(p,q['mode'],beta1)
        moves={k:float(c.np.sqrt(c.np.mean((p[k]-old[k])**2))) for k in p}
        cm=float(c.np.sqrt(c.np.mean((coord-shift0)**2))); fm=float(c.np.sqrt(c.np.mean((full-shift0)**2)))
        max_error=max(max_error,abs(cm-r['calibration']['first_coordinate_move_rms']),abs(fm-r['calibration']['first_full_threshold_move_rms']))
        if q['rate']=='legacy': legacy_trial_error=max(legacy_trial_error,abs(moves['b']-r['calibration']['moves']['b']))
        rows.append(dict(id=identifier(q),actual_coordinate_moves=moves,actual_bias_effective_coordinate_rms=cm,actual_full_effective_threshold_rms=fm,beta1=beta1,calibration_trial_moves=r['calibration']['moves']))
    result=dict(runs=len(rows),max_first_step_record_error=max_error,max_legacy_trial_b_move_difference=legacy_trial_error,
                interpretation='calibration.moves is the initial trial from calibrate(); for legacy bias lr is subsequently overridden. first_coordinate_move_rms and first_full_threshold_move_rms already use the actual chosen rates and are correct. Actual moves for every group are recorded here.',rows=rows)
    assert max_error<1e-12
    c.save(c.OUT/'first_step_audit.json',result); print(c.json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__': main()
