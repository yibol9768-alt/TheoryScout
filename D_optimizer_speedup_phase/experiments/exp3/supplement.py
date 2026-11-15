"""Post-freeze algebra audit; no prediction, parameter or threshold changes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import datetime
import hashlib
import json
import resource
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def main():
    started=time.monotonic()
    props=subprocess.check_output(['systemctl','--user','show',sys.argv[1],'-p','MemoryMax','-p','MemorySwapMax'],text=True)
    assert 'MemoryMax=2147483648' in props and 'MemorySwapMax=0' in props
    folder=HERE/'formal'
    hold=json.loads((folder/'holdout.json').read_text())
    training=json.loads((folder/'training.json').read_text())
    rows=hold+[s for r in training for s in r['tests']]
    maximum_q=maximum_u=maximum_risk=maximum_formula=0.
    for row in rows:
        n=row['batches']
        with np.load(folder/row['file']) as z:
            mu=z['mean_full'];muu=z['u_mean_full'];b=z['B']
            qout=z['rowcov_full'].sum(axis=0)
            uout=z['u_rowcov_full'].sum(axis=0)
            maximum_q=max(maximum_q,float(np.max(np.abs(qout-n/(n-1)*(np.eye(4)-mu.T@mu)))))
            maximum_u=max(maximum_u,float(np.max(np.abs(uout-n/(n-1)*(b@z['gram_mean']@b-muu.T@muu)))))
            pred=z['prediction']
            maximum_formula=max(maximum_formula,float(np.max(np.abs(np.array(row['predicted_noise_output'])-(np.eye(4)-pred.T@pred)))))
            st=row['metrics']['full']
            reconstructed=float(np.sum(z['lam'][:,None]*mu*mu))+(n-1)/n*st['noise_risk']
            maximum_risk=max(maximum_risk,abs(reconstructed-st['total_risk']))
    assert max(maximum_q,maximum_u,maximum_risk,maximum_formula)<1e-10
    latex=[]
    for file in sorted((ROOT/'lemmas_v4').glob('L*.tex')):
        source=file.read_text()
        level=0
        for i,ch in enumerate(source):
            if i and source[i-1]=='\\':
                continue
            if ch=='{': level+=1
            if ch=='}': level-=1
            assert level>=0,str(file)
        assert level==0,str(file)
        assert source.count('\\begin{problem}')==source.count('\\end{problem}')==1
        assert source.count('\\[')==source.count('\\]')
        assert source.count('$')%2==0
        latex.append(dict(file=str(file.relative_to(ROOT)),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),static_pass=True))
    result=dict(kind='post-freeze algebra and static audit, not a new predictive test',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),states=len(rows),
        q_output_identity_max=maximum_q,u_output_identity_max=maximum_u,
        predicted_output_identity_max=maximum_formula,risk_moment_identity_max=maximum_risk,
        latex=latex,latex_compiled=False,qed_submitted=False,scope_properties=props,
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    analysis=json.loads((HERE/'analysis.json').read_text())
    result['total_program_compute_seconds_including_supplement']=analysis['total_compute_seconds']+result['elapsed_seconds']
    full=[g['parts']['full'] for g in analysis['groups']]
    online=[g['metrics'] for g in analysis['online_groups']]
    def ranges(stats):
        return dict(error_max=max(s['error'] for s in stats),
            gain_min=min(s['gain'] for s in stats),gain_max=max(s['gain'] for s in stats),
            snr_min=min(s['snr'] for s in stats),
            risk_noise_min=min(s['noise_risk_ratio'] for s in stats),
            risk_noise_max=max(s['noise_risk_ratio'] for s in stats),
            output_noise_max=max(s['noise_output_error'] for s in stats),
            unresolved=sum(not s['resolved'] for s in stats),
            drift_violations=sum(s['error']>.25 for s in stats),
            drift_confirmed_violations=sum(s['error']>.25 and s['resolved'] for s in stats),
            gain_violations=sum(abs(s['gain']-1)>.2 for s in stats),
            noise_risk_violations=sum(not .7<=s['noise_risk_ratio']<=1.3 for s in stats),
            noise_output_violations=sum(s['noise_output_error']>.30 for s in stats))
    result['local_full_summary']=ranges(full)
    result['online_summary']=ranges(online)
    result['unresolved_local_parts']=[dict(alpha=g['alpha'],n=g['n'],amplitude=g['amplitude'],part=part,snr=s['snr'])
        for g in analysis['groups'] for part,s in g['parts'].items() if not s['resolved']]
    result['amplitude_summary']=[]
    for amplitude in (1.,.25,.0625):
        selected=[r for r in hold if r['amplitude']==amplitude]
        error=sum(r['metrics']['full']['prediction_error'] for r in selected)
        scalar=sum(r['metrics']['full']['scalar_error'] for r in selected)
        energy=sum(r['metrics']['full']['prediction_energy'] for r in selected)
        result['amplitude_summary'].append(dict(amplitude=amplitude,
            matrix_error=float(np.sqrt(max(0.,error)/energy)),scalar_error=float(np.sqrt(max(0.,scalar)/energy))))
    result['elapsed_seconds']=time.monotonic()-started
    result['total_program_compute_seconds_including_supplement']=analysis['total_compute_seconds']+result['elapsed_seconds']
    (HERE/'supplement.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
