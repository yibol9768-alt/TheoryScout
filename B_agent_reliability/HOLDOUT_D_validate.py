"""Read-only mathematical/input verification; first run creates only a new D audit.

No network, no runner/client imports, no actual or mock execution batches.
Run with python -B under the frozen systemd resource limits.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    design=json.loads((ROOT/'HOLDOUT_D_DESIGN.json').read_text())
    audit=json.loads((ROOT/'HOLDOUT_D_SEED_AUDIT.json').read_text())
    results=json.loads((ROOT/'HOLDOUT_D_OFFLINE_RESULTS.json').read_text())
    md=(ROOT/'HOLDOUT_D_PREREG.md').read_text()
    changed=[p for p,h in audit['source_hashes'].items() if digest(ROOT/p)!=h]
    assert not changed, changed
    assert design['N']==144 and design['max_length']==64
    assert design['seeds']==list(range(116064001,116064145))
    assert len(set(design['seeds'])|set(design['auxiliary_seeds'].values()))==150
    assert design['budget']['successful_return_hard_limit']==9000
    assert design['budget']['http_attempt_hard_limit']==12000
    assert design['budget']['concurrency']==2
    assert design['engineering']['runner_delivered'] is False
    assert design['engineering']['execution_ready'] is False
    assert audit['collisions']==[] and audit['prefix_cases']==576
    effective={}
    for L in (8,16,32,64):
        p=design['predictions'][str(L)]
        q=(1-34/1463)**min(L,16)*(1-14/1339)**max(L-16,0)
        assert abs(p['S_D']-q)<1e-14
        # Independent direct combinatorial calculation, not log-gamma PMF.
        pmf=np.array([math.comb(144,k)*q**k*(1-q)**(144-k) for k in range(145)])
        assert abs(pmf.sum()-1)<1e-13
        ci=[int(np.searchsorted(pmf.cumsum(),alpha)) for alpha in (.005,.995)]
        assert ci==p['count99_D']
        effective[L]=[max(ci[0],math.ceil(144*(q-.10))),min(ci[1],math.floor(144*(q+.10)))]
    assert effective=={8:[107,130],16:[85,113],32:[70,97],64:[46,74]}
    pa=np.array(design['comparison']['cells_A'])
    pd=np.array(design['comparison']['cells_D'])
    weights=np.array(design['comparison']['weights_log_D_over_A'])
    assert np.max(np.abs(np.log(pd/pa)-weights))<1e-14
    assert abs(pa.sum()-1)<1e-14 and abs(pd.sum()-1)<1e-14
    u=np.ceil(weights/.001).astype(int)
    assert u.tolist()==[419,351,-455,-352,78]
    assert (u*.001>=weights).all()
    # Independent polynomial-power FFT check of the repeated-shift DP cutoff.
    shift=u-u.min()
    degree=int(144*shift.max())
    fft_len=1<<(degree.bit_length())
    poly=np.zeros(fft_len)
    for j,p in zip(shift,pa):
        poly[j]+=p
    coeff=np.fft.irfft(np.fft.rfft(poly)**144,n=fft_len)[:degree+1]
    assert abs(coeff.sum()-1)<1e-11
    assert coeff.min()>-1e-12
    cutoff=design['comparison']['cutoff']['critical']
    assert cutoff==.541
    index=round(cutoff/.001)-144*int(u.min())
    tail=float(coeff[index+1:].sum())
    previous_tail=float(coeff[index:].sum())
    assert abs(tail-design['comparison']['cutoff']['null_tail_upper'])<1e-10
    assert tail<.025-1e-12 and previous_tail>.025
    assert abs(results['protocol_D']['cost_mean']-design['budget']['expected_D'])<5
    assert abs(results['protocol_A']['cost_mean']-design['budget']['expected_A'])<5
    assert .94<results['protocol_D']['rates']['llr']<.97
    assert .020<results['protocol_A']['rates']['llr']<.026
    assert results['protocol_D']['cap_exceed_count']==results['protocol_A']['cap_exceed_count']==0
    # Check frozen displayed counts/weights and key decisions in the Chinese document.
    for text in ('T>0.541','[107,130]','[85,113]','[70,97]','[46,74]',
                 '9000','12000','信息不足','HTTP408','setsid nohup systemd-run',
                 '0.4184580058230245','0.3500786341675218','0.07713858608750793'):
        assert text in md,text
    # Offline source must not contain any actual network/client invocations.
    offline=(ROOT/'HOLDOUT_D_offline.py').read_text()
    for forbidden in ('urllib','requests.','socket.','Client(','subprocess.'):
        assert forbidden not in offline
    frozen=ROOT/'HOLDOUT_D_PREREG.sha256'
    manifest_checked=False
    if frozen.exists():
        for line in frozen.read_text().splitlines():
            h,name=line.split('  ',1)
            assert name.startswith('HOLDOUT_D_') and '/' not in name
            assert digest(ROOT/name)==h,name
        manifest_checked=True
    output=dict(verified_utc=datetime.now(timezone.utc).isoformat(),status='PASS',
                existing_input_files_unchanged=len(audit['source_hashes']),
                independent_binomial_counts=True,effective_thresholds=effective,
                independent_fft_lattice_tail=tail,previous_lattice_tail=previous_tail,
                expected_null_upper=design['comparison']['cutoff']['null_tail_upper'],
                seed_and_prefix_audit=True,budget_and_power_consistency=True,
                no_runner_no_network_no_batch=True,
                llm_calls=0,actual_quota_probes=0,
                cap_exceed_probability_one_sided95_upper_for_zero_in_100000=1-.05**(1/100000),
                design_and_prereg_consistent=True,
                note='Manifest is created after this first audit; repeat validator also checks every frozen digest.')
    report=ROOT/'HOLDOUT_D_VALIDATION.json'
    if not report.exists():
        with report.open('x') as f:
            json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write('\n')
    print(json.dumps(dict(status='PASS',input_files_unchanged=len(audit['source_hashes']),
                         manifest_checked=manifest_checked,lattice_tail=tail,
                         effective_thresholds=effective),ensure_ascii=False))


if __name__=='__main__':
    main()
