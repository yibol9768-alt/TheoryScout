"""Run all week-1 studies without API calls.

OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  /home/liuyibo/venvs/theory/bin/python experiments/run_week1.py
"""
import os
import sys
import signal
import platform
import resource
import traceback
from datetime import datetime, timezone
from pathlib import Path

from common import OUT, ROOT, save, Budget, sha, rss_bytes
import numpy as np
import scipy
import torch
torch.set_num_threads(3)
torch.set_num_interop_threads(1)
import matplotlib
from training import run_training, gradient_checks, constructions
from e0 import run_e0


def main():
    budget = Budget()
    resource.setrlimit(resource.RLIMIT_AS,(int(3.5*1024**3),int(3.5*1024**3)))
    def alarm(*_):
        raise TimeoutError('60-minute hard execution stop')
    signal.signal(signal.SIGALRM,alarm)
    signal.alarm(3600)
    manifest = {'started_utc':datetime.now(timezone.utc).isoformat(),
                'executable':sys.executable,'python':sys.version,'platform':platform.platform(),
                'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,
                'matplotlib':matplotlib.__version__,
                'threads':{k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
                'torch_num_threads':torch.get_num_threads(),'torch_num_interop_threads':torch.get_num_interop_threads(),
                'compute_device':'CPU','external_API_calls':0,
                'address_space_limit_bytes':int(3.5*1024**3),'RSS_stop_bytes':3_000_000_000,
                'seed_e0':20261105,'seeds_training':[11,23,47],
                'formulation_sha256':sha(ROOT/'FORMULATION.md'),
                'preregistration_sha256':sha(ROOT/'RESULTS_WEEK1.md'),
                'source_sha256':{p.name:sha(p) for p in sorted((ROOT/'experiments').glob('*.py'))},
                'status':'running','completed_e1':[],'completed_e2':[]}
    save(OUT/'preregistration.json',{'report_before_execution':(ROOT/'RESULTS_WEEK1.md').read_text(),
                                    'manifest_at_start':manifest.copy()})
    save(OUT/'manifest.json',manifest)
    try:
        checks = gradient_checks()
        print('Gradient checks passed (finite differences and full CPU autograd)',flush=True)
        calibration = run_training(11,'isotropic','muon0',.03,'base',50,budget,
                                   tag='calibration',diagnose=False)
        estimated = calibration['elapsed_seconds']/50*600*96*2+5*60+5*60
        steps,etas,e2_enabled = 600,[.01,.03,.1],True
        if estimated>55*60:
            steps = 400
            estimated = calibration['elapsed_seconds']/50*steps*96*2+10*60
        if estimated>55*60:
            etas = [.03]
            estimated = calibration['elapsed_seconds']/50*steps*48*2+10*60
        if estimated>55*60:
            e2_enabled = False
        manifest['calibration'] = {'steps':50,'seconds':calibration['elapsed_seconds'],
                                   'includes':'data generation, measurement, full-train CE, gzip directions, JSON output',
                                   'estimated_total_seconds_with_factor2_and_10min_reserve':estimated,
                                   'selected_steps':steps,'selected_etas':etas,'e2_enabled':e2_enabled}
        save(OUT/'manifest.json',manifest)
        print(f'Calibration {calibration["elapsed_seconds"]:.3f}s/50 steps; plan {steps} steps, etas={etas}',flush=True)
        run_e0(budget)
        manifest['e0_complete'] = True
        save(OUT/'manifest.json',manifest)
        selection = {eta:[] for eta in etas}
        for condition in ['isotropic','anisotropic']:
            for optimizer in ['muon0','muon09','adamw','signsgd']:
                for eta in etas:
                    for seed in [11,23,47]:
                        r = run_training(seed,condition,optimizer,eta,'base',steps,budget)
                        manifest['completed_e1'].append(r['run_id'])
                        if optimizer=='muon0':
                            selection[eta].append(float(np.mean([a['ce'] for a in r['train'] if 0<a['step']<=150])))
                        manifest['elapsed_seconds'] = budget.elapsed()
                        manifest['peak_RSS_bytes'] = rss_bytes()
                        save(OUT/'manifest.json',manifest)
                        print(f'E1 {len(manifest["completed_e1"]):02d} {r["run_id"]}: {r["elapsed_seconds"]:.2f}s',flush=True)
        selected_eta = min(etas,key=lambda e:(np.mean(selection[e]),e))
        manifest['e2_selection'] = {'rule':'mean full-training CE at steps 50,100,150 across 6 paired runs',
                                    'scores':{eta:float(np.mean(v)) for eta,v in selection.items()},
                                    'selected_eta':selected_eta}
        save(OUT/'manifest.json',manifest)
        print(f'E2 eta selected solely from first 150 steps: {selected_eta}',flush=True)
        if e2_enabled:
            for condition in ['isotropic','anisotropic']:
                for intervention in ['no_decay','layer_norm','head_norm','qk_clip']:
                    for seed in [11,23,47]:
                        r = run_training(seed,condition,'muon0',selected_eta,intervention,steps,budget,
                                         tag='e2',diagnose=False)
                        manifest['completed_e2'].append(r['run_id'])
                        manifest['elapsed_seconds'] = budget.elapsed()
                        manifest['peak_RSS_bytes'] = rss_bytes()
                        save(OUT/'manifest.json',manifest)
                        print(f'E2 {len(manifest["completed_e2"]):02d} {r["run_id"]}: {r["elapsed_seconds"]:.2f}s',flush=True)
        constructions()
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest['status'] = 'budget_stopped' if isinstance(exc,TimeoutError) else 'error'
        manifest['exception'] = repr(exc)
        manifest['traceback'] = traceback.format_exc()
        raise
    finally:
        manifest['elapsed_seconds'] = budget.elapsed()
        manifest['peak_RSS_bytes'] = rss_bytes()
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(OUT/'manifest.json',manifest)
        signal.alarm(0)
    print(f'Complete: {budget.elapsed():.1f}s, RSS {rss_bytes()/1024**2:.1f} MiB',flush=True)


if __name__=='__main__':
    main()
