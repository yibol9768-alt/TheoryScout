"""仅确定性故障夹具，检查换行修复、门控、实际进程退出与退出观察。"""
import copy
import json
import os
import subprocess
import sys
import numpy as np
from common_v2 import dump, sha, read, usage
from protocol_v2 import publish, verify_accepted
from analyze_v2 import make_report, csv_text

def observer_receipt(data,mh,rc=0):
    dump(data/'SUPERVISOR_EXIT.json',dict(launch_status_sha256=sha(data/'launch_status.json'),
        normal=True,wall=2.,cpu=1.,manifest=mh))
    dump(data/'EXIT_OBSERVED.json',dict(returncode=rc,killed=rc!=0,manifest=mh,wall=2.,cpu=2.,
        self_cpu=1.,launch_status_sha256=sha(data/'launch_status.json'),
        receipt_sha256=sha(data/'SUPERVISOR_EXIT.json')))

def extra_checks(T):
    def newline():
        text=csv_text([dict(a='x',b=.2)])
        T.require('\r' not in text and text=='a,b\nx,0.2\n')
        T.require((T.BASE/'good/analysis/curves.csv').read_bytes().decode()==
                  csv_text(read(T.BASE/'good/analysis/report.json')['curves']))
    T.check('LF_bytes_equal_replayed_CSV',newline)

    def no_observer():
        data,cfg,mh=T.fixture('observer_missing')
        T.require(publish(data,mh,cfg,T.GOOD,2.,1.,1.)['confirmation_valid'])
        T.rejects(lambda:verify_accepted(data,mh,cfg))
    T.check('accepted_without_observed_supervisor_exit_rejected',no_observer)

    def killed_supervisor():
        data,cfg,mh=T.fixture('supervisor_killed')
        T.require(publish(data,mh,cfg,T.GOOD,2.,1.,1.)['confirmation_valid'])
        p=subprocess.run([sys.executable,'-B','-c','import os,signal;os.kill(os.getpid(),signal.SIGKILL)'])
        T.require(p.returncode==-9); observer_receipt(data,mh,p.returncode)
        T.rejects(lambda:verify_accepted(data,mh,cfg))
    T.check('actual_supervisor_SIGKILL_after_candidate_rejected',killed_supervisor)

    for stage in ['runner','analysis']:
        def actual_kill(stage=stage):
            data,cfg,mh=T.fixture('actual_kill_'+stage)
            # A complete PENDING exists before the child's actual SIGKILL.
            p=subprocess.run([sys.executable,'-B','-c','import os,signal;os.kill(os.getpid(),signal.SIGKILL)'])
            states=copy.deepcopy(T.GOOD)
            states[0 if stage=='runner' else 1].update(returncode=p.returncode,killed=True)
            T.require(p.returncode==-9)
            T.require(not publish(data,mh,cfg,states,2.,1.,1.)['confirmation_valid'])
            T.require(not (data/'ACCEPTED.json').exists())
        T.check('actual_'+stage+'_SIGKILL_after_complete_evidence',actual_kill)

    def gates():
        data,cfg,mh=T.fixture('gates')
        # Fixed fabricated new recall is matched; replacing target only in analysis
        # must suppress directional claims even though old differences are positive.
        original=make_report(data,mh,cfg)
        T.require(original['decisions']['primary_all3_directions'])
        T.require(len(original['all_completed'])==3*2*3*4)
        T.require(len(original['matching'])==2*3*(4+3))
        # An artificial stricter band cannot create a match by old-task significance.
        strict=copy.deepcopy(cfg);strict['match_band']=0.
        report=make_report(data,mh,strict)
        T.require(not any(r['direction_supported'] for r in report['old_pairs']))
        T.require(not report['decisions']['coefficient_confirmation'])
        for r in original['scaled_residuals']:
            if r['obs'] is not None:
                T.require(abs(r['scaled_residual']-r['d']*(r['obs']-r['R_infty']))<1e-13)
    T.check('complete_matching_errors_curve_coverage_and_gate_enforced',gates)

    def missing():
        data,cfg,mh=T.fixture('complete_disclosure')
        (data/'eval_128_1.npz').unlink()
        report=make_report(data,mh,cfg)
        T.require(len(report['all_completed'])==2*2*3*4)
        T.require(not report['decisions']['primary_all3_directions'])
        T.require(not report['decisions']['primary_all7_matching'])
    T.check('missing_dataset_retains_all_other_readable_results_no_confirmation',missing)

    def negative_budget():
        T.fault('negative_budget',lambda p:None,children=-1.)
    T.check('negative_CPU_rejected',negative_budget)

    def direct_worker():
        for module in ['run_v2.py','analyze_v2.py']:
            # Not a formal invocation; verify fails before any RNG construction.
            p=subprocess.run([sys.executable,'-B',str(T.OUT/module)],capture_output=True)
            T.require(p.returncode!=0 and b'supervisor only' in p.stderr)
        T.require(not (T.OUT/'formal').exists())
    T.check('direct_workers_reject_without_draws',direct_worker)
