"""E1–E3负向授权测试；从不填写任何已审凭据。"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from engine import Engine, Paused, IntegrityError
from evidence import FIRST, RELEASE, RUNNER, version, smoke_snapshot, validate_smoke
from protocol import ROOT, dump, sha, Task
from transport import MockServer, HTTPTransport, HTTPQuota
from run import execute
from score import score, read_wal, rebuild, resolved_counts, mechanism, c_replay, cell, primary_pass
from test_mock import BASE, Clock

S=FIRST[0]
def good():return dict(five_hour=0.,week=0.)

class Amend2Tests(unittest.TestCase):
    def eng(self,name,fixture=None):return Engine(BASE/('v2_'+name),verify=False,fixture=fixture,clock=Clock())
    def smoke(self,name):
        e=self.eng(name);execute(e,MockServer(policy='fatal_at_8'),good,mock=False)
        self.assertEqual(sum(t.terminal for t in e.tasks.values()),2)
        return e
    def test_E1_shell_and_direct_sentinels(self):
        self.assertFalse(RELEASE.exists())
        binpath=BASE/'HOLDOUT_D_sentinel_bin';binpath.mkdir()
        shim=binpath/'python3'
        shim.write_text('#!/bin/sh\nexec '+sys.executable+' -B '+str(RUNNER/'HOLDOUT_D_cli_sentinel.py')+' "$@"\n');shim.chmod(0o700)
        env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH'])
        cases=[['--live'],['--liv'],['--li'],['--l'],['--live=true'],['--liv=true'],
               ['--live','--entry-check'],['--live','--quota-url','unused'],
               ['--live','--test-unfrozen'],['--live','--environment-only']]
        evidence=[]
        for args in cases:
            for entry in ('shell','python'):
                cmd=([str(RUNNER/'launch.sh')] if entry=='shell' else [sys.executable,'-B',str(RUNNER/'HOLDOUT_D_cli_sentinel.py')])+args
                p=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=10)
                self.assertEqual(p.returncode,2,p.stdout+p.stderr)
                self.assertIn('BLOCKED_BEFORE_CLIENTS',p.stdout)
                self.assertNotIn('CLIENT_OR_ENGINE_CONSTRUCTED',p.stderr)
                evidence.append(dict(entry=entry,args=args,exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
        (BASE/'HOLDOUT_D_ENTRY_SENTINELS.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
        self.assertFalse((RUNNER/'HOLDOUT_D_live').exists())
    def test_E1_direct_engine_and_client_constructors(self):
        for kw in ({'mode':'live'},{'mode':'live','live':True}):
            with self.assertRaisesRegex(ValueError,'live_requires_explicit|live_release_file_missing'):
                Engine(BASE/'v2_forbidden_live',**kw)
        with self.assertRaisesRegex(ValueError,'live_release_file_missing'):HTTPTransport(live=True)
        with self.assertRaisesRegex(ValueError,'live_release_file_missing'):HTTPQuota('unused',live=True)
        self.assertFalse((BASE/'v2_forbidden_live').exists())
    def test_E2_direct_transport_requires_guarded_reservation(self):
        tr=object.__new__(HTTPTransport);tr.testing=False
        with patch('transport.require_live_release',return_value=None):
            with self.assertRaisesRegex(ValueError,'requires_engine_reservation'):tr.send(Task(S+2).body())
    def test_E2_snapshot_covers_all_first_two_records(self):
        e=self.smoke('snapshot');snap=smoke_snapshot(e.wal.events,version());template=e.smoke_review_template()
        lines=e.wal.path.read_bytes().splitlines(keepends=True)
        self.assertEqual(snap['wal_sha256'],sha(b''.join(lines[:snap['wal_seq']])))
        for seed in FIRST:
            ev=[x for x in e.wal.events if x['kind']=='reserve' and x['seed']==seed]
            self.assertEqual(len(ev),11)
            self.assertTrue(all(x['seq']<=snap['wal_seq'] for x in ev))
            self.assertTrue(all(x['seq']<=snap['wal_seq'] for x in e.wal.events if x['kind'] in ('response','action')))
        self.assertFalse(template['approved']);self.assertFalse(any(template['checked_items'].values()))
        self.assertEqual(template['reviewer'],'');self.assertFalse(e.smoke)
        e.close()
    def test_E2_old_prefix_missing_fields_wrong_snapshot(self):
        e=self.smoke('invalid_credentials');template=e.smoke_review_template()
        # 所有材料保持 approved=False；故意错误的绑定须先被具体拒绝。
        for key,value in [('wal_digest',e.wal.events[0]['digest']),('wal_seq',1),('wal_sha256','0'*64),('runner_version','old')]:
            z=copy.deepcopy(template);z['snapshot'][key]=value
            with self.assertRaisesRegex(ValueError,'snapshot_mismatch'):validate_smoke(z,e.wal.events,version())
        for field in template:
            z=copy.deepcopy(template);del z[field]
            with self.assertRaisesRegex(ValueError,'schema'):validate_smoke(z,e.wal.events,version())
        for bad in ({},['mock'],{k:1 for k in template['checked_items']}):
            z=copy.deepcopy(template);z['checked_items']=bad
            with self.assertRaisesRegex(ValueError,'checklist_schema'):validate_smoke(z,e.wal.events,version())
        for change in ('task_identity','terminal','bool_seq'):
            z=copy.deepcopy(template)
            if change=='task_identity':z['snapshot']['tasks'][0]['seed']=S+2
            elif change=='terminal':z['snapshot']['tasks'][0]['c']=7
            else:z['snapshot']['wal_seq']=True
            with self.assertRaisesRegex(ValueError,'snapshot_mismatch'):validate_smoke(z,e.wal.events,version())
        with self.assertRaisesRegex(ValueError,'smoke_not_finished'):smoke_snapshot(e.wal.events[:2],version())
        with self.assertRaises(Paused):e.approve_smoke(template)
        self.assertFalse(e.smoke);e.close()
    def test_E2_direct_live_send_and_reserve_blocked(self):
        # 仅投影 live 门分支，替换入口文件检查为无操作；无凭据/客户端/已审事件。
        e=self.smoke('direct_gate')
        class Sentinel:
            testing=False
            def send(self,body):raise AssertionError('THIRD_TASK_SENT')
        before=e.attempts
        with patch.object(Engine,'mode',new=property(lambda self:'live')),patch('engine.require_live_release',return_value=None):
            for seed in (S+2,S+50,S+143):
                with self.assertRaisesRegex(Paused,'human_smoke_release_required'):e.reserve(seed,good)
                with self.assertRaisesRegex(Paused,'human_smoke_release_required'):e.request(seed,Sentinel(),good)
            with self.assertRaisesRegex(ValueError,'mock_scheduler_forbidden_live'):execute(e,Sentinel(),good,mock=True)
        self.assertEqual(e.attempts,before);self.assertFalse(e.smoke);e.close()
    def test_E2_unfinished_live_third_send_blocked(self):
        e=self.eng('unfinished_gate')
        with patch.object(Engine,'mode',new=property(lambda self:'live')),patch('engine.require_live_release',return_value=None):
            with self.assertRaisesRegex(Paused,'human_smoke_release_required'):e.reserve(S+2,good)
        self.assertEqual(e.attempts,0);e.close()
    def test_E2_recovery_rejects_unapproved_or_stale_events(self):
        for name,stale in [('unapproved',False),('stale',True)]:
            e=self.smoke('recovery_'+name);z=e.smoke_review_template()
            if stale:z['snapshot']['wal_digest']=e.wal.events[0]['digest']
            e.wal.add('smoke_approved',approval=z) # 明示未批准的非法事件，非伪造已审凭据。
            p=e.wal.out;e.close()
            for _ in range(2):
                with self.assertRaisesRegex(IntegrityError,'invalid_recovered_smoke'):Engine(p,verify=False)
    def test_E2_mock_cannot_use_real_transport(self):
        e=self.eng('transport_gate')
        class Sentinel:
            testing=False
            def send(self,body):raise AssertionError('real transport used')
        with self.assertRaisesRegex(Paused,'mock_cannot_use_live_transport'):e.request(S,Sentinel(),good)
        self.assertEqual(e.attempts,0);e.close()
    def test_E3_boundaries_actual_WAL(self):
        for c,expected in [(7,0),(8,1),(15,1),(16,2),(31,2),(32,3),(63,3),(64,4)]:
            e=self.eng('boundary_'+str(c));server=MockServer()
            while not e.tasks[S].terminal:
                t=e.tasks[S]
                if t.step==c:
                    r,b=e.reserve(S,good);e.receive(r['call'],200,bad_response(b))
                else:e.request(S,server,good)
            self.assertEqual(e.tasks[S].step,c)
            result=score(e.wal.path);self.assertEqual(result['five_cells'],[int(i==expected) for i in range(5)])
            self.assertIsNone(result['T']);self.assertFalse(result['G1']);self.assertIsNone(result['G3'])
            self.assertEqual(result['decision'],'信息不足')
            (e.wal.out/'HOLDOUT_D_SCORE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');e.close()
    def test_E3_service_stop_missing(self):
        e=self.eng('service_score');e.request(S,MockServer([408]),good)
        z=score(e.wal.path);self.assertEqual(z['five_cells'],[0]*5)
        self.assertEqual(z['prefix_bounds']['8']['U'],144);self.assertIsNone(z['T']);e.close()
    def test_E3_strict_primary_boundary_and_calibration(self):
        import math
        self.assertFalse(primary_pass(.541));self.assertTrue(primary_pass(math.nextafter(.541,math.inf)))
        self.assertFalse(primary_pass(math.nextafter(.541,-math.inf)))
        e=self.eng('calibration');z=score(e.wal.path)
        self.assertEqual([v['effective_acceptance'] for v in z['prefix_bounds'].values()],[[107,130],[85,113],[70,97],[46,74]])
        self.assertFalse(z['G0']);self.assertFalse(z['G1']);self.assertIsNone(z['G2']);self.assertIsNone(z['G3']);e.close()
    def test_E3_budget_interrupted_review_resolved(self):
        e=self.eng('budget_score',dict(success=8999));r,b=e.reserve(S,good);e.receive(r['call'],200,bad_response(b))
        with self.assertRaises(Paused):e.reserve(S,good)
        z=score(e.wal.path);self.assertEqual(e.tasks[S].counts[0],[1,0,1,0,0])
        self.assertEqual(z['resolved_per_task'][0],[[0]*5,[0]*5]);self.assertEqual(z['five_cells'],[0]*5)
        self.assertIsNone(z['T']);self.assertEqual(z['prefix_bounds']['8']['success_count_bound'],[0,144]);e.close()
    def test_E3_full_mock_score_and_stale_summary(self):
        e=self.eng('full_score');execute(e,MockServer(policy='fatal_at_8'),good,mock=True)
        (e.wal.out/'HOLDOUT_D_SUMMARY.json').write_text('{"terminal":0,"successful":999999}')
        z=score(e.wal.path);self.assertEqual(z['five_cells'],[0,144,0,0,0]);self.assertEqual(z['T'],144*0.3500786341675218)
        self.assertTrue(z['G1']);self.assertFalse(z['G0']);self.assertIsNone(z['G2']);self.assertIsNone(z['G3'])
        self.assertEqual(z['decision'],'信息不足')
        (e.wal.out/'HOLDOUT_D_SCORE.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n');e.close()
    def test_E3_C_replay_all_40(self):
        z=c_replay();self.assertEqual((z['files'],z['actions'],z['full64_cells']),(40,1288,[5,2,1,2,10]))
        (BASE/'HOLDOUT_D_C_REPLAY.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')
    def test_E3_corrupt_and_torn_WAL_rejected_readonly(self):
        e=self.eng('corrupt_score');e.request(S,MockServer(),good);p=e.wal.path;e.close();raw=p.read_bytes()
        for suffix in (b'{',b'{}\n'):
            p.write_bytes(raw+suffix)
            with self.assertRaises((ValueError,KeyError)):score(p)
            self.assertEqual(p.read_bytes(),raw+suffix)
    def test_E3_bootstrap_paired_exact_reference(self):
        import numpy as np
        x=[]
        for i in range(144):x.append([[16,1,3,2,0],[20+i%10,1,3,3,2]])
        z=mechanism(x);rng=np.random.default_rng(116990005);a=np.asarray(x)[rng.integers(0,144,size=(10000,144))].sum(axis=1)
        diffs=np.column_stack((a[:,0,1]/a[:,0,0]-a[:,1,1]/a[:,1,0],a[:,1,4]/a[:,1,3]-a[:,0,4]/a[:,0,3]))
        self.assertEqual(z['valid'],10000);self.assertEqual(z['intervals'],np.quantile(diffs,[.025,.975],axis=0,method='linear').T.tolist())
        self.assertTrue(z['passed'])

def bad_response(body):
    status,raw,reason=MockServer().send(body);z=json.loads(raw);z['choices'][0]['message']['content']='not-json'
    return dump(z).encode()

if __name__=='__main__':unittest.main(verbosity=2)
