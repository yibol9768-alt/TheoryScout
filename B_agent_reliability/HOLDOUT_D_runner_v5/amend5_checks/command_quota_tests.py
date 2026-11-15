"""固定命令的离线替身验收。真实 plus_quota.py 永不执行。"""
import json,os,subprocess,sys,time,unittest
from contextlib import contextmanager
from unittest.mock import patch
from harness import ROOT,OUT,resources,engine_module,thread_counts
engine=engine_module(ROOT/'HOLDOUT_D_runner_v5')
import evidence,run,transport
from protocol import sha
CMD=['python3','/home/liuyibo/auto-llm-driving/TheoryScout/tools/plus_quota.py']
S=116064001
class Clock:
    def __init__(self):self.t=1000.
    def __call__(self):return self.t

@contextmanager
def fake(output='12.5 34.5',mode='ok'):
    with patch.dict(os.environ,PATH=str(OUT/'command_bin')+os.pathsep+os.environ['PATH'],QUOTA_FAKE_OUTPUT=output,QUOTA_FAKE_MODE=mode),patch('transport.require_live_release',return_value=None):
        yield transport.CommandQuota(live=True)

class CommandQuotaTests(unittest.TestCase):
    def eng(self,name):
        c=Clock();e=engine.Engine(OUT/'command_engine'/name,verify=False,clock=c)
        self.addCleanup(e.close);return e,c
    def assert_failed(self,e,c,probe):
        with self.assertRaises(engine.Paused):e.quota(probe)
        q=e.wal.events[-1]
        self.assertEqual((q['kind'],q['ok'],q['reason'],q['checked']),('quota',False,'query_failed',c()))
        self.assertNotIn('five_hour',q);self.assertNotIn('week',q)
        self.assertTrue(e.need_probe);self.assertEqual(e.probe_retry,c()+300)
        attempts=e.attempts;server=transport.MockServer()
        with patch('transport.subprocess.run',side_effect=AssertionError('EARLY_OR_STALE_PROBE')):
            for at in (c(),c()+299.999):
                c.t=at
                with self.assertRaises(engine.Paused):e.request(S,server,probe)
        self.assertEqual((e.attempts,server.bodies),(attempts,[]))
        c.t=e.probe_retry
        with fake('1 2') as fresh:e.quota(fresh)
        self.assertFalse(e.need_probe)
        self.assertEqual(e.wal.events[-1]['five_hour'],.01)
    def test_01_constructor_requires_explicit_live_and_release(self):
        with patch('transport.subprocess.run',side_effect=AssertionError('EXECUTED')) as execute:
            with self.assertRaisesRegex(ValueError,'requires_--live'):transport.CommandQuota()
            with self.assertRaisesRegex(ValueError,'requires_--live'):transport.CommandQuota(live=1)
            with self.assertRaisesRegex(ValueError,'live_release_file_missing'):transport.CommandQuota(live=True)
            with self.assertRaises(TypeError):transport.CommandQuota(live=True,command=['anything'])
            execute.assert_not_called()
    def test_02_fixed_command_normal_and_observed_time(self):
        with fake() as probe:
            original=subprocess.run
            with patch('transport.subprocess.run',wraps=original) as execute:
                before=time.time();q=probe();after=time.time()
            execute.assert_called_once_with(CMD,capture_output=True,text=True,timeout=50,check=True)
        self.assertEqual((q['five_hour'],q['week']),(.125,.345))
        self.assertTrue(before<=q['observed_unix']<=after)
    def test_03_thresholds_and_inclusive_percent_range(self):
        for i,(output,allowed) in enumerate((('0 0',True),('79.999 89.999',True),('80 0',False),('0 90',False),('100 100',False))):
            with self.subTest(output=output),fake(output) as probe:
                e,c=self.eng('threshold_'+str(i))
                if allowed:e.quota(probe);self.assertFalse(e.need_probe)
                else:
                    with self.assertRaises(engine.Paused):e.quota(probe)
                    self.assertTrue(e.need_probe);self.assertEqual(e.probe_retry,c()+300)
                    self.assertNotIn('reason',e.wal.events[-1])
                self.assertEqual(e.wal.events[-1]['ok'],allowed)
    def test_04_malformed_outputs_pause_and_never_reuse_old_value(self):
        for i,output in enumerate(('', '1', '1 2 3', 'hello world', '{"five_hour":1,"week":2}', '1% 2%', 'nan 2', '1 inf')):
            with self.subTest(output=output):
                e,c=self.eng('malformed_'+str(i))
                with fake('1 2') as old:e.quota(old)
                e.need_probe=True
                with fake(output) as bad:self.assert_failed(e,c,bad)
    def test_05_out_of_range_outputs_are_query_failure(self):
        for i,output in enumerate(('-0.001 0','0 -1','100.001 0','0 101','-inf 0')):
            with self.subTest(output=output),fake(output) as probe:
                e,c=self.eng('range_'+str(i));self.assert_failed(e,c,probe)
    def test_06_nonzero_exit_with_valid_stdout_is_failure(self):
        e,c=self.eng('exit')
        with fake('1 2') as old:e.quota(old)
        e.need_probe=True
        with fake('1 2','exit') as probe:self.assert_failed(e,c,probe)
    def test_07_actual_fake_command_timeout_50_seconds(self):
        e,c=self.eng('timeout')
        with fake('1 2') as old:e.quota(old)
        e.need_probe=True;original=subprocess.run;observed=[]
        def timed(*args,**kwargs):
            self.assertEqual((args[0],kwargs['timeout']),(CMD,50));started=time.monotonic()
            try:return original(*args,**kwargs)
            finally:
                observed.append(time.monotonic()-started);c.t+=50
        with fake('1 2','timeout') as probe,patch('transport.subprocess.run',side_effect=timed) as execute:
            # assert_failed also performs a fresh recovery query; restore real run for it.
            with self.assertRaises(engine.Paused):e.quota(probe)
            execute.assert_called_once()
        self.assertTrue(49<=observed[0]<60,observed)
        self.assertEqual((c(),e.probe_retry,e.need_probe),(1050.,1350.,True))
        self.assertEqual(e.wal.events[-1]['reason'],'query_failed')
        server=transport.MockServer()
        with patch('transport.subprocess.run',side_effect=AssertionError('EARLY_PROBE')):
            c.t=1349.999
            with self.assertRaises(engine.Paused):e.request(S,server,probe)
        self.assertEqual((e.attempts,server.bodies),(0,[]))
        c.t=1350.
        with fake('2 3') as fresh:e.quota(fresh)
        (OUT/'timeout_evidence.json').write_text(json.dumps(dict(command=CMD,timeout=50,elapsed_seconds=observed[0],checked=1050,pause_until=1350,real_quota_executed=False),indent=2)+'\n')
    def test_08_release_rechecked_before_each_execution(self):
        with patch('transport.require_live_release',return_value=None):probe=transport.CommandQuota(live=True)
        with patch('transport.require_live_release',side_effect=ValueError('release_revoked')),patch('transport.subprocess.run') as execute:
            with self.assertRaisesRegex(ValueError,'release_revoked'):probe()
            execute.assert_not_called()
    def test_09_cli_switch_mutual_exclusion_and_live_gate(self):
        cases=[(['--quota-command','--quota-url','unused'],'not allowed with argument'),(['--quota-command'],'quota_command_requires_--live'),(['--live','--quota-command'],'live_release_file_missing'),(['--quota-command=anything'],'ignored explicit argument')]
        with patch('run.Engine',side_effect=AssertionError('ENGINE')),patch('run.CommandQuota',side_effect=AssertionError('PROVIDER')),patch('transport.subprocess.run',side_effect=AssertionError('COMMAND')):
            for args,message in cases:
                with self.subTest(args=args),patch.object(sys,'argv',['run.py',*args]):
                    # argparse errors carry exit code 2; live/explicit gate carries a diagnostic.
                    with self.assertRaises((SystemExit,ValueError)) as caught:run.main()
                    if not isinstance(caught.exception,SystemExit) or caught.exception.code!=2:self.assertIn(message,str(caught.exception))
            with patch.object(sys,'argv',['run.py','--live','--quota-command','--entry-check']),patch('run.require_live_release',return_value=None) as release:
                run.main();release.assert_called_once_with(True)
    def test_10_release_schema_binds_amend5_not_amend4(self):
        path=OUT/'SYNTHETIC_RELEASE_SCHEMA_INPUT_NOT_AUTHORIZATION.json'
        z=dict(schema='HOLDOUT_D_AMEND3_LIVE_RELEASE_V1',explicit_user_authorization=True,provider_evidence='SYNTHETIC_OFFLINE_TEST_ONLY',authorized_by='SYNTHETIC_NOT_AUTHORIZATION',authorized_at='2020-01-01T00:00:00+00:00',runner_version=evidence.version(),amend3_manifest_sha256=sha((ROOT/'HOLDOUT_D_AMEND4.sha256').read_bytes()))
        path.write_text(json.dumps(z))
        with patch.object(evidence,'RELEASE',path),patch('transport.subprocess.run') as execute:
            with self.assertRaisesRegex(ValueError,'live_release_freeze'):transport.CommandQuota(live=True)
            z['amend3_manifest_sha256']=sha((ROOT/'HOLDOUT_D_AMEND5.sha256').read_bytes());path.write_text(json.dumps(z))
            probe=transport.CommandQuota(live=True)
            z['explicit_user_authorization']=False;path.write_text(json.dumps(z))
            with self.assertRaisesRegex(ValueError,'live_release_authorization'):probe()
            execute.assert_not_called()
    def test_11_cli_selects_command_or_http_provider(self):
        from contextlib import ExitStack
        from unittest.mock import Mock
        for command in (True,False):
            with self.subTest(command=command),ExitStack() as stack:
                e=Mock();e.wal.events=[]
                stack.enter_context(patch.object(sys,'argv',['run.py','--live',*(['--quota-command'] if command else ['--quota-url','unused'])]))
                stack.enter_context(patch('run.require_live_release',return_value=None))
                stack.enter_context(patch('run.allowed_dir',side_effect=lambda p:p.resolve()))
                stack.enter_context(patch('run.Engine',return_value=e))
                stack.enter_context(patch('run.environment',return_value={}))
                stack.enter_context(patch('run.HTTPTransport'))
                cmd=stack.enter_context(patch('run.CommandQuota'));http=stack.enter_context(patch('run.HTTPQuota'))
                execute=stack.enter_context(patch('run.execute',return_value={}))
                run.main()
                (cmd if command else http).assert_called_once()
                (http if command else cmd).assert_not_called()
                self.assertIs(execute.call_args.args[2],(cmd if command else http).return_value)
                e.close.assert_called_once()

if __name__=='__main__':
    resources();binpath=OUT/'command_bin';binpath.mkdir()
    # PATH 中 python3 是独立假命令；只检查固定参数并输出合成值，不打开目标脚本。
    fake_script='#!'+sys.executable+'\nimport os,sys,time\nassert sys.argv[1:]==['+repr(CMD[1])+']\nmode=os.environ.get("QUOTA_FAKE_MODE","ok")\nif mode=="timeout":time.sleep(55)\nprint(os.environ.get("QUOTA_FAKE_OUTPUT","1 2"))\nraise SystemExit(7 if mode=="exit" else 0)\n'
    p=binpath/'python3';p.write_text(fake_script);p.chmod(0o700)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CommandQuotaTests))
    (OUT/'command_quota_results.json').write_text(json.dumps(dict(tests=result.testsRun,passed=result.wasSuccessful(),errors=[(str(t),e) for t,e in result.errors],failures=[(str(t),e) for t,e in result.failures],resources=resources(),thread_counts=thread_counts,fake_command=str(p.relative_to(ROOT)),real_quota_executed=False),indent=2,ensure_ascii=False)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
