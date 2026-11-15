import subprocess
from unittest.mock import patch
from common import *
guards();sys.path.insert(0,str(RUNNER))
from transport import HTTPTransport,HTTPQuota,URL
from protocol import Task
from engine import Engine
r=resources();results=[]
shimdir=OUT/'entry_bin';shimdir.mkdir()
shim=shimdir/'python3';shim.write_text('#!/bin/sh\nexec '+sys.executable+' -B '+str(RUNNER/'audit_regression/entry_sentinel.py')+' "$@"\n');shim.chmod(0o700)
base_env=dict(os.environ,PATH=str(shimdir)+os.pathsep+os.environ['PATH'],PYTHONPATH=str(RUNNER/'audit_regression'))
env_attacks=[{}, {'HOLDOUT_D_LIVE':'1','LIVE':'true','SOLKEY':'AUDIT_DUMMY_NOT_SECRET'},
             {'HOLDOUT_D_LIVE_RELEASE':str(OUT/'forged_release.json'),'HOLDOUT_D_RELEASE_FILE':str(OUT/'forged_release.json'),
              'HOLDOUT_D_MODE':'live','HOLDOUT_D_LIVE_AUTHORIZED':'true','SOLKEY':'AUDIT_DUMMY_NOT_SECRET'}]
save('forged_release.json',dict(schema='FORGED_AUDIT_INPUT_NOT_AUTHORIZATION',explicit_user_authorization=True))
argslist=[]
for opt in ('--live','--out','--quota-url','--smoke-approval','--environment-only','--test-unfrozen','--entry-check'):
    for n in range(3,len(opt)):
        prefix=opt[:n]
        # Include every strict long-option prefix; supply appropriate dummy values and live flag.
        tail=[str(OUT/'unused')] if opt in ('--out','--quota-url','--smoke-approval') else []
        argslist.append(([prefix]+tail) if opt=='--live' else ['--live',prefix]+tail)
argslist += [['--live'],['--live=true'],['--live=1'],['--live','--entry-check'],['--live','--environment-only'],
             ['--live','--test-unfrozen'],['--live','--quota-url',URL],['--','--live'],['--live','--live']]
for eidx,extra in enumerate(env_attacks):
    for args in argslist:
        for entry in ('python','shell'):
            cmd=([sys.executable,'-B',str(RUNNER/'audit_regression/entry_sentinel.py')] if entry=='python' else [str(RUNNER/'launch.sh')])+args
            p=subprocess.run(cmd,cwd=ROOT,env=dict(base_env,**extra),capture_output=True,text=True,timeout=15)
            assert p.returncode==2 and 'REJECTED_BEFORE_CONSTRUCTION' in p.stdout and 'CONSTRUCTED_BEFORE_REJECTION' not in p.stderr,(cmd,p.stdout,p.stderr)
            results.append(dict(entry=entry,args=args,env_attack=eidx,code=p.returncode,out=p.stdout,err=p.stderr))
classes=[]
for label,call in [('engine',lambda:Engine(OUT/'never_live',mode='live',live=True)),
                   ('transport',lambda:HTTPTransport(live=True)),('quota',lambda:HTTPQuota(URL.replace('chat/completions','quota'),live=True))]:
    try:call()
    except ValueError as ex:classes.append(dict(label=label,rejected=str(ex)))
    else:raise AssertionError(label+' unexpectedly constructed')

# No socket is opened. Capture the urllib request at its final open boundary.
# The constructor and send method below are unmodified production code.
reached=[]
class AtOpen:
    def open(self,req,timeout):
        reached.append(dict(url=req.full_url,body_sha256=digest(req.data),timeout=timeout,headers=dict(req.header_items())))
        raise RuntimeError('AUDIT_STOP_BEFORE_NETWORK')
tests=[URL,'http://127.0.0.1:18319/anything','http://127.0.0.1:18319@audit.invalid/v1/chat/completions']
attacks=[]
for target in tests:
    with patch('urllib.request.build_opener',return_value=AtOpen()):
        try:HTTPTransport(live=True,test_mock_url=target)
        except ValueError as ex:attacks.append(dict(target=target,rejected=str(ex),open_reached=0))
        else:raise AssertionError('production test URL bypass')
assert not reached
save('entry_results.json',dict(resources=r,cli_cases=len(results),cli=results,classes=classes,transport_bypasses=attacks,open_sentinel=reached,
                             release_exists=(RUNNER/'HOLDOUT_D_LIVE_RELEASE.json').exists(),live_ledger_exists=(RUNNER/'HOLDOUT_D_live').exists()))
print(json.dumps(dict(cli_cases=len(results),transport_bypasses=attacks),ensure_ascii=False))
