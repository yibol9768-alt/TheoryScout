"""Supervisor-issued pipe credentials, hash-chained events, fail-closed publication."""
import hashlib
import json
import os
from pathlib import Path
import time
from common_v2 import OUT, dump, read, sha, plan, jobname, usage

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def credential(stage, manifest):
    # No environment-only bypass. Descriptor exists only when supervisor passes it.
    fd = int(os.environ['Z4_CREDENTIAL_FD'])
    with os.fdopen(fd) as f: payload=json.load(f)
    launch=read(OUT/'formal/launch.json')
    if payload['stage']!=stage or payload['manifest']!=manifest or payload['supervisor_pid']!=os.getppid():
        raise ValueError('credential binding')
    if launch['pid']!=os.getppid() or digest(payload)!=launch['credentials'][stage]:
        raise ValueError('credential not issued by supervisor')
    return digest(payload)

class Events:
    def __init__(self,path,manifest,credential_hash):
        self.f=Path(path).open('x'); self.prev=manifest; self.cred=credential_hash; self.index=0
    def append(self,state,**kwargs):
        obj=dict(index=self.index,state=state,prev=self.prev,credential=self.cred,usage=usage(),**kwargs)
        h=digest(obj)
        self.f.write(json.dumps(dict(**obj,hash=h),allow_nan=False)+'\n')
        self.f.flush(); os.fsync(self.f.fileno()); self.prev=h; self.index+=1
    def close(self): self.f.close()

def inspect(data,manifest,cfg):
    errors=[]; expected=plan(cfg); committed=[]; active=None; position=0; stop=None
    hashes={}; previous=manifest; prior_usage=None
    def err(s): errors.append(s)
    try:
        launch=read(data/'launch.json'); start=read(data/'start.json'); status=read(data/'runner_status.json')
        if launch['config']!=cfg or launch['manifest']!=manifest or start['manifest']!=manifest: err('manifest binding')
        if start['credential']!=launch['credentials']['runner']: err('runner credential')
        if start['streams']!=sha(OUT/'streams.json') or start['config']!=cfg: err('start binding')
        lines=(data/'events.jsonl').read_text().splitlines()
        for i,line in enumerate(lines):
            # Strict duplicate/nonfinite parser, identical to stored JSON parser.
            class Text:
                def __fspath__(self): return ''
            def pairs(xs):
                o={}
                for k,v in xs:
                    if k in o: raise ValueError('duplicate event key')
                    o[k]=v
                return o
            def bad(x): raise ValueError('nonfinite event')
            event=json.loads(line,object_pairs_hook=pairs,parse_constant=bad)
            h=event.pop('hash')
            if h!=digest(event) or event['prev']!=previous or type(event['index']) is not int or event['index']!=i:
                err('event hash/order')
            previous=h
            if stop is not None: err('event after stop')
            if event['credential']!=start['credential']: err('event credential')
            u=event['usage']
            if set(u)!={'cpu','mono','rss_mib'} or any(type(v) not in [int,float] or not 0<=v<float('inf') for v in u.values()): err('usage schema')
            if prior_usage and any(u[k]<prior_usage[k] for k in ['cpu','mono']): err('usage decreasing')
            prior_usage=u
            state=event['state']
            base={'index','state','prev','credential','usage'}
            if state=='started':
                if set(event)!=base|{'job'}: err('started schema')
                if active is not None or position>=len(expected) or event['job']!=expected[position]: err('job order')
                active=event['job'];position+=1
            elif state=='complete':
                if set(event)!=base|{'job','sha256'}: err('complete schema')
                if active!=event['job'] or active is None: err('complete without start')
                name=jobname(event['job']); p=data/name
                if not p.is_file() or p.is_symlink() or sha(p)!=event['sha256']: err('commit hash: '+name)
                hashes[name]=event['sha256'];committed.append(event['job']);active=None
            elif state=='stopped':
                if set(event)!=base|{'reason','active','error'}: err('stop schema')
                stop=event
                if event['active']!=active or event['error'] is not None or event['reason'] not in ['planned_complete','cpu_soft','wall_soft']: err('abnormal stop')
                if event['reason']=='planned_complete' and (committed!=expected or active is not None): err('premature complete')
            else: err('unknown event')
        if stop is None: err('missing terminal event')
        if status['event_tail']!=previous or status['completed']!=committed: err('status/event mismatch')
        if stop and (status['reason']!=stop['reason'] or status['active']!=stop['active'] or status['error']!=stop['error']): err('status stop mismatch')
        if status['usage']['cpu']>cfg['runner_cpu_hard'] or status['wall']>cfg['runner_wall_hard']: err('runner over budget')
        wanted={jobname(j) for j in committed}
        actual={p.name for p in data.glob('*.npz')}|({'lock.json'} if (data/'lock.json').exists() else set())
        if wanted!=actual: err('artifact/event set mismatch')
        if list(data.glob('*.partial')): err('partial artifact retained')
    except Exception as e: err('integrity parse: '+repr(e))
    return dict(errors=errors,hashes=hashes,committed=committed,
                full=committed==expected and active is None and stop is not None and stop['reason']=='planned_complete')

def publish(data,manifest,cfg,stages,wall,children_cpu,self_cpu,budget_probe=None):
    """Only supervisor can publish ACCEPTED. Recompute every statistic from raw arrays."""
    from analyze_v2 import make_report
    errors=[]
    audit=inspect(data,manifest,cfg); errors+=audit['errors']
    if not audit['full']: errors.append('not all planned jobs complete')
    if stages!=[dict(stage='runner',returncode=0,killed=False),dict(stage='analysis',returncode=0,killed=False)]:
        errors.append('abnormal child exit/hard kill')
    if wall>cfg['total_wall_hard'] or children_cpu+self_cpu>cfg['total_cpu_hard']-cfg['preparation_cpu_reserve']:
        errors.append('total formal budget exceeded')
    report=None
    try:
        pending=read(data/'analysis/PENDING.json')
        launch=read(data/'launch.json')
        if pending['credential']!=launch['credentials']['analysis'] or pending['manifest']!=manifest:
            errors.append('analysis credential/manifest')
        if pending['confirmation_valid'] is not False: errors.append('PENDING cannot confirm')
        if not 0 <= pending['usage']['cpu'] <= cfg['analysis_cpu_hard']: errors.append('analysis CPU')
        for name,h in pending['files'].items():
            if name not in ['report.json','curves.csv','matching.csv','old_pairs.csv','scaled_residuals.csv','availability.csv','all_completed.csv']:
                raise ValueError('unexpected report path')
            if sha(data/'analysis'/name)!=h: errors.append('report mutated: '+name)
        if set(pending['files'])!={'report.json','curves.csv','matching.csv','old_pairs.csv','scaled_residuals.csv','availability.csv','all_completed.csv'}:
            errors.append('report set')
        report=make_report(data,manifest,cfg)
        if report!=read(data/'analysis/report.json'): errors.append('report does not recompute')
        # Check CSV values and coverage, not just candidate booleans or row counts.
        from analyze_v2 import csv_text
        for name,key in [('curves.csv','curves'),('matching.csv','matching'),('old_pairs.csv','old_pairs'),
                         ('scaled_residuals.csv','scaled_residuals'),('availability.csv','availability'),('all_completed.csv','all_completed')]:
            if (data/'analysis'/name).read_text()!=csv_text(report[key]): errors.append('table mismatch: '+name)
        errors+=report['data_errors']
        if report['raw_hashes']!=audit['hashes']: errors.append('report/raw event hashes')
    except Exception as e: errors.append('publication: '+repr(e))
    if budget_probe:
        wall,self_cpu=budget_probe()
    if wall>cfg['total_wall_hard'] or self_cpu>=cfg['supervisor_cpu_hard'] or children_cpu+self_cpu>cfg['total_cpu_hard']-cfg['preparation_cpu_reserve']:
        errors.append('publication budget exceeded')
    if any(type(v) not in (int,float) or not 0 <= v < float('inf') for v in [wall,children_cpu,self_cpu]):
        errors.append('invalid budget number')
    status=dict(manifest=manifest,stages=stages,wall=wall,children_cpu=children_cpu,self_cpu=self_cpu,
                confirmation_valid=not errors,errors=errors)
    dump(data/'launch_status.json',status)
    if errors:
        dump(data/'INVALID.json',status)
    else:
        names={'launch.json','start.json','events.jsonl','runner_status.json','launch_status.json'}
        names|=set(audit['hashes'])
        names|={'analysis/'+n for n in pending['files']}
        names.add('analysis/PENDING.json')
        evidence={n:sha(data/n) for n in sorted(names)}
        dump(data/'ACCEPTED.pending',dict(**status,evidence=evidence,decisions=report['decisions']))
        if budget_probe:
            wall,self_cpu=budget_probe()
            if wall>cfg['total_wall_hard'] or self_cpu>=cfg['supervisor_cpu_hard'] or children_cpu+self_cpu>cfg['total_cpu_hard']-cfg['preparation_cpu_reserve']:
                dump(data/'PUBLICATION_ABORTED.json',dict(wall=wall,cpu=self_cpu))
                return dict(confirmation_valid=False,errors=['budget before atomic commit'])
        os.link(data/'ACCEPTED.pending',data/'ACCEPTED.json')
        fd=os.open(data,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        (data/'ACCEPTED.pending').unlink()
    return status

def verify_accepted(data,manifest,cfg):
    """Full replay of acceptance evidence; no scientific draws or overwrites."""
    from analyze_v2 import make_report, csv_text
    observed=read(data/'EXIT_OBSERVED.json')
    if observed['returncode']!=0 or observed['killed'] is not False or observed['manifest']!=manifest: raise ValueError('supervisor did not exit normally')
    if observed['launch_status_sha256']!=sha(data/'launch_status.json'): raise ValueError('observer status hash')
    receipt=read(data/'SUPERVISOR_EXIT.json')
    if observed['receipt_sha256']!=sha(data/'SUPERVISOR_EXIT.json') or receipt['normal'] is not True or receipt['manifest']!=manifest: raise ValueError('exit receipt')
    if receipt['launch_status_sha256']!=sha(data/'launch_status.json'): raise ValueError('receipt status hash')
    if not 0 <= observed['wall'] <= cfg['total_wall_hard'] or not 0 <= observed['cpu'] <= cfg['total_cpu_hard']-cfg['preparation_cpu_reserve']: raise ValueError('observer total budget')
    if not 0 <= observed['self_cpu'] < 10: raise ValueError('observer CPU')
    accepted=read(data/'ACCEPTED.json');status=read(data/'launch_status.json')
    if (data/'PUBLICATION_ABORTED.json').exists() or (data/'INVALID.json').exists(): raise ValueError('invalid publication')
    if not accepted['confirmation_valid'] or accepted['manifest']!=manifest: raise ValueError('invalid credential')
    expected={'launch.json','start.json','events.jsonl','runner_status.json','launch_status.json',
              'analysis/PENDING.json','lock.json'}
    expected|={p.name for p in data.glob('*.npz')}
    expected|={'analysis/'+n for n in ['report.json','curves.csv','matching.csv','old_pairs.csv','scaled_residuals.csv','availability.csv','all_completed.csv']}
    if set(accepted['evidence'])!=expected: raise ValueError('evidence coverage')
    for n,h in accepted['evidence'].items():
        if sha(data/n)!=h: raise ValueError('evidence mutated: '+n)
    for key in status:
        if accepted[key]!=status[key]: raise ValueError('status mismatch')
    if status['stages']!=[dict(stage='runner',returncode=0,killed=False),dict(stage='analysis',returncode=0,killed=False)]: raise ValueError('hard kill')
    if status['wall']>cfg['total_wall_hard'] or status['self_cpu']>=cfg['supervisor_cpu_hard'] or status['self_cpu']+status['children_cpu']>cfg['total_cpu_hard']-cfg['preparation_cpu_reserve']: raise ValueError('budget')
    audit=inspect(data,manifest,cfg)
    if audit['errors'] or not audit['full']: raise ValueError('integrity replay')
    report=make_report(data,manifest,cfg)
    if report['data_errors'] or report!=read(data/'analysis/report.json') or report['decisions']!=accepted['decisions']: raise ValueError('analysis replay')
    for name,key in [('curves.csv','curves'),('matching.csv','matching'),('old_pairs.csv','old_pairs'),
                     ('scaled_residuals.csv','scaled_residuals'),('availability.csv','availability'),('all_completed.csv','all_completed')]:
        if (data/'analysis'/name).read_text()!=csv_text(report[key]): raise ValueError('table replay')
    return accepted
