"""Portable command runner. Supply Python with numpy, scipy and mpmath.

Writes only to a new relative destination and records wall time and exit code.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
def main():
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['polar','adam','history','check-adam','check-adam-integrals','check-adam-all','replay','tables','full-plan','full-experiments','certificates','query-tables'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--certificates',type=Path,default=Path('two_step_adam_certificate/reference'))
    p.add_argument('--replay-root',type=Path,default=Path('experiments_replay/reference/small_replay'))
    a=p.parse_args();assert not a.output.is_absolute() and '..' not in a.output.parts
    os.chdir(BASE);a.output.mkdir(parents=True,exist_ok=False);dest=a.output
    jobs=[]
    def add(script,*args):jobs.append([script,*map(str,args)])
    if a.task in ['polar','certificates']:
        add('polar_certificate/polar_variance.py','--grid',4096,'--output',dest/'polar_variance.json')
        add('polar_certificate/check_polar.py','--certificate',dest/'polar_variance.json','--output',dest/'polar_check.json')
    if a.task in ['adam','certificates']:
        add('two_step_adam_certificate/stabilizer_ordering.py','--output',dest/'stabilizer_ordering.json')
        add('two_step_adam_certificate/matched_chain.py','--output',dest/'matched_chain.json')
        add('two_step_adam_certificate/moment_history.py','--points',dest/'matched_chain.json','--output',dest/'moment_history.json')
    if a.task in ['history','certificates']:add('two_step_adam_certificate/history_reversal.py','--output',dest/'history_reversal.json')
    if a.task=='certificates':add('two_step_adam_certificate/check_adam.py','--certificate-dir',dest,'--reintegrate','none','--output',dest/'adam_check.json')
    if a.task.startswith('check-adam'):
        mode={'check-adam':'none','check-adam-integrals':'sample','check-adam-all':'all'}[a.task]
        add('two_step_adam_certificate/check_adam.py','--certificate-dir',a.certificates,'--reintegrate',mode,'--output',dest/'adam_check.json')
    if a.task=='replay':
        for design in ['matched','confirmation']:
            add('experiments_replay/replay_retrieval.py','--design',design,'--output',dest/design)
    if a.task=='query-tables':
        for design in ['matched','confirmation']:
            add('experiments_replay/measurements_from_queries.py','--input',a.replay_root/design,'--output',dest/(design+'_measurements.csv'))
        add('experiments_replay/recompute_tables.py','--matched-measurements',dest/'matched_measurements.csv','--confirmation-measurements',dest/'confirmation_measurements.csv','--output',dest/'tables')
    if a.task=='tables':add('experiments_replay/recompute_tables.py','--output',dest/'tables')
    if a.task=='full-plan':add('experiments_replay/replay_retrieval.py','--plan','--full','--output',dest/'plan.json')
    if a.task=='full-experiments':
        for design in ['matched','confirmation']:
            add('experiments_replay/replay_retrieval.py','--full','--design',design,'--output',dest/design)
    if a.task=='full-experiments':
        running=[];records=[]
        for index,job in enumerate(jobs):
            handle=(dest/f'command_{index+1}.log').open('x')
            print('RUN python '+' '.join(job),flush=True)
            start=time.monotonic();proc=subprocess.Popen([sys.executable,'-B','-u',*job],stdout=handle,stderr=subprocess.STDOUT)
            running.append((proc,handle,job,start))
        failed=False
        while running:
            for item in list(running):
                proc,handle,job,start=item
                status=proc.poll()
                if status is None:continue
                handle.close();running.remove(item)
                record=dict(command='python '+' '.join(job),seconds=time.monotonic()-start,exit_code=status)
                records.append(record);print(json.dumps(record),flush=True)
                (dest/'timings.json').write_text(json.dumps(records,indent=2))
                if status:failed=True
            if failed:
                for proc,handle,job,start in running:proc.terminate();proc.wait();handle.close()
                raise SystemExit(1)
            if running:time.sleep(.5)
        return
    records=[]
    for index,job in enumerate(jobs):
        start=time.monotonic();print('RUN python '+' '.join(job),flush=True)
        with (dest/f'command_{index+1}.log').open('x') as log:
            result=subprocess.run([sys.executable,'-B','-u',*job],stdout=log,stderr=subprocess.STDOUT)
        record=dict(command='python '+' '.join(job),seconds=time.monotonic()-start,exit_code=result.returncode)
        records.append(record);print(json.dumps(record),flush=True)
        (dest/'timings.json').write_text(json.dumps(records,indent=2))
        if result.returncode:
            print((dest/f'command_{index+1}.log').read_text()[-4000:],flush=True);raise SystemExit(result.returncode)
if __name__=='__main__':main()
