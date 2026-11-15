from pathlib import Path
p=Path('supplement/reproduce.py');s=p.read_text();mark='    records=[]\n'
insert='''    if a.task=='full-experiments':
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
'''
assert mark in s;s=s.replace(mark,insert+mark);p.write_text(s)
