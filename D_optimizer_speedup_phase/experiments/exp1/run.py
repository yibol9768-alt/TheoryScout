#!/usr/bin/env python3
"""CPU only. Invoke under the preregistered systemd scope and interpreter."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE))  # Only this experiment's modules; never another project.
import numpy as np
import proxy
import matrix


class Context:
    def __init__(self,out,seconds):
        self.out=out
        out.mkdir(parents=True,exist_ok=False)
        self.start=time.monotonic()
        self.deadline=self.start+seconds

    def save(self,name,value):
        tmp=self.out/(name+".json.tmp")
        tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
        tmp.replace(self.out/(name+".json"))

    def event(self,event,**kw):
        row=dict(event=event,elapsed=time.monotonic()-self.start,**kw)
        with (self.out/"events.jsonl").open("a") as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)

    def guard(self,deadline=None):
        if time.monotonic()>min(self.deadline,deadline if deadline else self.deadline):
            raise TimeoutError("pre-registered stage/global budget reached; incomplete")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>1_700_000_000:
            raise MemoryError("1.7GB soft RSS stop; incomplete")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("output must remain inside this project")
    prereg=ROOT/"PREREG_EXP1.md"
    digest=hashlib.sha256(prereg.read_bytes()).hexdigest()
    expected=(ROOT/"PREREG_EXP1.sha256").read_text().split()[0]
    assert digest==expected,"frozen preregistration changed"
    ctx=Context(out,160 if args.smoke else 3200)
    meta=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              prereg_sha256=digest,python=sys.executable,numpy=np.__version__,
              mode="smoke" if args.smoke else "formal",threads={k:os.environ[k] for k in
              ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")},
              source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (HERE/"run.py",HERE/"proxy.py",HERE/"matrix.py")},
              scope=os.environ.get("SYSTEMD_EXEC_PID"),status="running")
    ctx.save("manifest",meta)
    try:
        a=proxy.run(ctx,args.smoke)
        ctx.save("proxy",a)
        b=matrix.run(ctx,args.smoke)
        if args.smoke:
            ctx.save("smoke_matrix",b)
        meta["status"]="complete"
    except Exception as e:
        meta["status"]="incomplete"
        meta["exception"]=repr(e)
        (out/"exception.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        meta["elapsed_seconds"]=time.monotonic()-ctx.start
        meta["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        meta["finished_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
        ctx.save("manifest",meta)
        ctx.event("finished",**meta)
    if meta["status"]!="complete":
        raise SystemExit(1)


if __name__=="__main__":
    main()
