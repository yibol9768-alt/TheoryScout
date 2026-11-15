"""Post-freeze integrity/static/precision audit. No new simulations or fitted parameters."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"experiments/killer_v3"
start=time.monotonic()
result={"start_utc":datetime.now(timezone.utc).isoformat(),"scope":os.environ["KILLER_SCOPE"],"static":{},"precision":{},"centered_noise_reconstruction_max_abs":0.}
for filename in ("run.py","analyze.py","post_audit.py"):
    ast.parse((BASE/filename).read_text())
for filename in sorted((ROOT/"lemmas_v5").glob("L*.tex")):
    text=filename.read_text()
    stack=[]
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}",text):
        side,env=match.groups()
        if side=="begin":
            stack.append(env)
        else:
            assert stack.pop()==env,(filename,env)
    assert not stack
    assert text.count(r"\begin{problem}")==1 and text.count(r"\end{problem}")==1
    assert text.count(r"\[")==text.count(r"\]")
    assert len(re.findall(r"(?<!\\)\$",text))%2==0
    level=0
    for j,c in enumerate(text):
        if c in "{}" and (j==0 or text[j-1]!="\\"):
            level+=1 if c=="{" else -1
            assert level>=0,filename
    assert level==0,filename
    result["static"][filename.name]="balanced environments/braces/math delimiters; not TeX compiled, not QED"
data=json.loads((BASE/"formal/data.json").read_text())
for kind in ("local","path"):
    rows=[r for r in data["records"] if r["kind"]==kind]
    snr=[r["drift_norm"]/r["drift_se"] for r in rows]
    rel=[r["drift_error"]/r["drift_norm"] for r in rows]
    result["precision"][kind]={"SNR_range":[min(snr),max(snr)],"SNR_below_10":sum(s<10 for s in snr),
                              "raw_relative_drift_error_range":[min(rel),max(rel)]}
    for r in rows:
        lam=np.arange(1,r["n"]+1,dtype=float)**(-r["alpha"])
        mean=np.array(r["mean_u"])
        var=np.array(r["variance_u"])
        d=np.array(r["exact_d"])
        reconstruct=np.dot(lam,var*(r["N"]-1)/r["N"]+(mean-d)**2)
        result["centered_noise_reconstruction_max_abs"]=max(result["centered_noise_reconstruction_max_abs"],abs(float(reconstruct)-r["noise_estimate"]))
rows=[r for r in data["records"] if r["kind"]=="trajectory"]
after=[r for r in rows if r["t"]>0]
result["localization_after_initial_max_upper6se"]=max(r["localization_upper_max"] for r in after)
result["last_checkpoint_risk_examples"]=[{k:r[k] for k in ("alpha","n","eta","algorithm","t","total_mean","represented_mean","represented_se","tail")}
                                         for r in rows if r["alpha"] in (1.5,3.) and r["n"]==256 and r["eta"]==.016 and r["t"]==2048]
result["max_upper_lower_reference_ratio_after_initial"]=max(r["reference_upper"]["represented"]/r["reference_lower"]["represented"] for r in after)
props=subprocess.run(["systemctl","--user","show",os.environ["KILLER_SCOPE"],"-p","MemoryMax","-p","MemorySwapMax"],capture_output=True,text=True,check=True).stdout
result["scope_properties"]=props
assert "MemoryMax=2147483648" in props and "MemorySwapMax=0" in props
subprocess.run(["sha256sum","-c",str(BASE/"FROZEN.sha256")],cwd=ROOT,capture_output=True,text=True,check=True)
assert result["centered_noise_reconstruction_max_abs"]<1e-10
result["end_utc"]=datetime.now(timezone.utc).isoformat()
result["elapsed_seconds"]=time.monotonic()-start
result["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
result["executable"]=sys.executable
result["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
total=result["elapsed_seconds"]
for mode in ("smoke","formal"):
    for name in ("manifest.json","analysis_manifest.json"):
        total+=json.loads((BASE/mode/name).read_text())["elapsed_seconds"]
result["all_numerical_analysis_process_seconds"]=total
(BASE/"post_audit.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps(result,indent=2))
