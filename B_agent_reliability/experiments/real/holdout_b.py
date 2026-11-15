"""Local-only frozen length holdout; inherits the R2 HTTP protocol unchanged."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import threading
import time
import reliability_core as d
import reliability_r2 as r
import report_reliability_r2 as oldreport

ROOT = d.ROOT
OUT = ROOT / "holdout_b_real"
SOURCES = ("PREREG_HOLDOUT_B.md", "holdout_b.py", "report_holdout_b.py", "test_holdout_b.py",
           "reliability_core.py", "reliability_r2.py", "report_reliability_r2.py", "difficulty.py",
           "PREDICTIONS_HOLDOUT_B.json")

def write(path, obj):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n")
    pending.replace(path)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def schedule():
    return [dict(family="sheet", level=0, knob=2, seed=5100000+10000*length+rep, length=length)
            for rep in range(1,21) for length in (16,64)]

def stats(trace):
    a = [x for x in trace if x["event"] == "action"]
    c = collections.Counter()
    seen = False
    previous_error = None
    for x in a:
        stage = x["review"]
        bad = not x["correct"]
        if stage == 0:
            c["n"] += 1
            c["e"] += bad
            prefix = "after" if seen else "before"
            c[prefix+"_n"] += 1
            c[prefix+"_e"] += bad
            if previous_error is not None:
                prefix = "next_repaired" if previous_error else "next_clean"
                c[prefix+"_n"] += 1
                c[prefix+"_e"] += bad
            previous_error = bad
            seen |= bad
            c["block_"+str(x["step"]//16)+"_n"] += 1
            c["block_"+str(x["step"]//16)+"_e"] += bad
        else:
            pre = "r"+str(stage)
            c[pre+"_n"] += 1
            c[pre+"_ok"] += not bad
            detected = bool(x["action"] and x["action"]["detect"])
            c[pre+"_detect"] += detected
            c[pre+"_joint_"+str(int(detected))+"_"+str(int(not bad))] += 1
            delta = x["n_after"]-x["n_before"]
            c["review_"+("up" if delta>0 else "down" if delta<0 else "same")] += 1
            c["review_jump_"+str(delta)] += 1
    return dict(c)

RATES = {"error":("e","n"), "q_fb":("r1_detect","r1_n"),
         "r1":("r1_ok","r1_n"), "r2":("r2_ok","r2_n"),
         "reerror_next":("next_repaired_e","next_repaired_n"),
         "next_clean":("next_clean_e","next_clean_n"),
         "before":("before_e","before_n"), "after":("after_e","after_n")}

def params(counts):
    return {k: counts.get(a,0)/counts[b] if counts.get(b,0) else None for k,(a,b) in RATES.items()}

def predict(p, length):
    fatal = (1-p["r1"])*(1-p["r2"])
    main = (1-p["error"]*fatal)**length
    clean, past = 1.,0.
    for _ in range(length):
        clean,past = clean*(1-p["before"]), past*(1-p["after"]*fatal)+clean*p["before"]*(1-fatal)
    return dict(main=main, no_repair=(1-p["error"])**length,
                terminal_only=(31/50)**(length/32), history_two_state=clean+past)

def quantile(xs, q):
    return sorted(xs)[int(q*(len(xs)-1))]

def cluster(rows, seed=862731):
    rng = random.Random(seed)
    samples=[]
    for _ in range(10000):
        c=collections.Counter()
        for row in rng.choices(rows,k=len(rows)):
            c.update(row)
        p=params(c)
        if all(p[k] is not None for k in ("error","r1","r2","before","after")):
            samples.append(p)
    return samples

def binom_probs(n,p):
    return [math.comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)]

def predictive_interval(samples, length, n=20):
    mix=[0.]*(n+1)
    for p in samples:
        for k,mass in enumerate(binom_probs(n,predict(p,length)["main"])):
            mix[k]+=mass/len(samples)
    assert abs(sum(mix)-1)<1e-9
    def inverse(q):
        s=0.
        for k,mass in enumerate(mix):
            s+=mass
            if s>=q: return k
        return n
    return {"n":n,"coverage":.975,"counts":[inverse(.0125),inverse(.9875)],"pmf":mix}

def fit():
    # Full old independent replay, no filtering by success and no new model calls.
    audited=oldreport.inspect(ROOT/"calib1_real", "calib1")
    rows=[]
    for result in audited["summary"]["results"]:
        trace=[json.loads(s) for s in (ROOT/"calib1_real"/(result["id"]+".jsonl")).read_text().splitlines()]
        rows.append(stats(trace))
    c=collections.Counter()
    for row in rows: c.update(row)
    assert (c["e"],c["n"],c["r1_ok"],c["r1_n"],c["r2_ok"],c["r2_n"])==(283,1279,222,283,42,61)
    p=params(c)
    samples=cluster(rows)
    rates={k:{"e":c.get(a,0),"n":c.get(b,0),"point":p[k],
              "cluster95":[quantile([x[k] for x in samples if x[k] is not None],q) for q in (.025,.975)]}
           for k,(a,b) in RATES.items()}
    predictions={}
    for length in (16,64):
        vals=[predict(x,length)["main"] for x in samples]
        predictions[str(length)]={**predict(p,length),"calibration95":[quantile(vals,q) for q in (.025,.975)],
                                  "predictive":predictive_interval(samples,length)}
    widths=[v["calibration95"][1]-v["calibration95"][0] for v in predictions.values()]
    g0=len(samples)>=9900 and c["e"]>=30 and max(widths)<=.40
    obj=dict(utc=d.now(),calibration_tasks=len(rows),counts=dict(c),rates=rates,parameters=p,
             fatal_episode=(1-p["r1"])*(1-p["r2"]),fatal_per_step=p["error"]*(1-p["r1"])*(1-p["r2"]),
             bootstrap_valid=len(samples),bootstrap_seed=862731,predictions=predictions,
             calibration_replay_checks=audited["checks"],G0=g0)
    path=ROOT/"PREDICTIONS_HOLDOUT_B.json"
    with path.open("x") as f: f.write(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"G0":g0,"valid":len(samples),"predictions":predictions},ensure_ascii=False,indent=2))

def freeze():
    assert json.loads((ROOT/"PREDICTIONS_HOLDOUT_B.json").read_text())["G0"]
    oldreport.r.verify("calib1")
    inputs=sorted((ROOT/"calib1_real").glob("*.jsonl"))+[ROOT/"calib1_real/summary.json",ROOT/"FREEZE_CALIB1.json"]
    obj=dict(utc=d.now(),sha256={n:sha(ROOT/n) for n in SOURCES},
             input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
             review_sha256=sha(ROOT.parent.parent/"REVIEW_B.md"),schedule=schedule())
    with (ROOT/"FREEZE_HOLDOUT_B.json").open("x") as f:
        f.write(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
    print("FROZEN",sha(ROOT/"FREEZE_HOLDOUT_B.json"))

def verify():
    obj=json.loads((ROOT/"FREEZE_HOLDOUT_B.json").read_text())
    assert obj["sha256"]=={n:sha(ROOT/n) for n in SOURCES}, "freeze_mismatch"
    assert all(sha(ROOT/n)==h for n,h in obj["input_sha256"].items()), "calibration_changed"
    assert obj["schedule"]==schedule()
    return obj

class Budget(r.Budget):
    def __init__(self,out,mock=False,resume=False):
        if not resume:
            super().__init__(out,"holdout",mock)
            self.cap=2000
            return
        events=[json.loads(s) for s in (out/"budget.jsonl").read_text().splitlines()]
        assert not any(x["event"]=="halt" for x in events), "cannot_resume_halt"
        self.lock=threading.RLock()
        self.calls=sum(x["event"]=="reserve" for x in events)
        self.cap=2000;self.phase="holdout";self.real=not mock;self.halt=None
        self.quota_at=max((x["calls"] for x in events if x["event"]=="quota"),default=-1)
        self.task_calls=collections.Counter(x["task"] for x in events if x["event"]=="reserve")
        self.config_calls=collections.Counter(x["config"] for x in events if x["event"]=="reserve")
        outcomes=[x for x in events if x["event"]=="outcome"]
        self.recent=collections.deque((x["failed"] for x in outcomes[-20:]),maxlen=20)
        self.fail_streak=0
        for event in events:
            if event["event"]=="outcome": self.fail_streak=event["streak"]
            elif event["event"]=="cooldown": self.fail_streak=0
        cooldowns=[x for x in events if x["event"]=="cooldown"]
        self.cooldown_until=0
        if cooldowns:
            remaining=datetime.fromisoformat(cooldowns[-1]["utc"]).timestamp()+600-time.time()
            self.cooldown_until=time.monotonic()+max(0,remaining)
        self.journal=d.Journal.__new__(d.Journal)
        self.journal.f=(out/"budget.jsonl").open("a")
        assert self.calls<=2000


def execute(resume=False,mock=False):
    frozen=verify()
    out=ROOT/"holdout_b_mock" if mock else OUT
    if resume:
        summary=json.loads((out/"summary.json").read_text())
        assert summary["stage"]=="awaiting_human_smoke_review"
        assert len(summary["results"])==2 and all(x["advances"]>0 for x in summary["results"])
        gate=json.loads((out/"SMOKE_APPROVED.json").read_text())
        assert gate.get("approved") is True and gate.get("reviewer") and gate.get("checked_items")
        summary["smoke_approval"]=gate
    else:
        out.mkdir(exist_ok=False)
        summary=dict(started_utc=d.now(),phase="holdout_b",mode="mock" if mock else "real",model=d.MODEL,
                     concurrency=2,cap=2000,schedule=schedule(),frozen_sha256=frozen["sha256"],results=[],stage="smoke")
    budget=Budget(out,mock,resume)
    client=r.Client(budget,mock)
    def save():
        summary.update(calls=budget.calls,halt=budget.halt,updated_utc=d.now())
        write(out/"summary.json",summary)
    save()
    if resume:
        budget.journal.add("smoke_approved",approval=summary["smoke_approval"],calls=budget.calls)
        summary["stage"]="formal"
    try:
        with budget.lock: budget.quota()
    except d.Stop:
        summary["stage"]="stopped";save();budget.journal.f.close();return
    cursor=len(summary["results"])
    limit=len(schedule()) if resume or mock else 2
    with ThreadPoolExecutor(max_workers=2) as pool:
        while cursor<limit and not budget.halt:
            futures=[pool.submit(d.run_task,row,client,out) for row in schedule()[cursor:cursor+2]]
            for f in as_completed(futures):
                try: result=f.result()
                except Exception:
                    budget.stop("unexpected_local_exception");save()
                    raise RuntimeError("local execution failed; secret-bearing details suppressed") from None
                summary["results"].append(result)
                save();print(d.dump(result),flush=True)
            cursor+=2
    with budget.lock:
        if budget.calls%100==0 and not budget.halt:
            try: budget.quota()
            except d.Stop: pass
    if budget.halt: summary["stage"]="stopped"
    elif cursor==2 and not resume and not mock: summary["stage"]="awaiting_human_smoke_review"
    else: summary["stage"]="finished";summary["finished_utc"]=d.now()
    save();budget.journal.f.close()
    print("STAGE",summary["stage"],"CALLS",budget.calls,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["fit","freeze","smoke","formal","mock"])
    args=ap.parse_args()
    if args.action=="fit":fit()
    elif args.action=="freeze":freeze()
    else:execute(resume=args.action=="formal",mock=args.action=="mock")
