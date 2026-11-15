"""Independent offline replay, operational audit and Chinese report for HOLDOUT_B."""
import collections
from datetime import datetime
import json
import math
from pathlib import Path
import random
import holdout_b as h
import reliability_core as d
import report_reliability_r2 as reference


def events(path):
    return [json.loads(s) for s in path.read_text().splitlines()]

def timestamp(s):
    return datetime.fromisoformat(s).timestamp()

def intervals(rows,seed=963812):
    rng=random.Random(seed);values=collections.defaultdict(list)
    for _ in range(10000):
        c=collections.Counter()
        for row in rng.choices(rows,k=len(rows)): c.update(row)
        for k,p in h.params(c).items():
            if p is not None: values[k].append(p)
    return {k:dict(cluster95=[h.quantile(v,q) for q in (.025,.975)],valid=len(v)) for k,v in values.items()}

def audit(out):
    frozen=h.verify()
    summary=json.loads((out/"summary.json").read_text())
    assert summary["frozen_sha256"]==frozen["sha256"]
    budget=events(out/"budget.jsonl")
    reserves=[x for x in budget if x["event"]=="reserve"]
    assert [x["call"] for x in reserves]==list(range(1,summary["calls"]+1))
    assert summary["calls"]<=2000
    quotas=[x for x in budget if x["event"]=="quota"]
    checked=set();n=0;cooldown_to=0;streak=0;cooldowns=0
    for x in budget:
        if x["event"]=="quota":
            assert x["calls"]==n
            checked.add(n)
        elif x["event"]=="reserve":
            if summary["mode"]=="real" and n%100==0: assert n in checked
            assert timestamp(x["utc"])+.02>=cooldown_to
            n+=1
        elif x["event"]=="outcome":
            streak=streak+1 if x["failed"] else 0
            assert x["streak"]==streak
        elif x["event"]=="cooldown":
            assert streak>=5 and x["seconds"]==600
            streak=0;cooldowns+=1;cooldown_to=timestamp(x["utc"])+600
    if summary["mode"]=="real" and summary["calls"]%100==0 and not summary["halt"]:
        assert summary["calls"] in checked
    if summary["halt"]!="quota_unavailable" and summary["mode"]=="real":
        assert set(range(0,summary["calls"]+1,100))<=checked
        assert all(max(x["five_hour"],x["week"])<80 for x in quotas) or summary["halt"]=="quota_80"
    trace_events=[];timeline=[];rows=[];checks=0;requests={};responses={};service=[]
    seen=set()
    for result in summary["results"]:
        row={k:result[k] for k in ("family","level","knob","seed","length")}
        assert row in h.schedule() and result["seed"] not in seen
        seen.add(result["seed"])
        ev=events(out/(result["id"]+".jsonl"));trace_events.extend(ev)
        t=ev[0]["task"]
        assert t==d.task(row) and t["prefix"]==reference.oracle(t)
        checks+=len(t["operations"])
        local_requests={x["call"]:x for x in ev if x["event"]=="request"}
        local_responses={x["call"]:x for x in ev if x["event"]=="response"}
        actual=t["initial"].copy();expected_step=0;expected_review=0
        for x in ev:
            if x["event"]=="request":
                assert x["model"]==d.MODEL and x["call"] not in requests
                requests[x["call"]]=x;timeline.append((x["utc"],1,x["call"]))
                assert x["messages"][0]=={"role":"system","content":d.COMMON+d.RULES["sheet"]}
                if x["attempt"]:
                    prev=[z for z in local_requests.values() if z["call"]<x["call"] and
                          (z["step"],z["review"])==(x["step"],x["review"])]
                    prev=max(prev,key=lambda z:z["call"])
                    assert x["attempt"]==prev["attempt"]+1<=4 and x["messages"]==prev["messages"]
                    fail=next(z for z in ev if z["event"]=="service_error" and z["call"]==prev["call"])
                    delay=(10,30,60,120)[x["attempt"]-1]
                    assert timestamp(x["utc"])-timestamp(fail["utc"])+.05>=delay
                checks+=1
            elif x["event"] in ("response","service_error"):
                timeline.append((x["utc"],-1,x["call"]))
                if x["event"]=="service_error":service.append(x)
                else:
                    responses[x["call"]]=x
                    resp=x["response"]
                    assert resp["model"]==d.MODEL
                    u=resp["usage"]
                    assert all(type(u[k]) is int and u[k]>=0 for k in ("prompt_tokens","completion_tokens","total_tokens"))
                    assert u["prompt_tokens"]<=16000 and u["completion_tokens"]<=4096
                    assert u["total_tokens"]==u["prompt_tokens"]+u["completion_tokens"]
            elif x["event"]=="action":
                assert (x["step"],x["review"])==(expected_step,expected_review)
                assert x["before"]==actual
                if x["review"]==0: assert actual==t["prefix"][x["step"]]
                resp=local_responses[x["call"]]["response"]
                assert x["content"]==resp["choices"][0]["message"]["content"]
                parsed,_=d.parse(x["content"],list(t["prefix"][x["step"]+1]))
                assert parsed==x["action"]
                target=t["prefix"][x["step"]+1]
                assert x["expected"]==target
                assert x["n_before"]==(0 if x["review"]==0 else sum(actual.get(k)!=v for k,v in target.items()))
                if parsed: actual.update(parsed["writes"])
                wrong=sum(actual.get(k)!=v for k,v in target.items())+int(parsed is None)
                assert actual==x["after"] and wrong==x["n_after"] and x["correct"]==(wrong==0)
                if x["correct"]:expected_step+=1;expected_review=0
                else:expected_review+=1
                checks+=1
        c=h.stats(ev)
        assert (c.get("n",0),c.get("e",0))==(result["advances"],result["seed_errors"])
        assert result["steps"]==expected_step
        assert result["repair_attempts"]==c.get("r1_n",0)+c.get("r2_n",0)
        assert result["repaired"]==c.get("r1_ok",0)+c.get("r2_ok",0)
        assert ev[-1]["event"]=="result" and ev[-1]["result"]==result
        if result["status"]=="success": assert expected_step==row["length"]
        elif result["status"]=="unrepaired_after_2_reviews":assert expected_review==3
        rows.append(dict(result,counts=c))
    assert set(requests)==set(range(1,summary["calls"]+1))
    assert set(x["call"] for x in reserves)==set(requests)
    active=set();peak=0
    for _,delta,call in sorted(timeline):
        if delta==1:
            assert call not in active;active.add(call);peak=max(peak,len(active))
        else:assert call in active;active.remove(call)
    assert peak<=2 and (not active or summary["halt"])
    outcomes=[x for x in budget if x["event"]=="outcome"]
    assert len(outcomes)==len(responses)+len(service)
    assert {x["call"] for x in outcomes if x["failed"]}=={x["call"] for x in service}
    usage=collections.Counter();models=collections.Counter()
    for x in responses.values():
        response=x["response"];models[response["model"]]+=1
        usage.update({k:response["usage"][k] for k in ("prompt_tokens","completion_tokens","total_tokens")})
        usage["reasoning_tokens"]+=(response["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens",0)
    smoke=[x for x in budget if x["event"]=="smoke_approved"]
    if len(summary["results"])>2 and summary["mode"]=="real":
        assert len(smoke)==1
        approved=smoke[0]["approval"]
        assert approved["approved"] is True and approved["reviewer"] and approved["checked_items"]
        first_two={f"sheet_d0_{r['seed']}" for r in h.schedule()[:2]}
        assert all(x["task"] in first_two for x in reserves if x["call"]<=smoke[0]["calls"])
        assert all(timestamp(x["utc"])>=timestamp(smoke[0]["utc"]) for x in reserves if x["call"]>smoke[0]["calls"])
    return dict(summary=summary,rows=rows,checks=checks,peak_concurrency=peak,quotas=quotas,
                usage=dict(usage),models=dict(models),responses=len(responses),service_errors=len(service),
                retries=sum(x["attempt"]>0 for x in requests.values()),cooldowns=cooldowns,
                smoke_approved=bool(smoke),all_events=trace_events)


def report(out=h.OUT):
    a=audit(out);s=a["summary"];cal=json.loads((h.ROOT/"PREDICTIONS_HOLDOUT_B.json").read_text())
    groups={};scores=collections.defaultdict(list)
    final_status={"success","unrepaired_after_2_reviews"}
    for length in (16,64):
        rows=[x for x in a["rows"] if x["length"]==length]
        counts=collections.Counter()
        for row in rows:counts.update(row["counts"])
        successes=sum(x["status"]=="success" for x in rows)
        known=sum(x["status"] in final_status for x in rows)
        unknown=20-known
        pred=cal["predictions"][str(length)]
        bounds=[successes/20,(successes+unknown)/20]
        n=len(rows);g2=known==20 and n==20
        obs=successes/20 if g2 else None
        acceptable=pred["predictive"]["counts"]
        g3=g2 and abs(obs-pred["main"])<=.20 and acceptable[0]<=successes<=acceptable[1]
        boot=intervals([x["counts"] for x in rows]) if rows else {}
        rate_values=h.params(counts)
        rates={}
        for name,(num,den) in h.RATES.items():
            ci=boot.get(name,{}).get("cluster95")
            cc=cal["rates"][name]
            sufficient=counts.get(den,0)>=20 and cc["n"]>=20
            disjoint=ci and (ci[1]<cc["cluster95"][0] or ci[0]>cc["cluster95"][1])
            flag=bool(sufficient and abs(rate_values[name]-cc["point"])>.10 and disjoint)
            rates[name]=dict(e=counts.get(num,0),n=counts.get(den,0),point=rate_values[name],
                             **boot.get(name,{}),sufficient=sufficient,drift=flag)
        for name in ("main","no_repair","terminal_only","history_two_state"):
            loss=[];brier=[]
            p=pred[name]
            for row in rows:
                if row["status"] not in final_status:continue
                y=int(row["status"]=="success")
                loss.append(-math.log(max(1e-15,p if y else 1-p)));brier.append((y-p)**2)
            if loss:scores[name].append(dict(length=length,log_loss=sum(loss)/len(loss),brier=sum(brier)/len(brier)))
        groups[str(length)]=dict(attempted=n,terminal_known=known,successes=successes,unknown=unknown,
                                success_bounds=bounds,observed=obs,prediction=pred,counts=dict(counts),rates=rates,
                                G2=g2,G3=g3,statuses=dict(collections.Counter(x["status"] for x in rows)),
                                task_success_exact95=reference.exact(successes,20) if g2 else None,
                                legacy_count_gate=g2 and successes>=30 and 20-successes>=30,
                                legacy_probability_rejection=g2 and abs(obs-pred["main"])>max(.03,5*math.sqrt(pred["main"]*(1-pred["main"])/20)))
    g2=all(x["G2"] for x in groups.values())
    g3=all(x["G3"] for x in groups.values())
    drift=any(r["drift"] for g in groups.values() for r in g["rates"].values())
    if not g2:conclusion="证据不足：正式长度留出尚未完整完成，不能判断B的真实长程预测是否相容。"
    elif not g3:conclusion="冻结的局部参数未通过本轮长程成功率预测门槛；B在该反馈协议下的齐次长度外推不符。"
    elif drift:conclusion="整题长度预测在本次粗精度下相容，但局部参数有迁移警示；不能认证机制。"
    else:conclusion="冻结的局部参数与本轮整题长度成功率相容；这仅支持辅助反馈协议的粗粒度外推，不能认证B的容量理论。"
    lines=["# HOLDOUT_B 结果报告", "", "**"+conclusion+"**", "",
           f"模型仅`{d.MODEL}`；HTTP尝试{s['calls']}/2000（含冒烟、重试），返回{a['responses']}，服务失败{a['service_errors']}（未知token耗用另计）；并发峰值{a['peak_concurrency']}。阶段`{s['stage']}`，全局停止原因`{s['halt']}`。",
           "", "|长度|已尝试/计划|终局已知|成功|冻结预测|校准95%|97.5%预测成功数|计划分母成功率界|G3|",
           "|---|---|---|---|---|---|---|---|---|"]
    for length,g in groups.items():
        p=g["prediction"]
        lines.append(f"|{length}|{g['attempted']}/20|{g['terminal_known']}|{g['successes']}|{p['main']:.4f}|{p['calibration95']}|{p['predictive']['counts']}|{g['success_bounds']}|{g['G3'] if g['G2'] else '尚不可判'}|")
    lines += ["", "失败在两次复查后仍错时就是模型终局失败；服务、预算和未执行槽是缺失。界以全部计划20题为分母，不把缺失当成模型错误或成功。整题成功的精确95%区间见ANALYSIS_HOLDOUT_B.json。",
              "", "## 逐项对照冻结门槛", "",
              f"- G0：{cal['G0']}；CALIB1独立重放{cal['calibration_replay_checks']}项，283个自然错误，整题bootstrap有效{cal['bootstrap_valid']}/10000；预测校准区间宽度均≤0.40。",
              f"- G1：两题冒烟完成{min(len(a['rows']),2)}/2；人工明确放行={a['smoke_approved']}。无放行时只保留冒烟，自动检查不代替人工。",
              f"- G2：完整40题终局={g2}；各长度删失/未执行和失败原因见机器报告。",
              f"- G3：两长度粗粒度误差≤0.20且预测计数区间通过={g3 if g2 else '尚不可判'}。这是有限预算外推诊断，不是±0.20等价性认证。",
              f"- G4：局部迁移警示={drift}；分母<20的修复项标记信息不足，不能据此宣称稳定。",
              f"- G5：已发生部分冻结、原始动作重放、硬限、并发、额度与服务规则审计通过，独立检查{a['checks']}项；重试{a['retries']}，冷却{a['cooldowns']}。未发生部分不冒充完成。",
              "", "## 局部参数与再错", "", "|长度|量|CALIB1 e/n|新题 e/n|新题整题簇95%|漂移/信息不足|", "|---|---|---|---|---|---|"]
    for length,g in groups.items():
        for name,r in g["rates"].items():
            c=cal["rates"][name]
            lines.append(f"|{length}|{name}|{c['e']}/{c['n']}|{r['e']}/{r['n']}|{r.get('cluster95')}|{'漂移' if r['drift'] else '信息不足' if not r['sufficient'] else '未触发旗标'}|")
    lines += ["", "q_fb仅为明确失败反馈后的自报发现；检测与修复联合表、复查错格上/下/零跳、位置区块、每题自然错数及终止原因均在分析JSON。紧接修好的再错只统计实际可观察下一步，终局尾部不补零。后段位置受幸存者选择，不能当作历史因果效应。",
              "", "## 成本、对照与理论判断", "",
              f"已知usage={a['usage']}；reasoning已包含在completion中，不重复累加。不虚构美元成本。额度检查={[{k:x[k] for k in ('calls','five_hour','week')} for x in a['quotas']]}。",
              "", "工作W按实际模型动作计，服务失败不算动作；每题W、推进和复查、success及W≤5L在分析JSON。磁盘记录另计，不能把两次复查预算叫容量K。Brier/log loss按题、各长度等权；只比较事前冻结对照，不以测试数据重新拟合。无修复预测是不同协议的反事实，不是随机实验对照。",
              "", "旧门槛保持原样：每格仅20题，成功/失败各≥30不可能满足；week1的max(0.03,5SE)仅列辅助拒绝，不能称旧科学准入通过。H50两倍、历史残余效应0.03、rho、自然q、Gamma异质性及容量K收益均未检验。即使本轮G3通过，也不证明生灭链一维充分性或原容量相图成立。",
              "", "复现：`systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 python3 experiments/real/report_holdout_b_fixed.py`（离线，无LLM调用）。输入、代码、预测及门槛冻结见FREEZE_HOLDOUT_B.json；旧CALIB1文件与冻结保持不变。"]
    for row in a["rows"]:
        row["W"]=row["advances"]+row["repair_attempts"]
        row["within_5L_and_success"]=row["status"]=="success" and row["W"]<=5*row["length"]
    obj={k:v for k,v in a.items() if k!="all_events"}
    obj.update(utc=d.now(),conclusion=conclusion,groups=groups,scores=dict(scores),
               G0=cal["G0"],G1=a["smoke_approved"],G2=g2,G3=g3 if g2 else None,G4_drift=drift,G5_observed_audit=True)
    h.write(h.ROOT/"ANALYSIS_HOLDOUT_B.json",obj)
    (h.ROOT/"HOLDOUT_B.md").write_text("\n".join(lines)+"\n")
    print(conclusion);print("AUDIT",a["checks"],"CALLS",s["calls"])
    return obj

if __name__=="__main__":report()
