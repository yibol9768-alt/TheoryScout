"""Prewritten EXP2 gates, Student t intervals, raw-rank audit and Chinese report."""
import csv,hashlib,json,math,statistics,time
from collections import defaultdict
import numpy as np
from design import *

# Incomplete beta continued fraction, no SciPy or project dependencies.
def betacf(a,b,x):
    qab=a+b;qap=a+1.;qam=a-1.;c=1.;dd=1.-qab*x/qap
    if abs(dd)<1e-300:dd=1e-300
    dd=1/dd;h=dd
    for j in range(1,401):
        m2=2*j;aa=j*(b-j)*x/((qam+m2)*(a+m2))
        dd=1+aa*dd;dd=dd if abs(dd)>1e-300 else 1e-300
        c=1+aa/c;c=c if abs(c)>1e-300 else 1e-300;dd=1/dd;h*=dd*c
        aa=-(a+j)*(qab+j)*x/((a+m2)*(qap+m2))
        dd=1+aa*dd;dd=dd if abs(dd)>1e-300 else 1e-300
        c=1+aa/c;c=c if abs(c)>1e-300 else 1e-300;dd=1/dd;delta=dd*c;h*=delta
        if abs(delta-1)<3e-14:return h
    raise ArithmeticError('incomplete beta did not converge')
def betai(a,b,x):
    if x<=0:return 0.
    if x>=1:return 1.
    bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    return bt*betacf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-bt*betacf(b,a,1-x)/b
def tcdf(x,df):
    z=.5*betai(df/2,.5,df/(df+x*x));return 1-z if x>=0 else z
_tcache={}
def tq(p,df):
    key=(p,df)
    if key in _tcache:return _tcache[key]
    lo=0.;hi=1.
    while tcdf(hi,df)<p:hi*=2
    for _ in range(65):
        mid=(lo+hi)/2
        if tcdf(mid,df)<p:lo=mid
        else:hi=mid
    _tcache[key]=(lo+hi)/2;return _tcache[key]
def stat(values):
    x=np.asarray(values,dtype=float);return float(x.mean()),float(x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else float('inf')
def critical(alpha,n,two=True):return tq(1-alpha/(2 if two else 1),max(1,n-1))
def metric_values(row,metric,m):
    rr=np.array(row['ranks'][:m]);n=row['n']
    return (rr-1)/(n-1) if metric=='mean_rank' else (rr<=math.ceil(float(metric[3:])*n)).astype(float)
def savecsv(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def check_frozen():
    digest=hashlib.sha256((ROOT/'PREREG_EXP2.md').read_bytes()).hexdigest()
    assert digest==(ROOT/'PREREG_EXP2.sha256').read_text().split()[0]
    for f,h in json.loads((OUT/'frozen_inputs.json').read_text()).items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
    return digest

def main():
    digest=check_frozen()
    assert abs(tcdf(1.,1)-.75)<1e-12 and abs(tq(.975,10)-2.2281388519649385)<1e-9
    rows=[json.loads(s) for s in (OUT/'formal_rows.jsonl').read_text().splitlines()]
    groups=defaultdict(dict)
    for r in rows:
        assert len(r['ranks'])==M and len(set(r['query_indices']))==M
        assert all(1<=v<=r['n'] for v in r['ranks'])
        key=(r['d'],r['kappa'],r['name']);assert r['seed'] not in groups[key];groups[key][r['seed']]=r
        for size,prefix in ((128,'prefix128'),(512,'full')):
            for metric in METRICS:assert abs(float(metric_values(r,metric,size).mean())-r[prefix][metric])<1e-14
        if 'actual' in r['name'] and '_T1_' in r['name']:assert r['weight_mse']==r['diagonal_mse']==0.
    def paired(d,k,name,base,metric,m=512,include_extra=False):
        aa=groups.get((d,k,name),{});bb=groups.get((d,k,base),{});dd=[];vv=[]
        for seed in sorted(aa.keys()&bb.keys()):
            if d<512 and seed>=16 and not include_extra:continue
            u=aa[seed];v=bb[seed];assert u['query_indices']==v['query_indices']
            t=metric_values(u,metric,m)-metric_values(v,metric,m)
            dd.append(float(t.mean()));vv.append(float((1-m/u['n'])*t.var(ddof=1)/m))
        return np.array(dd),np.array(vv)
    forecast=json.loads((OUT/'forecasts.json').read_text());powers=json.loads((OUT/'power.json').read_text())
    p2=[]
    for f in forecast:
        if f['kind']!='P2':continue
        for d in (f['original_d'],512):
            dd,v=paired(d,f['kappa'],f['name'],f['baseline'],f['metric'],include_extra=True);mean,se=stat(dd);n=len(dd)
            crit=critical(.05/4,n,False);sgn=1 if '_a0.6_' in f['name'] else -1
            eqcrit=critical(.05/4,n)
            p2.append(dict(original_d=f['original_d'],d=d,kappa=f['kappa'],name=f['name'],metric=f['metric'],
                seeds=n,planned_seeds=24 if d==512 else P2_SEEDS,mean=mean,SE=se,target=f['target'],
                expected_sign=sgn,one_sided_critical_t=crit,directional_lower=sgn*mean-crit*se,
                direction_confirmed=sgn*mean-crit*se>0 and n==(24 if d==512 else P2_SEEDS),
                noise_compatible_1pp=abs(mean-f['target'])+eqcrit*se<=.01 and n==24,
                achieved_MDE80=(crit+N.inv_cdf(.8))*se))
    savecsv('P2_gates.csv',p2)
    models=[];noise=[];pair_records=[]
    for f in forecast:
        dims=(512,) if f['kind']=='P2' else (256,512)
        for d in dims:
            if (d,f['kappa'],f['name']) not in groups:continue
            dd,v=paired(d,f['kappa'],f['name'],f['baseline'],f['metric'],include_extra=f['kind']=='P2');n=len(dd);mean,se=stat(dd)
            total=4 if f['kind']=='P2' else 240;tc=critical(.05/(total*3),n)
            expected=24 if d==512 else (P2_SEEDS if f['kind']=='P2' else 16);complete=n==expected
            compatible=[];support=[]
            for model in ('zero','0.5','1.0'):
                fpred=f['target'] if model=='zero' else f['models'][model]['predictions'][str(d)]
                fvar=0. if model=='zero' else f['models'][model]['b_SE']**2/d**(2*float(model))
                model_tc=tc if model=='zero' else critical(.05/(total*3),5)
                error=mean-fpred;bound=model_tc*math.sqrt(se*se+fvar);ok=abs(error)<=bound and complete
                b_sig=False if model=='zero' else abs(f['models'][model]['b'])>model_tc*f['models'][model]['b_SE']
                models.append(dict(kind=f['kind'],original_d=f['original_d'],d=d,kappa=f['kappa'],name=f['name'],metric=f['metric'],
                    model=model,seeds=n,complete=complete,mean=mean,SE=se,prediction=fpred,forecast_SE=math.sqrt(fvar),
                    error=error,bound=bound,critical_t=model_tc,compatible=ok,calibration_b_significant=b_sig))
                if ok:compatible.append(model)
                if ok and b_sig and model!='zero':support.append(model)
            if not complete:category='未完成'
            elif 'zero' not in compatible and support:category='有限修正支持_'+','.join(support)
            elif not compatible:category='修正族不足'
            else:category='模型未分辨'
            if f['kind']=='P3':
                tcv=critical(.05/240,n);taa=critical(.05/240,n,False)
                aa=dd*dd-v;amean,ase=stat(aa);a_hi=amean+taa*ase;a_lo=amean-taa*ase
                eq=abs(mean)+tcv*se<=.005
                zeroexact='_T1_' in f['name'] and all(dd==0.)
                enough=eq and a_hi<=.005**2 and complete
                if zeroexact:nclass='首步精确重合'
                elif enough:nclass='查询噪声相容_0.5pp'
                elif complete and (a_lo>0 or abs(mean)>tcv*se):nclass='显著有限维残留'
                else:nclass='未分辨'
                low,vlow=paired(d,f['kappa'],f['name'],f['baseline'],f['metric'],128)
                abs128=float(np.mean(np.abs(low)));abs512=float(np.mean(np.abs(dd)))
                noise.append(dict(d=d,kappa=f['kappa'],name=f['name'],metric=f['metric'],seeds=n,complete=complete,
                    mean=mean,SE=se,signed_equivalence=eq,A=amean,A_SE=ase,A_lower=a_lo,A_upper=a_hi,
                    query_noise_variance=float(v.mean()),prefix128_query_variance=float(vlow.mean()),
                    abs128=abs128,abs512=abs512,abs_ratio_512_over_128=abs512/abs128 if abs128 else '',
                    folded_zero_query_floor=float(np.mean(np.sqrt(2*v/math.pi))),
                    noise_class=nclass,model_class=category,compatible_models=','.join(compatible),
                    achieved_MDE80=(tcv+N.inv_cdf(.8))*se))
                for seed,di,vi in zip(sorted(groups[d,f['kappa'],f['name']]),dd,v):
                    pair_records.append(dict(d=d,kappa=f['kappa'],name=f['name'],metric=f['metric'],seed=seed,D=float(di),v=float(vi),debiased_square=float(di*di-vi)))
    savecsv('model_gates.csv',models);savecsv('P3_noise_gates.csv',noise);savecsv('P3_seed_decomposition.csv',pair_records)
    # Mechanistic prediction gates (no coefficient refit).
    mechanism=json.loads((OUT/'mechanism.json').read_text());mech=[];diaggates=[]
    for (d,k,name),rr in sorted(groups.items()):
        if 'actual' not in name:continue
        rr={seed:r for seed,r in rr.items() if d==512 or seed<16}
        sign=name.startswith('Sign2_') and ('_a0.6_' in name or '_a1.5_' in name)
        adam16=name=='Adam_b0.9_e0.1_T16_actual'
        if not (sign or adam16):continue
        prediction=jump_B(float(name.split('_a')[1].split('_')[0]),k)/math.sqrt(d) if sign else next(p['K'] for p in mechanism if p['kappa']==k)/d
        ratios=[r['relative_weight_mse']/prediction for r in rr.values()];mean,se=stat(ratios);tc=critical(.05,len(rr))
        item=dict(d=d,kappa=k,name=name,seeds=len(rr),prediction=prediction,observed=float(np.mean([r['relative_weight_mse'] for r in rr.values()])),
            ratio=mean,ratio_SE=se,ratio_lower=mean-tc*se,ratio_upper=mean+tc*se,
            pass_gate=mean-tc*se>=.75 and mean+tc*se<=1.25 and len(rr)==(24 if d==512 else 16))
        mech.append(item)
        if sign and d==512:
            ratios=[r['normalized_diagonal_mse']/r['relative_weight_mse'] for r in rr.values()];mean,se=stat(ratios)
            diaggates.append(dict(d=d,kappa=k,name=name,seeds=len(rr),ratio=mean,SE=se,lower=mean-tc*se,upper=mean+tc*se,
                                  pass_gate=mean-tc*se>=.7 and mean+tc*se<=1.3 and len(rr)==24))
    savecsv('mechanism_gates.csv',mech);savecsv('diagonal_gates.csv',diaggates)
    trends=[]
    for (d,k,name),rs in sorted(groups.items()):
        if d!=128 or 'actual' not in name:continue
        for metric in ('weight_mse','diagonal_mse')+METRICS:
            values=[];complete=True
            for dim in (128,256):
                if metric in METRICS:
                    dd,v=paired(dim,k,name,name.replace('actual','proxy'),metric);values.append(float(np.mean(np.abs(dd))))
                    complete=complete and len(dd)==16
                else:
                    rr={seed:r for seed,r in groups.get((dim,k,name),{}).items() if seed<16};values.append(float(np.mean([r[metric] for r in rr.values()])))
                    complete=complete and len(rr)==16
            exact='_T1_' in name and values==[0.,0.]
            trends.append(dict(kappa=k,name=name,metric=metric,value128=values[0],value256=values[1],
                               complete=complete,pass_trend=complete and (values[1]<values[0] or exact)))
    savecsv('all_trends.csv',trends)
    oldfails=json.loads((OUT/'old_P3_failures.json').read_text());oldretest=[]
    for f in oldfails:
        metric=f['metric'].replace('_absolute_difference','')
        r=next(t for t in trends if t['kappa']==float(f['kappa']) and t['name']==f['name'] and t['metric']==metric)
        ng=next(t for t in noise if t['d']==256 and t['kappa']==float(f['kappa']) and t['name']==f['name'] and t['metric']==metric)
        oldretest.append(dict(**r,EXP1_value128=f['value128'],EXP1_value256=f['value256'],
                              noise_class=ng['noise_class'],model_class=ng['model_class']))
    savecsv('old_76_retest.csv',oldretest)
    diag=[]
    for (d,k,name),rs in sorted(groups.items()):
        vals=list(rs.values());item=dict(d=d,kappa=k,name=name,seeds=len(vals),score_mean_target=vals[0]['c']/math.sqrt(k))
        for key,values in [('true_mean',[np.mean(r['true_scores']) for r in vals]),('true_variance',[np.var(r['true_scores'],ddof=1) for r in vals]),
                            ('wrong_mean',[r['wrong_mean'] for r in vals]),('wrong_second',[r['wrong_second'] for r in vals])]:
            item[key],item[key+'_SE']=stat(values)
        diag.append(item)
    savecsv('score_diagnostics.csv',diag)
    smoke=json.loads((OUT/'smoke_metadata.json').read_text());meta=json.loads((OUT/'formal_metadata.json').read_text())
    exact_pairs=0
    for (d,k,name),rs in groups.items():
        if '_T1_actual' in name:
            for seed,r in rs.items():assert r['ranks']==groups[d,k,name.replace('actual','proxy')][seed]['ranks'];exact_pairs+=1
    audit=dict(prereg_sha256=digest,raw_rows=len(rows),rank_metric_consistency=True,query_uniqueness=True,
               paired_query_consistency=True,first_step_exact_pairs=exact_pairs,student_t_reference_checks=True,
               expected_rows=5040,expected_configurations=528,complete=meta['completed']==528 and len(rows)==5040,
               resource_limits_verified=all(m['resource_start']['memory.max']=='2147483648' and m['resource_start']['memory.swap.max']=='0' for m in (smoke,meta)),
               numerical_seconds_so_far=time.time()-json.loads((OUT/'budget_start.json').read_text())['start_unix'])
    (OUT/'integrity_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    def counts(rr,key):
        out=defaultdict(int)
        for r in rr:out[str(r[key])]+=1
        return dict(out)
    p3low=[r for r in noise if r['d']==256];p3high=[r for r in noise if r['d']==512]
    summary=dict(audit=audit,P2_confirmed=sum(r['direction_confirmed'] for r in p2),P2_total=len(p2),
        mechanism_pass=sum(r['pass_gate'] for r in mech),mechanism_total=len(mech),
        diagonal_pass=sum(r['pass_gate'] for r in diaggates),diagonal_total=len(diaggates),
        P3_low_noise=counts(p3low,'noise_class'),P3_high_noise=counts(p3high,'noise_class'),
        P3_low_models=counts(p3low,'model_class'),P3_high_models=counts(p3high,'model_class'),
        curve_trends_pass=sum(r['pass_trend'] for r in trends if r['metric'] in METRICS),curve_trends_total=sum(r['metric'] in METRICS for r in trends),
        moment_trends_pass=sum(r['pass_trend'] for r in trends if r['metric'] not in METRICS),
        old76_retest_pass=sum(r['pass_trend'] for r in oldretest),old76_retest_total=len(oldretest),
        P2_planning_underpowered=[p for p in powers if p['kind']=='P2' and p['d']==p['original_d'] and p['power']<.8],
        P3_low_planning_power80=sum(p['power']>=.8 for p in powers if p['kind']=='P3' and p['d']==256),
        metadata=meta)
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# EXP2 验证结果','',
        f"结论：EXP2 的四个旧 P2 失败在原维度确认 {sum(r['direction_confirmed'] for r in p2 if r['d']==r['original_d'])}/4，在 d=512 确认 {sum(r['direction_confirmed'] for r in p2 if r['d']==512)}/4。跳点/光滑有限维修正的权重误差定量门槛通过 {summary['mechanism_pass']}/{len(mech)}，训练对角比率通过 {summary['diagonal_pass']}/{len(diaggates)}。P3旧式曲线趋势通过 {summary['curve_trends_pass']}/240，原76失败中 {summary['old76_retest_pass']} 项新样本下缩小；旧失败保留原判决。有限偏差、查询噪声和未分辨项分别报告，不能将修正机制成立等同全部曲线偏差由它解释。主定理 M 的结论仍须结合证明审计，数值实验不能证明它；当前成果还不足以单独撑起一篇成熟的主会理论论文。",'',
        '## G0：冻结、实现与资源','',
        f"预注册 SHA-256：`{digest}`。预注册、数值预测、功效附件、机制常数、原理论/积分及运行分析代码在冒烟前冻结，分析时逐项复核一致。冒烟检查全部通过：流重放与独立稠密全候选秩、显式移除正确候选、Sign阈值、Adam原始W单位递推、首步实际/代理一致性、正缩放。原始秩及128/512查询指标逐项重算通过；首步零差对 {exact_pairs}。所有方法共用数据和查询，查询/候选没有作为独立种子计数。",'',
        f"完成 {meta['completed']}/528 数据配置、{len(rows)}/5040 方法记录。冒烟 {smoke['elapsed_seconds']:.2f} 秒；正式 {meta['elapsed_seconds']:.2f} 秒；分析到此累计数值墙钟 {audit['numerical_seconds_so_far']:.2f} 秒。峰值RSS {max(smoke['max_rss_MiB'],meta['max_rss_MiB']):.2f} MiB；最大单次矩阵乘法 {meta['max_single_matmul_seconds']:.3f} 秒。CPU、Python `{meta['python']}`、BLAS/OMP/MKL均2线程。systemd scope实测memory.max=2147483648、memory.swap.max=0，未发生OOM（见metadata）。G0覆盖判决：{'通过' if audit['complete'] and audit['resource_limits_verified'] else '未全部完成'}。未调用网络/LLM/API，未导入其他项目文件。",'',
        '## 事前功效和实际精度','',
        f"P3使用16个低维种子、24个高维种子和512查询；P2四个原失败条件使用112个低维种子；EXP1为5种子、128查询。功效采用EXP1随机效应分解并将方差膨胀25%，正态规划近似，最终检验使用种子级Student t。原失败四项中 {len(summary['P2_planning_underpowered'])} 项事前规划功效不足80%；P3 d256的240格中仅 {summary['P3_low_planning_power80']} 格对0.5个百分点偏差规划功效达到80%（包含精确首步零差项，不能把它们计作有信息的功效成功）。五种子pilot估计误差很大，功效不足项见power.json，达成MDE见各门槛CSV。未用不显著倒推噪声。",'',
        '| 原d | kappa | alpha | 指标 | 规划功效 | 80% MDE | 所需种子 |','|---:|---:|---:|---|---:|---:|---:|']
    for p in powers:
        if p['kind']=='P2' and p['d']==p['original_d']:lines.append(f"| {p['d']} | {p['kappa']} | {p['name']} | {p['metric']} | {p['power']:.3f} | {p['mde80']:.6f} | {p['seeds80']} |")
    lines+=['','## G1：P2四个旧失败的独立复核','',
        '差为两步减一步。alpha=.6期待正，alpha=1.5期待负；确认须单侧Bonferroni四项的98.75%种子t界严格离零，方向正确但区间跨零也不确认。','',
        '| 原d | 新d | kappa | alpha/指标 | 均值 ± SE | 极限差 | 方向确认 | 噪声相容≤1pp（仅512） |',
        '|---:|---:|---:|---|---:|---:|---|---|']
    for r in p2:lines.append(f"| {r['original_d']} | {r['d']} | {r['kappa']} | {r['name']}/{r['metric']} | {r['mean']:.6f} ± {r['SE']:.6f} | {r['target']:.6f} | {'是' if r['direction_confirmed'] else '未确认'} | {r['noise_compatible_1pp'] if r['d']==512 else '—'} |")
    lines+=['','## G2：可预测的有限维修正','',
        '两步Sign的局部跳点展开预测相对权重MSE=B/√d+O(1/d)。B直接由alpha和kappa算出，不拟合实验；Adam16的K/d只在EXP1校准，随后冻结。每格95%种子t比率区间必须全部落入[.75,1.25]才通过。该检验支持平方误差的机制，不直接证明秩曲线首项系数。','',
        '| d | kappa | 方法 | 实测相对MSE | 预报MSE | 实测/预报95%区间 | 通过 |',
        '|---:|---:|---|---:|---:|---:|---|']
    for r in mech:lines.append(f"| {r['d']} | {r['kappa']} | {r['name']} | {r['observed']:.6f} | {r['prediction']:.6f} | [{r['ratio_lower']:.3f},{r['ratio_upper']:.3f}] | {r['pass_gate']} |")
    lines+=['',f"d512的归一化训练对角MSE/相对权重MSE95%区间全部位于[.7,1.3]者 {summary['diagonal_pass']}/4。这里只抽样512查询，不是完整n查询的式(17)证明；详情 diagonal_gates.csv。",'',
        '## G3：冻结的曲线修正能否胜过零偏差模型','',
        '三个模型同时比较：极限值、EXP1校准的b/√d、EXP1校准的b/d；区间计入旧拟合误差并按全部context×三模型校正。不能在两个修正都兼容时选择误差较小者宣称速率已确定。','',
        f"P3 d256：{json.dumps(summary['P3_low_models'],ensure_ascii=False)}；d512代表24格：{json.dumps(summary['P3_high_models'],ensure_ascii=False)}。逐模型预测、误差、临界t、校准SE及判决见 model_gates.csv。",'',
        '| P2原d | kappa | 方法/指标 | d512实测 | 极限 | 1/√d预报 | 1/d预报 | 兼容模型 |',
        '|---:|---:|---|---:|---:|---:|---:|---|']
    for f in forecast:
        if f['kind']!='P2':continue
        gg=[r for r in models if r['kind']=='P2' and r['original_d']==f['original_d'] and r['kappa']==f['kappa'] and r['name']==f['name'] and r['metric']==f['metric']]
        lines.append(f"| {f['original_d']} | {f['kappa']} | {f['name']}/{f['metric']} | {gg[0]['mean']:.6f} | {f['target']:.6f} | {f['models']['0.5']['predictions']['512']:.6f} | {f['models']['1.0']['predictions']['512']:.6f} | {','.join(r['model'] for r in gg if r['compatible'])} |")
    lines+=['','## G4：查询噪声与有限数据集残留','',
        '逐查询配对差的条件方差v含无放回有限总体修正。A=种子均值(D²-v)保留负值；A的正下界表示查询抽样以外仍有真实有限数据集曲线差。只有有符号偏差等效在±.005且A上界≤.005²，才允许“查询噪声足以解释到0.5个百分点”；精确首步单列。区间用种子t近似，小N下没有严格覆盖保证。','',
        f"d256：{json.dumps(summary['P3_low_noise'],ensure_ascii=False)}；d512：{json.dumps(summary['P3_high_noise'],ensure_ascii=False)}。未分辨项保留，不把P>阈值解释成零效应。P3_noise_gates.csv同时给128→512查询的绝对差比、噪声底、去噪平方差和所有区间。",'',
        '## G5：EXP1冻结趋势原样复核','',
        f"全部240曲线项中 {summary['curve_trends_pass']} 项缩小或精确重合、{240-summary['curve_trends_pass']} 项仍失败；120矩误差趋势中 {summary['moment_trends_pass']} 项通过。原76曲线失败的新数据复核 {summary['old76_retest_pass']} 项缩小、{76-summary['old76_retest_pass']} 项未缩小，逐项见 old_76_retest.csv。该旧式门槛不加显著性豁免；G3/G4也不能回写EXP1判决。高维只测3个代表方法×两负载，未覆盖全部旧76项。",'',
        '## 分布诊断与解释边界','',
        '正确得分均值/方差和全候选错误得分一二阶矩保存在score_diagnostics.csv；没有为这些诊断追加通过门槛。为集中预算于精确全候选排名，本轮未重算完整错误候选CDF网格，因此不报告(8)已经验验证。所有正式查询都和训练集耦合，未替换成独立测试标签。既有P1/P4成功仍不等于正确方差、经验CDF与真实迭代转移已经完成数学证明。','',
        '## 主定理M的证明与论文判断','',
        '本节的QED源码核查、缺口和投稿判断由研究员在读取允许的证明证据后补全；不得把此占位句当作最终交付。','',
        '## 产物','',
        '固定设计及数字预测：PREREG_EXP2.md、PREREG_EXP2.sha256、forecasts.json、mechanism.json、power.json、frozen_inputs.json。原始数据：smoke/formal_rows.jsonl；运行资源：metadata、events及日志。全部门槛：P2_gates.csv、mechanism_gates.csv、diagonal_gates.csv、model_gates.csv、P3_noise_gates.csv、all_trends.csv；76项逐项：old_76_retest.csv。代码run.py/design.py/analyze.py拒绝覆盖原始run；请勿直接重跑正式命令覆盖文件。']
    (ROOT/'RESULTS_EXP2.md').write_text('\n'.join(lines)+'\n')
    check_frozen();print(json.dumps({k:v for k,v in summary.items() if k not in ('metadata','P2_planning_underpowered')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
