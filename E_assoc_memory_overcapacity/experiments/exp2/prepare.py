"""Read EXP1 only; write predictions, planning power, then freeze EXP2."""
import csv,hashlib,json,math
from datetime import datetime,timezone
import numpy as np
from design import *
assert not (OUT/'budget_start.json').exists()
assert (OUT/'predata_v1/PREREG_EXP2.md').exists()
old=[json.loads(s) for s in (ROOT/'experiments/exp1/formal_rows.jsonl').read_text().splitlines()]
old=[r for r in old if r['phase']=='B']
by={(r['d'],r['kappa'],r['seed'],r['name']):r for r in old}
def vals(r,metric):
    ranks=np.array(r['ranks']);n=r['n']
    return (ranks-1)/(n-1) if metric=='mean_rank' else (ranks<=math.ceil(float(metric[3:])*n)).astype(float)
def stats(d,k,a,b,metric):
    rr=[];vv=[]
    for seed in range(5):
        u=by[d,k,seed,a];v=by[d,k,seed,b];dif=vals(u,metric)-vals(v,metric)
        rr.append(float(dif.mean()));vv.append(float(dif.var(ddof=1)/128*(1-128/u['n'])))
    return np.array(rr),np.array(vv)
forecasts=[]
# All P3 cells and four P2 failure contexts, fitted on EXP1 only.
contexts=[]
names=sorted({r['name'] for r in old if 'actual' in r['name']})
for k in (.5,1.,2.):
    for name in names:
        for metric in METRICS:contexts.append(('P3',k,name,name.replace('actual','proxy'),metric,0.,None))
for d,k,a,metric in FOCI:
    target=pred(sign_c(a),k,metric)-pred(math.sqrt(2/math.pi),k,metric)
    contexts.append(('P2',k,f'Sign2_a{a:g}_actual','Sign1',metric,target,d))
powers=[]
for kind,k,a,b,metric,target,original_d in contexts:
    st=[stats(d,k,a,b,metric) for d in (128,256)]
    means=[float(t[0].mean()) for t in st];vars_=[float(t[0].var(ddof=1)/5) for t in st]
    f=dict(kind=kind,kappa=k,name=a,baseline=b,metric=metric,target=target,original_d=original_d,
           old_means=means,old_variances=vars_,models={})
    for p in (.5,1.):
        # Fixed equally weighted regression through asymptotic target; no EXP2 refitting.
        xx=np.array([128.**(-p),256.**(-p)]);den=float(xx@xx)
        coef=float(xx@(np.array(means)-target)/den)
        coefvar=float(np.sum(xx*xx*np.array(vars_))/den**2)
        f['models'][str(p)]=dict(b=coef,b_SE=math.sqrt(coefvar),
            predictions={str(d):target+coef/d**p for d in (128,256,512)})
    forecasts.append(f)
    # Random effects power planning, EXP1 query contribution and dataset component explicitly separated.
    for d,(v,w) in zip((128,256),st):
        tau2=max(0.,float(v.var(ddof=1)-w.mean()))
        querynew=float(w.mean())*128/M*(1-M/(k*d*d))/(1-128/(k*d*d))
        sd=math.sqrt(1.25*(tau2+querynew)) # variance inflation for 5-seed pilot
        ns=P2_SEEDS if kind=='P2' else LOW_SEEDS;z=N.inv_cdf(1-.05/4) if kind=='P2' else N.inv_cdf(1-.05/(2*240))
        effect=abs(target) if kind=='P2' else .005
        power=cdf(effect*math.sqrt(ns)/max(sd,1e-15)-z)
        powers.append(dict(kind=kind,kappa=k,name=a,metric=metric,original_d=original_d,d=d,
            pilot_between_seed_sd=float(v.std(ddof=1)),pilot_query_variance=float(w.mean()),
            tau2=tau2,planned_sd=sd,seeds=ns,queries=M,effect=effect,critical_z=z,power=power,
            mde80=(z+N.inv_cdf(.8))*sd/math.sqrt(ns),
            seeds80=math.ceil(((z+N.inv_cdf(.8))*sd/max(effect,1e-15))**2)))
# High-dimensional planning conservatively retains the d256 variance components.
for p in list(powers):
    if p['d']!=256 or p['kappa'] not in (1.,2.):continue
    if p['kind']=='P3' and p['name'] not in ('Sign2_a0.6_actual','Sign2_a1.5_actual','Adam_b0.9_e0.1_T16_actual'):continue
    pp=dict(p);pp.update(d=512,seeds=HIGH_SEEDS,variance_extrapolation='hold d256 components; no dimension gain assumed')
    pp['power']=cdf(pp['effect']*math.sqrt(HIGH_SEEDS)/max(pp['planned_sd'],1e-15)-pp['critical_z'])
    pp['mde80']=(pp['critical_z']+N.inv_cdf(.8))*pp['planned_sd']/math.sqrt(HIGH_SEEDS)
    powers.append(pp)
(OUT/'forecasts.json').write_text(json.dumps(forecasts,indent=2)+'\n')
(OUT/'power.json').write_text(json.dumps(powers,indent=2)+'\n')
p2=[p for p in powers if p['kind']=='P2' and p['d']==p['original_d']]
failures=[r for r in csv.DictReader((ROOT/'experiments/exp1/stability_trends.csv').open()) if r['shrinks']=='False']
(OUT/'old_P3_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
text='''# EXP2 预注册：有限维修正与查询噪声的可分辨性

冻结时间 TIME。最初只含16个低维种子的规划版散列为12f9dc764f65846231000ddc0e5837f33a82fdd3007a934b52b5bf8953e472c5，保存在experiments/exp2/predata_v1；其事前功效表发现P2三项不足80%，因此在任何EXP2随机数据生成前，仅增补四个P2原失败条件到112种子，再冻结本最终版本；无新数据驱动的改门槛或选样。在任何 EXP2 冒烟或正式随机数据生成前冻结。已看过 EXP1；不是盲法首次注册。不修改 EXP1 判决：P2 44/48、P3 曲线76/240失败。EXP1 只用于规划、校准并冻结外推，不并入 EXP2 样本。不调用任何 API、网络或 GPU，不导入其他项目代码。所有项目读写在本题目录；目录外 QED 访问仅在用户确认指定只读例外时进行，否则使用目录内 EXP1 快照并声明时效限制。

## 模型和固定网格

保持 FORMULATION_V2 的 IID N(0,I/d)、真实 S、Z、重复全批量训练、全 n 个训练候选、不白化、不归一化。float64。Sign1 与 Sign2 alpha={0.2,0.6,1.2,1.5}，完整 Adam 16 参数组合的真实和代理均保留；Adam alpha=.2，原始学习率 alpha/sqrt(n)、epsilon=e/sqrt(n)，状态与偏差校正完整。q,c 直接读取冻结 EXP1 moments.json，不拟合 c。

低维层：d={128,256}、kappa={0.5,1,2}、16 个新独立数据种子，各512个无放回均匀训练查询，方法为 Sign1 + 四个两步 Sign 的真实/代理 + 16个 Adam 的真实/代理，共41方法。高维层：d=512、kappa={1,2}、24 个新种子，各512查询，只保留 Sign1、两步Sign alpha=.6/1.5真实/代理、Adam beta=(.9,.999),e=.1,T=16真实/代理，共7方法。不把低维全参数覆盖外推为高维全参数覆盖。

数据 seed entropy=[20261106,2002,d,n,seed,tag]；正式tag=10，冒烟tag=90，seed从0起；X、Y、query分别再追加0、1、2。与 EXP1 及 V2 流不重叠。以8192列固定块生成X/Y，累加真实S、Z并保存抽样查询，再用同一Y流重放全候选；不写大型X/Y缓存、不构造n×n或d²×d²矩阵。各方法同数据同查询。查询512的前128个构成事前嵌套低查询读出；128和512不是两组独立重复。

顺序固定为 seed=0..15，每seed先跑 d128/k1,d256/k1,d512/k1,d128/k2,d256/k2,d512/k2,d128/k.5,d256/k.5；再seed=16..23仅d512/k1,k2。随后补P2定向配置：seed=16..111，依次d128/k1、d128/k2（只测Sign1和Sign2 alpha=.6真实），d256/k1、d256/k2（只测Sign1和Sign2 alpha=1.5真实）。同seed=0..15的P2数据复用全网格记录，不重复计数；P3分析严格只取低维seed<16。最终共528数据配置、5040方法记录。任何预算中止只保留已完成项，缺少指定种子的门槛一律“未完成”，不把缺失当通过，不根据结果增删种子或改网格。

冒烟：稠密独立小模型的全候选秩/查询/候选移除、正缩放、流重放、Sign2阈值、原始W单位Adam与缩放递推、首步实际/代理一致性；另外d=32,k=1及d=128,k=1各1个不交叠冒烟种子、512查询（小模型不足时取全部），含低维全方法。只在检查全部通过后正式，不将冒烟计入估计。代码若修复，记录原因并保留失败输出；不改预注册。

## 事前有限维修正及定量预测

跳点展开给两步Sign的归一化权重误差预测
 E||U2-h2(Z)||F²/(d² q2) = B(alpha,kappa)/sqrt(d)+O(1/d)，
 B=8 alpha³ phi(alpha) sqrt(2/(pi*kappa))/q2，q2=8 alpha² Phi(-alpha)。
推导：残差扰动delta≈N(0,alpha²/(kappa*d))，在±alpha两边界改变符号的概率≈2 phi(alpha) E|delta|，每个翻转的平方权重差为4 alpha²。这是待检验的局部展开，未证明条件独立或余项；不是把它升级为定理。符号跳跃的得分扰动RMS约d^(-1/4)，有符号分布偏差预计首个非零项d^(-1/2)；光滑、e>0的Adam权重平方误差及有符号分布偏差候选为d^(-1)。训练对角MSE应同阶，但不预定精确系数。

Sign2的B值与d=128/256/512预测：
JUMPTABLE

P2每个失败context的实测两步减一步预测为 delta_inf+b_p/d^p；P3真实减代理曲线预测为 b_p/d^p，p预先只取{1/2,1}。delta_inf来自M的高斯秩式。b_p仅用EXP1两维种子均值等权最小二乘、过既定极限截距校准：b=sum_d d^(-p)(mean_d-target)/sum_d d^(-2p)。它们及校准SE、每维数字预测全部存forecasts.json并一同冻结；不得用EXP2重拟合。它们是带旧数据估计误差的外推假设。预测方差包括 b_SE²/d^(2p)，不能把五种子旧拟合当无误差常数。P2数字见：
P2TABLE

## 噪声分解、功效与最小效应

独立重复单位始终为数据种子。对每个真实/代理或两步/一步配对指标，保存512个逐查询差t_i（top为{-1,0,1}、秩为连续配对差）；D=mean t，v=(1-M/n)*samplevar(t)/M 是给定数据集均匀无放回抽样的条件查询方差估计。保存D、|D|、D²-v（后者允许为负、禁止截零），其种子均值估计完整数据集曲线差的二阶矩。前128查询重复同一计算；若真实全数据差近零，E|D|≈sqrt(2v/pi)，M从128到512时查询噪声约减半；有限数据集随机差不会随M消失。这比把所有候选/查询当独立更保守。

功效用EXP1逐查询配对差估计v_old，tau²=max(0,Var_seed(D_old)-mean(v_old))；新种子方差规划为1.25*[tau²+v_old*(128/512)*(1-512/n)/(1-128/n)]。1.25为冻结的方差膨胀；五种子tau估计不精确，规划不是保证。检验均用种子级Student t（NumPy+stdlib以不完全beta求CDF/分位），功效表用正态规划近似；最终另报达成SE与MDE，不追算“观察功效”。固定N，不按结果补样本。P2四个失败context按Bonferroni单侧alpha=.05/4计算，目标效应为M预测方向差；表中为原失败维度。P3按240个曲线context的双侧Bonferroni .05/240、最小有意义有符号偏差0.005计算。高维功效保守沿用d256方差分量而不假设维度带来额外方差下降，只将种子数换成24。完整功效/MDE/80%所需种子见power.json。功效不足80%的项事前标不足，不能把不显著解释为噪声已证实；预算内也不能保证每个微小偏差均可分类。

POWERTABLE

## 固定判决门槛（不得运行后放宽）

G0 完整性：注册、forecasts、power、旧失败列表、moments及执行代码散列运行前后相同；冒烟全部检查成功；所有秩1..n、由原始秩重算指标一致、查询无重复、首步实际=代理；528配置、5040方法记录和预定方法/种子齐全；资源约束满足。一项缺失明示未完成。

G1 P2复核：四个旧失败在原d/k/alpha/q的512查询、112新种子配对差方向正确且单侧98.75% Student t界在预测方向严格离开0才记“方向获确认”，否则“未确认”；另报告d512/24种子同四个context，门槛相同，维度重合的context不重复计独立证据。EX P1失败不改判。所有新方向与逐项SE同时列出。

G2 跳点机制：低维三个负载及高维两个负载的alpha=.6/1.5各格，相对权重MSE/(B/sqrt(d))的双侧95%种子t区间须完全位于[0.75,1.25]才记“定量修正符合”；否则“未满足”。高维同alpha两负载，训练对角MSE/相对权重MSE的种子比率95%区间须完全位于[0.7,1.3]。Adam16代表的相对权重MSE候选K/d，K由EXP1低维d*MSE均值冻结于forecasts.json补充文件 mechanism.json；同样25%等效门槛。其余方法的MSE趋势仅描述，不将代表结果外推全网格。

G3 预测模型区分：P2四个context主读出为d512；P3全部240个context主读出为d256，高维覆盖的24格另列复制。比较H0=target、Hhalf=target+bhalf/sqrt(d)、Hone=target+bone/d。单模型95%家族区间临界t分别使用P2 .05/(4*3)、P3 .05/(240*3)，双侧；含旧校准误差的修正预测使用保守df=4的t临界值，H0使用新种子df=N-1；旧b的显著性亦使用df=4。判兼容当 |Dbar-pred|<=tcrit*sqrt(SE_new²+var_forecast)。H0的var_forecast=0，另两者如上。模型同时兼容不可挑最小误差宣称胜出。支持某修正须H0被拒、相应修正兼容且旧校准b的同级区间不含0；两个修正均兼容则只支持“有限偏差、速率未分辨”。全部被拒须报告“该有限修正族不足”。仅H0兼容也不直接判噪声。

G4 噪声/残留边界：P2在d512，若方向确认且 |Dbar-delta_inf|+tcrit*SE <=0.01（四context双侧Bonferroni .05/4）则记“噪声解释相容到1个百分点”，而非证明旧失败由噪声造成；若修正按G3成立优先记修正。P3在d256及已覆盖高维，只有有符号D的240格Bonferroni区间完全在[-.005,.005]，且 A=mean_seed(D²-v) 的同级单侧上界<=.005²，才记“查询噪声足以解释到0.5个百分点”；否则区分显著残留（A下界>0或D区间不含0）与未分辨。A=t_i构造的去噪统计，种子t区间是有限N近似，保留原值及区间，不宣称严格有限样本覆盖。均报告128→512的|D|缩小比、查询方差及去噪A；其比值不另设事后门槛。

G5 旧趋势原样复核：低维全部20个真实/代理对×3负载×4曲线=240格，以seed平均|D|在256<128为通过，首步精确重合通过；权重/对角MSE同样趋势。列出76个旧失败逐项新状态；这一旧式门槛独立于G3/G4，不能因为“无显著性”改判通过。高维只覆盖代表，不冒称所有76项均做了d512检验。

若G3/G4均未能分类，如实写“未分辨”，不得强迫二选一；P3矩差缩小也不等于曲线项逐格单调或M已证明。没有极值top1、增长T、e趋0、长尾、实用epsilon、语言模型外推。

## 预算、审计及输出

CPU单进程，线程固定OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=2（<=2），NUMEXPR_NUM_THREADS=2，PYTHONDONTWRITEBYTECODE=1；所有数值Python使用/home/liuyibo/venvs/theory/bin/python，外包systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0。块内检索最多1024候选、全部方法查询向量堆叠；RSS>=1700MiB停止。/proc与cgroup仅读资源元数据，Python和系统库作为运行依赖读取，不访问其他项目文件。

从冒烟启动记录budget_start，全部数值工作含冒烟、正式和分析共<=3600秒；冒烟最多120秒，正式截止总起点+3120秒，独立外部timeout3150s，余480秒分析和报告。每大乘法前预留max(3秒,2*历史最大乘法耗时)。到期停止启动新配置并记录缺失；不能为凑齐在预算外追加正式数据。每配置逐方法写JSONL并flush，资源/耗时/流标签/代码与冻结摘要保留。

交付RESULTS_EXP2.md中文先结论，逐项G0..G5、全部旧失败与功效不足/未分辨列表、资源及缺口、QED证据时效、主定理M当前闭环与独立审计需求、主会论文诚实定位。QED自动PASS不直接当人工完成证明；需要核对其problem与本地目标是否一致，以及Adam代理是否满足L2正则性。论文判断仅基于目录内现有查新材料，不声称重新完成外部全文查新。

## 冻结附件

HASHES
'''
jump=[]
for k in (.5,1.,2.):
    for a in (.6,1.5):
        b=jump_B(a,k);jump.append(f'| {k} | {a} | {b:.6f} | '+' | '.join(f'{b/math.sqrt(d):.6f}' for d in (128,256,512))+' |')
jump='| kappa | alpha | B | d128 | d256 | d512 |\n|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(jump)
p2table='| 原d | kappa | alpha | 指标 | 极限差 | d512无修正 | d512+1/√d | d512+1/d |\n|---:|---:|---:|---|---:|---:|---:|---:|\n'
for f in forecasts:
    if f['kind']=='P2':p2table+=f"| {f['original_d']} | {f['kappa']} | {f['name']} | {f['metric']} | {f['target']:.6f} | {f['target']:.6f} | {f['models']['0.5']['predictions']['512']:.6f} | {f['models']['1.0']['predictions']['512']:.6f} |\n"
powertable='| 原d | kappa | 方法 | 指标 | 规划功效 | 80% MDE | 80%所需种子 |\n|---:|---:|---|---|---:|---:|---:|\n'
for p in p2:powertable+=f"| {p['d']} | {p['kappa']} | {p['name']} | {p['metric']} | {p['power']:.3f} | {p['mde80']:.6f} | {p['seeds80']} |\n"
mechanism=[]
for k in (.5,1.,2.):
    rr=[]
    for d in (128,256):rr.extend(d*r['relative_weight_mse'] for r in old if r['d']==d and r['kappa']==k and r['name']=='Adam_b0.9_e0.1_T16_actual')
    mechanism.append(dict(kappa=k,K=float(np.mean(rr))))
(OUT/'mechanism.json').write_text(json.dumps(mechanism,indent=2)+'\n')
files=['FORMULATION_V2.md','experiments/exp1/moments.json','experiments/exp2/forecasts.json','experiments/exp2/power.json','experiments/exp2/old_P3_failures.json','experiments/exp2/mechanism.json','experiments/exp2/design.py','experiments/exp2/run.py','experiments/exp2/analyze.py']
hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
(OUT/'frozen_inputs.json').write_text(json.dumps(hashes,indent=2)+'\n')
text=text.replace('TIME',datetime.now(timezone.utc).isoformat()).replace('JUMPTABLE',jump).replace('P2TABLE',p2table).replace('POWERTABLE',powertable).replace('HASHES','\n'.join(f'- `{f}`: `{h}`' for f,h in hashes.items()))
(ROOT/'PREREG_EXP2.md').write_text(text)
(ROOT/'PREREG_EXP2.sha256').write_text(hashlib.sha256((ROOT/'PREREG_EXP2.md').read_bytes()).hexdigest()+'  PREREG_EXP2.md\n')
print((ROOT/'PREREG_EXP2.sha256').read_text());print(powertable)
print('P3 planning power >=.8:',sum(p['power']>=.8 for p in powers if p['kind']=='P3'),'/',sum(p['kind']=='P3' for p in powers))
