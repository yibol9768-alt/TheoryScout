import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    assert os.environ.get(k)=='2'
import json, math, hashlib, csv, time
from pathlib import Path
import numpy as np
from scipy.stats import t as student

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'experiments/x1'
os.environ['MPLCONFIGDIR']=str(OUT/'X1_mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

M=['GD','Adam2','Adam1','Adam.1','Sign']
PAIRS=[('GD','Sign'),('Adam2','Adam.1'),('Adam.1','Sign')]
R=[.5,0.,-.5]
POP=json.loads((OUT/'X1_population.json').read_text())
TARGET=POP['target']
P={(p['ir'],p['method']):p for p in POP['rows']}
DES=json.loads((OUT/'X1_design_v3.json').read_text())

def load(name):return [json.loads(l) for l in (OUT/name).read_text().splitlines()]
def dump(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def writecsv(name,rows):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(vals,family):
    v=np.array(vals,dtype=float);n=len(v);mean=float(v.mean());sd=float(v.std(ddof=1));se=sd/math.sqrt(n)
    crit=float(student.ppf(1-.05/(2*family),n-1))
    return dict(n=n,mean=mean,sd=sd,se=se,lo=mean-crit*se,hi=mean+crit*se,halfwidth=crit*se)

rows=load('X1_formal_rows.jsonl')
groups={}
for row in rows:groups.setdefault((row['d'],row['ir'],row['method']),{})[row['seed']]=row
primary=[];matching=[];adjacent=[];newpairs=[]
for d in DES['dimensions']:
    for ir in range(3):
        for method in M:
            g=groups[(d,ir,method)]
            z=stats([v['new']-TARGET for v in g.values()],30 if d in [128,256] else 15)
            matching.append(dict(d=d,r=R[ir],ir=ir,method=method,population_target=TARGET,
                observed_new=TARGET+z['mean'],practical_match=z['lo']>=-.015 and z['hi']<=.015,**z))
        for a,b in PAIRS:
            ga=groups[(d,ir,a)];gb=groups[(d,ir,b)];seeds=sorted(set(ga)&set(gb))
            z=stats([ga[s]['old']-gb[s]['old'] for s in seeds],18 if d in [128,256] else 9)
            pred=P[(ir,a)]['old_recall']-P[(ir,b)]['old_recall']
            direction='正向确认' if z['lo']>0 else ('反向确认' if z['hi']<0 else '未分辨')
            coverage=z['lo']<=pred<=z['hi']
            primary.append(dict(d=d,r=R[ir],ir=ir,a=a,b=b,prediction=pred,
                direction=direction,prediction_covered=coverage,strong=z['lo']>0 and coverage,
                target_precision_met=z['halfwidth']<=min(.002,.4*pred),**z))
            newpairs.append(dict(d=d,r=R[ir],a=a,b=b,**stats([ga[s]['new']-gb[s]['new'] for s in seeds],18 if d in [128,256] else 9)))
        for a,b in zip(M[:-1],M[1:]):
            ga=groups[(d,ir,a)];gb=groups[(d,ir,b)];seeds=sorted(set(ga)&set(gb))
            adjacent.append(dict(d=d,r=R[ir],a=a,b=b,
                prediction=P[(ir,a)]['old_recall']-P[(ir,b)]['old_recall'],
                **stats([ga[s]['old']-gb[s]['old'] for s in seeds],1)))

sec=load('X1_secondary_rows.jsonl');sg={}
for r in sec:sg.setdefault((r['ir'],r['method']),{})[r['seed']]=r
secondary=[]
for ir in range(3):
    for a,b in PAIRS:
        ga=sg.get((ir,a),{});gb=sg.get((ir,b),{});seeds=sorted(set(ga)&set(gb))
        if len(seeds)>=2:
            secondary.append(dict(d=256,r=R[ir],a=a,b=b,
                population_deterministic_prediction=P[(ir,a)]['old_recall']-P[(ir,b)]['old_recall'],
                **stats([ga[s]['old']-gb[s]['old'] for s in seeds],1)))

# Check every stored paired query indicator against every scalar recall and rank.
audits=[]
for d in DES['dimensions']:
    for ir in range(3):
        seeds=sorted(groups[(d,ir,'GD')])
        for seed in seeds:
            path=OUT/f'X1_formal_d{d}_r{ir}_s{seed}.npz'
            with np.load(path) as data:
                ids=data['ids'];assert len(ids)==2048 and len(np.unique(ids))==len(ids)
                assert ids.min()>=0 and ids.max()<d*d//4
                for method in M:
                    row=groups[(d,ir,method)][seed]
                    assert row['phase']==22 and abs(row['amplitude']-P[(ir,method)]['amplitude'])<1e-14
                    for task in ['old','new']:
                        rank=data[method+'_'+task+'_ranks'];ok=data[method+'_'+task+'_success']
                        assert np.array_equal(ok,rank<=math.ceil(.1*(d*d//4)))
                        assert abs(ok.mean()-row[task])<1e-15
                        assert rank.min()>=1 and rank.max()<=d*d//4
                for a,b in PAIRS:
                    diff=data[a+'_old_success'].astype(int)-data[b+'_old_success'].astype(int)
                    delta=groups[(d,ir,a)][seed]['old']-groups[(d,ir,b)][seed]['old']
                    assert abs(diff.mean()-delta)<1e-15
            audits.append([d,ir,seed])

initial=json.loads((OUT/'X1_initial_hashes.json').read_text())
source_check={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in initial.items()}
assert all(ok for path,ok in source_check.items() if not path.startswith('paper/'))
for manifest in ['X1_PREREG.sha256','X1_DESIGN.sha256']:
    for line in (ROOT/manifest).read_text().splitlines():
        digest,path=line.split('  ',1)
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
fmeta=json.loads((OUT/'X1_formal_meta.json').read_text())
smeta=json.loads((OUT/'X1_secondary_meta.json').read_text())
assert fmeta['complete']
assert smeta['formal_plus_secondary_seconds']<=7200
for meta in [fmeta,smeta,json.loads((OUT/'X1_pilot_meta.json').read_text())]:
    assert meta['memory_max']=='3221225472' and meta['memory_swap_max']=='0'
    assert all(z['num_threads']<=2 for z in meta['threads'])

for name,data in [('X1_primary.csv',primary),('X1_new_matching.csv',matching),('X1_adjacent_descriptive.csv',adjacent),
                  ('X1_primary_new_differences.csv',newpairs),('X1_secondary_contrasts.csv',secondary)]:writecsv(name,data)
dump('X1_summary.json',dict(primary=primary,matching=matching,adjacent=adjacent,secondary=secondary,
    audited_datasets=len(audits),source_hashes_unchanged=source_check))

# Two standalone figures: paired old-recall contrasts and new matching errors.
colors=['#1764a0','#c45b16']
fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True)
for ir,ax in enumerate(axs):
    for j,d in enumerate([128,256]):
        rr=[z for z in primary if z['ir']==ir and z['d']==d]
        y=np.arange(3)+(j-.5)*.16
        ax.errorbar([z['mean'] for z in rr],y,xerr=[[z['mean']-z['lo'] for z in rr],[z['hi']-z['mean'] for z in rr]],fmt='o',capsize=3,color=colors[j],label=f'd={d}, simultaneous 95% CI')
    rr=[z for z in primary if z['ir']==ir and z['d']==128]
    ax.scatter([z['prediction'] for z in rr],np.arange(3),marker='x',s=65,color='black',label='Population prediction',zorder=4)
    ax.axvline(0,color='grey',lw=.8);ax.set_title(f'r={R[ir]:g}');ax.set_xlabel('Paired old-task recall difference')
    ax.set_yticks(np.arange(3),['GD - Sign','Adam(2) - Adam(.1)','Adam(.1) - Sign']);ax.invert_yaxis()
axs[-1].legend(fontsize=7,loc='best');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/f'X1_old_contrasts.{ext}',dpi=200)
plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True)
for ir,ax in enumerate(axs):
    ax.axvspan(-.015,.015,color='grey',alpha=.12)
    for j,d in enumerate([128,256]):
        rr=[z for z in matching if z['ir']==ir and z['d']==d]
        y=np.arange(5)+(j-.5)*.16
        ax.errorbar([z['mean'] for z in rr],y,xerr=[[z['mean']-z['lo'] for z in rr],[z['hi']-z['mean'] for z in rr]],fmt='o',capsize=3,color=colors[j],label=f'd={d}')
    ax.axvline(0,color='black',lw=.8);ax.set_title(f'r={R[ir]:g}');ax.set_xlabel('New-task recall minus fixed target')
    ax.set_yticks(np.arange(5),M);ax.invert_yaxis()
axs[-1].legend();fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/f'X1_new_matching.{ext}',dpi=200)
plt.close(fig)

base=[z for z in primary if z['d'] in [128,256]]
direction=sum(z['lo']>0 for z in base);cover=sum(z['prediction_covered'] for z in base);strong=sum(z['strong'] for z in base)
precision=sum(z['target_precision_met'] for z in base)
practical=sum(z['practical_match'] for z in matching if z['d'] in [128,256])
bad=[z for z in base if not z['strong']]
newmax=max(abs(z['mean']) for z in matching)
point_order=[]
for d in [128,256]:
    for ir in range(3):
        means={m:np.mean([v['old'] for v in groups[(d,ir,m)].values()]) for m in M}
        point_order.append(dict(d=d,r=R[ir],order=sorted(M,key=lambda m:-means[m]),means=means))

text='# X1 结果：总体首次匹配的新任务检索与旧任务排序\n\n'
text+=f'**结论：18项预注册主要配对旧recall差中，{direction}项确认理论正方向，{cover}项校正区间覆盖总体预测，{strong}项同时满足两条；按预注册全族判据，'+('主要预测得到一致强支持。' if strong==18 else '不能宣称全部主要预测得到一致强支持。')+'**\n\n'
text+=f'新任务固定目标recall={TARGET:.9f}；30格中{practical}格的校正区间完整落在±.015实用匹配带，最大绝对平均匹配误差为{newmax:.6f}。确定性总体匹配并不保证有限维经验recall相同；以下保留所有有限维偏差，未据正式数据重调参数。\n\n'
text+='## 设计与冻结\n\n'
text+='先读指定五份材料，主参数来自附录K/推论7.4(ii)总体首次分支，T=1，kappa=.25，rho0=rho1=.1，u=.68，rx=1、ry=r∈{.5,0,−.5}；使用实际残差Z1−Z0S1。GD、Sign、Adam(e=.1,1,2)在同一数据集和2048个相同查询上比较，全部训练候选精确排名。Adam状态重置、无衰减，稳定项按e sigma缩放。\n\n'
text+='[预注册](../../X1_PREREG.md)在任何X1随机数据之前冻结，SHA256：`'+initial['X1_PREREG.md']+'`。仅总体积分先于冻结；独立实现核查和pilot都在冻结后。冻结文档、设计、执行及原分析脚本、总体参数及四份输入的哈希在最终审计中不变；paper/sections/fine_tuning.tex在运行期间被外部进程改动，本任务没有写入它。初始哈希30ed2074099f96396810b85a2ecf8b09f2870535d5d25c3bf9c41e819cf24c87，末次观测哈希7633e6d6cdd9e3e31f6c354ade2009bd2ace0b64deb93350dea1bb07c4bdc062。根据首次读取内容重建的初始副本已与初始SHA256精确匹配，两个副本及三处措辞/标点变化保存在X1_fine_tuning_initial.tex、X1_fine_tuning_current.tex、X1_source_change.diff。数学模型、定理及匹配推论内容未变；外部变化未影响冻结参数或计算。冻结前缺少threadpoolctl的启动失败及无需安装包的修复另见X1_SETUP.md；原脚本保留，新建v2核查及试算；plan输出时numpy布尔值序列化失败，保留残缺输出，新建v3仅修复序列化和输出文件名后执行正式。详见X1_EXECUTION_NOTE.md。\n\n'
text+=f'试算独立48个数据集，每(d,r)8seed，不混入正式结果。按照冻结的方差90%上界、校正CI半宽≤min(.002,.4×预测差)、90%方向功效规则，最终每r正式seed数为{DES["counts"]}；未截断需求为{DES["intended"]}。实际达到预设半宽的主要端点为{precision}/18。完整功效/精度规划见[X1_DESIGN.md](../../X1_DESIGN.md)和X1_design_v3.json。\n\n'
text+=f'd512加入={DES["d512_added"]}：pilot d256最大新匹配偏差{DES["pilot_max_new_mismatch"]:.6f}，增加512的保守成本{DES["d512_extra_seconds"]:.1f}秒，基础正式及次要分析规划成本{DES["estimated_seconds"]:.1f}秒；按冻结的偏差与预算双条件执行，无事后维数选择。\n\n'
text+='主要只检验三个差：GD−Sign、Adam(2)−Adam(.1)、Adam(.1)−Sign，跨3个r和2个d共18项；双侧Bonferroni同时95%配对seed Student-t区间。方向确认要求下界>0；强支持另要求总体预测在区间内。查询依赖通过先求每seed配对差处理，独立样本量是seed数。CI是seed均值的常规t近似，不是有限样本精确保证。\n\n'
text+='## 主要旧任务recall配对差\n\n'
text+='单位均为recall比例；乘100即百分点。各格n为独立seed数。\n\n|d|r|配对差|n|总体预测|估计|校正95%区间|方向|覆盖预测|\n|---|---|---|---|---|---|---|---|---|\n'
for z in base:
    text+=f'|{z["d"]}|{z["r"]:g}|{z["a"]}−{z["b"]}|{z["n"]}|{z["prediction"]:.6f}|{z["mean"]:.6f}|[{z["lo"]:.6f}, {z["hi"]:.6f}]|{z["direction"]}|'+('是' if z['prediction_covered'] else '否')+'|\n'
text+='\n![旧任务配对差与预测](experiments/x1/X1_old_contrasts.png)\n\n'
if bad:
    text+='未满足“方向确认且覆盖预测”的端点：'+ '；'.join(f'd={z["d"]}, r={z["r"]:g}, {z["a"]}−{z["b"]}' for z in bad)+'。这些失败不被删除，不修改预注册，也不能仅凭有限维偏差判定渐近定理错误。\n\n'
text+='## 新任务匹配误差\n\n'
text+='下面误差为经验新recall−固定总体目标，使用独立30项族Bonferroni同时95%区间。“匹配”要求整个区间位于[−.015,.015]。\n\n|d|r|方法|新recall|误差|校正95%误差区间|实用匹配|\n|---|---|---|---|---|---|---|\n'
for z in matching:
    text+=f'|{z["d"]}|{z["r"]:g}|{z["method"]}|{z["observed_new"]:.6f}|{z["mean"]:+.6f}|[{z["lo"]:.6f}, {z["hi"]:.6f}]|'+('是' if z['practical_match'] else '未证实')+'|\n'
text+='\n![新任务匹配误差](experiments/x1/X1_new_matching.png)\n\n'
text+='相同主要配对的新recall差及其校正区间在X1_primary_new_differences.csv。它们量化有限维方法间未完全匹配的程度；主要对比解释为“总体参数匹配条件下的实际有限维差”，不能重新解释为每份样本经验新recall严格相等时的差。\n\n'
text+='## 完整方法序列与小间隔\n\n'
text+='全部五方法都实际运行。下列仅列旧recall的样本均值次序；其全部相邻差、总体预测和未校正95%CI见X1_adjacent_descriptive.csv。主要三个差不能确认完整链的每个相邻小间隔，尤其不能把Adam(1)的描述性位置冒充主要显著性结论。\n\n'
for z in point_order:text+=f'- d={z["d"]}, r={z["r"]:g}：'+ ' > '.join(z['order'])+'。\n'
text+='\n## 同数据检索选参：次要分析\n\n'
text+='仅复用d256每r前16个正式seed，每数据集使用同一批前256查询选参并评价。方向固定，幅度乘数网格[0,2]、间隔1/16，首个经验跨越区间内二分10次，再取访问点的最小匹配误差，平手取较小幅度。新recall阶梯与非单调性使它不等同于真正首次根；未证明选择器一致性，定理7.2/推论7.4不直接覆盖该程序。以下区间是描述性的未校正95%配对seed t区间，不参与主要判定；同数据选择与评价的复用必须保留。\n\n'
text+='|r|配对差|n|旧recall差估计|描述性95%区间|\n|---|---|---|---|---|\n'
for z in secondary:text+=f'|{z["r"]:g}|{z["a"]}−{z["b"]}|{z["n"]}|{z["mean"]:.6f}|[{z["lo"]:.6f}, {z["hi"]:.6f}]|\n'
brackets=sum(not r['bracketed'] for r in sec);secmx=max(abs(r['new']-TARGET) for r in sec)
text+=f'\n次要分析{len(sec)}个方法×数据集记录；未括住目标{brackets}个；最大绝对新匹配误差{secmx:.6f}。所有访问点、最终幅度与误差在X1_secondary_rows.jsonl，不能替代主要确定性参数结果。\n\n'
text+='## 资源、审计与复现\n\n'
text+=f'正式主计算{fmeta["elapsed"]:.2f}秒；正式加次要{smeta["formal_plus_secondary_seconds"]:.2f}秒（{smeta["formal_plus_secondary_seconds"]/60:.2f}分钟），小于2小时。pilot耗时'+f'{json.loads((OUT/"X1_pilot_meta.json").read_text())["elapsed"]:.2f}秒，另计、不混入正式。正式峰值RSS{fmeta["peak_rss_mib"]:.1f}MiB，次要峰值{smeta["peak_rss_mib"]:.1f}MiB。每次计算均置于指定systemd scope，cgroup实测MemoryMax=3221225472、MemorySwapMax=0；加载的每份OpenBLAS实测≤2线程。未调用LLM API，未修改paper或其他已有文件。\n\n'
text+=f'正式完成{len(audits)}个独立数据集，逐一核验2048查询ID、全部方法/任务成功指示与rank、配对差和标量汇总完全一致；总体首次支参数另以独立求根核对，d16实现检查以稠密排名及样本梯度恒等式核对。冻结文件哈希均未改变；一份paper输入的外部措辞变化另有前后副本及哈希记录，不声称整个工作区静止。未把试算seed或同数据选择结果混入主族。\n\n'
text+='复现顺序为X1_run_v3.py的population→冻结→smoke→pilot→plan→锁定design→formal→secondary，再运行X1_analyze_v4.py（v4仅如实披露外部输入变化，统计逻辑与冻结的v3一致）；现有输出使用排他创建，复跑需在本项目内新的X1前缀输出位置保留原结果。命令前缀：\n\n```bash\nsystemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \\\n  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\\n  NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \\\n  /home/liuyibo/venvs/theory/bin/python experiments/x1/X1_run_v3.py MODE\n```\n\n'
text+='原始主结果：X1_formal_rows.jsonl与X1_formal_d*_r*_s*.npz；独立试算：X1_pilot_rows.jsonl、X1_pilot_d*_r*_s*.npz；总体参数：X1_population.json；完整结果表：X1_primary.csv、X1_new_matching.csv、X1_adjacent_descriptive.csv、X1_secondary_contrasts.csv；图均有PNG/PDF。主要限制是只覆盖共享键的三个r及两个有限维度，未给有限维偏差理论，不声称普遍优化器排序或多步结论。\n'
# Report lives in project root, so project-relative links start there.
text=text.replace('(../../X1_PREREG.md)','(X1_PREREG.md)').replace('(../../X1_DESIGN.md)','(X1_DESIGN.md)')
with (ROOT/'X1_RESULTS.md').open('x') as f:f.write(text)
pilotmeta=json.loads((OUT/'X1_pilot_meta.json').read_text())
pilottext='# X1 独立试算记录\n\n预注册冻结之后、正式规模锁定之前运行；仅方差、新recall匹配偏差和成本用于冻结规则。48个独立数据集，不进入正式结果。\n\n'
pilottext+=f'耗时{pilotmeta["elapsed"]:.2f}秒，峰值RSS{pilotmeta["peak_rss_mib"]:.1f}MiB。phase=11，与正式phase=22分离。逐seed全部旧/新recall见X1_pilot_rows.jsonl；逐查询rank和成功指示见对应npz；全部18项配对方差及保守上界见X1_design_v3.json及根目录X1_DESIGN.md。\n'
with (OUT/'X1_PILOT.md').open('x') as f:f.write(pilottext)
dump('X1_audit.json',dict(audited_datasets=len(audits),source_hashes_unchanged=source_check,
    completed_formal=fmeta['complete'],formal_plus_secondary_seconds=smeta['formal_plus_secondary_seconds'],
    direction_confirmed=direction,prediction_covered=cover,strong_support=strong,precision_met=precision,
    practical_matching=practical))
paths=sorted([p for p in OUT.rglob('*') if p.is_file()]+[p for p in ROOT.glob('X1_*') if p.is_file()])
with (ROOT/'X1_ARTIFACTS.sha256').open('x') as f:
    for p in paths:f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n')
print(json.dumps(dict(direction=direction,coverage=cover,strong=strong,precision=precision,matching=practical,formal_seconds=smeta['formal_plus_secondary_seconds'])),flush=True)
