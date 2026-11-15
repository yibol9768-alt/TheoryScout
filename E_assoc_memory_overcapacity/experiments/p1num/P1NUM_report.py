"""Report formatting from fixed gates only. No new data, fitting, or model choice."""
import csv,json,hashlib,math,time
from pathlib import Path
from P1NUM_run import ROOT,OUT,put,metadata,frozen

def read(name):return list(csv.DictReader((OUT/name).open()))
def yes(x):return x is True or x=='True'
def pp(x):return f'{100*float(x):.4f}'
def fmt(x):return '未夹定' if x is None or x=='' else f'{float(x):.6f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in r)+' |' for r in rows])
def main():
 st=time.monotonic();meta=metadata();frozen();s=json.loads((OUT/'P1NUM_summary.json').read_text());audit=json.loads((OUT/'P1NUM_independent_audit.json').read_text());design=json.loads((OUT/'P1NUM_design.json').read_text())
 A=read('P1NUM_A_gates.csv');M=s['M'];bounds=s['boundaries'];contr=read('P1NUM_epsilon_contrast.csv');old=read('P1NUM_P3_old240_sensitivity.csv');p2=[r for r in M if r['kind']=='P2'];p3=[r for r in M if r['kind']=='P3']
 metas={key:json.loads((OUT/name).read_text()) for key,name in [('冒烟','P1NUM_smoke_metadata.json'),('正式','P1NUM_formal_metadata.json'),('分析','P1NUM_summary.json'),('独立重放','P1NUM_independent_audit.json')]}
 if (OUT/'P1NUM_plot_metadata.json').exists():metas['绘图']=json.loads((OUT/'P1NUM_plot_metadata.json').read_text())
 ncomplete=sum(r['complete'] for r in M);p2pass=sum(r.get('combined_pass',False) for r in p2);p3pass=sum(r.get('noise_compatible',False) for r in p3);olds=sum(yes(r['old240_noise_compatible']) for r in old)
 bline='；'.join(f"γ{g}："+'、'.join(f"d{r['d']} [{fmt(r['last_confirmed_below'])}, {fmt(r['first_confirmed_above'])}]" for r in bounds if r['gamma']==g) for g in (16,64,256))
 text=f'''# P1NUM 结果：有限 γ stabilizer 边界与 M 补精度

**结论：固定网格得到 {sum(r['bracket_valid'] for r in bounds)}/6 个有限维边界夹区间；预测中心的1pp等效门槛通过 {s['A_center_pass']}/{s['A_gates']}。P2 两项中 {p2pass}/2 达到方向确认及1pp等效的联合门槛；P3 六个代表中 {p3pass}/6 达到查询噪声相容0.5pp门槛。本轮完成 {s['completed_configurations']}/{s['planned_configurations']} 配置，M 的8项覆盖完整 {ncomplete}/8。任何未完成或未分辨项均不算通过。**

边界夹区间为：{bline}。这些是有限 d 的固定网格端点，不是精确根、单数据集根或唯一性证明。大 γ 极限 e* 没有作为有限 γ 真值拟合。实验不补足 EXT L2，不改变 M 的已有证明和 EXP1/EXP2/EXT 的历史判决。

## 1. 冻结、覆盖与资源：G0

预注册 SHA-256：`{(ROOT/'P1NUM_PREREG.sha256').read_text().split()[0]}`。注册、原冻结代码、旧数据和结果输入逐项散列复核一致；旧产物没有写入或重新生成。最初只用旧数据的 M2048 规划及最终预算受限设计均保留；任何新随机数据之前已经确定最终网格、N和门槛。冒烟不进入统计。

原分析脚本对不完整条件的状态显示有记账疏漏，运行期间仅依据代码发现、未检视新效果时新增 V2，按预注册强制显示“未完成”；原冻结文件不改，统计公式和门槛不变。见 [更正说明](../../P1NUM_ANALYSIS_NOTE.md)。本轮统计种子域910631、正式tag11，区别旧EXP1/2/EXT及本轮冒烟tag91；不把查询/候选算作重复。

冒烟3/3配置通过：A稠密得分与trace/对角、CE均值因子、M流重放和精确全候选排名、原始W单位Adam与代理缩放均一致；另有一个d512正式规模资源冒烟。正式写入 {s['records']} 条记录（计划2822条），完成 {s['completed_configurations']}/530 配置。正式成功标记为 `{metas['正式']['success']}`；错误/预算事件为 `{metas['正式']['error']}`。

'''
 resource_rows=[]
 for label,r in metas.items():
  end=r['resource_end'];sec=r.get('seconds',None)
  resource_rows.append([label,fmt(sec) if sec is not None else '见metadata',f"{end['rss_MiB']:.2f}",f"{int(end['controls']['memory.peak'])/2**20:.2f}"])
 text+=table(['阶段','脚本墙钟/秒','进程峰值RSS/MiB','scope memory.peak/MiB'],resource_rows)+'\n\n'
 maxrss=max(r['resource_end']['rss_MiB'] for r in metas.values())
 text+=f'''正式计算 {metas['正式']['seconds']:.2f} 秒，小于一小时；冒烟 {metas['冒烟']['seconds']:.2f} 秒。已记录阶段峰值RSS为 {maxrss:.2f} MiB；RSS与cgroup计数分列。每阶段均为CPU、指定Python、两份BLAS各2线程、2GiB硬内存、零swap；OOM计数见各metadata。旧数据规划没有单独留峰值，不能把上表声称为覆盖规划峰值。所有项目文件读写仅在允许目录；只读取必要系统运行/依赖元数据，无网络、LLM/API或GPU调用。

独立重放仅使用本轮已有正式种子：d128/γ16/seed0 的 {audit['dense_A_methods']} 个方法以完整2048×2048得分矩阵重算，最大margin/variance误差 {audit['max_dense_A_error']:.3g}；d256/κ.5/seed0 用稠密X/Y及原始W单位递推验证 {audit['rank_checks_total']} 个原始排名。全记录查询不重复、排名合法、D/v/A由原始排名重算、固定任务顺序及种子覆盖集合均审计；缺 {audit['missing_records']} 条计划记录。详见 `P1NUM_independent_audit.json`。

## 2. A1：种子级精度与事前功效

A的统计单位为数据种子。各γ取旧EXT两个d/四e中最大的种子配对标准差再乘2，方差放大4倍；目标为78格同时误差.005，正态规划80%，γ16/64/256分别29/30/25种子，每个d相同。实际种子t同时半宽≤.005的格数 {s['A_precision_pass']}/78。半宽事件本身没有被宣称具有严格80%功效；新网格方差可能不被旧网格最大值覆盖。

## 3. A2/A3：有限 γ 中心、包络和直接边界

每种子先计算 R=rho_e/rho_M−1，再平均。rho为全训练集平均margin除以平均错误variance平方根；没有替换为平均逐查询rho、独立候选或平均rho的比值。原始h矩阵未作q归一化，因MSR正缩放不变；原始variance字段不能直接当作归一化Tγ variance比较。

冻结中心 R0=Fγ(c)/[b(1−.1/γ)]−1；全部78格的 |Rbar−R0|+同时t半宽≤.01 者 {s['A_center_pass']}/78，最大等效界 {max(float(r['center_equivalence_bound']) for r in A):.6f}。中心不拟合任何常数，也不检验余项阶。

'''
 text+=table(['γ','d','最后确认below','最先确认above','宽度','冻结中心根','中心在夹区间内'],[[r['gamma'],r['d'],fmt(r['last_confirmed_below']),fmt(r['first_confirmed_above']),fmt(r['bracket_width']),fmt(r['center_root']),r['center_inside']] for r in bounds])+'\n\n'
 text+=f'''端点使用78格同时95%种子t区间严格离0；不作插值，不在观察crossing后加点。包络有充分方向的格 {s['A_envelope_called']} 个，其中 {s['A_envelope_confirmed']} 个方向获同时区间确认；其余包络未分辨格没有计入“预测通过”。Muon标准化MSR的六格区间落入[L−.005,U+.005]者 {s['muon_envelope_pass']}/6，.005是实验有限d容差，不能加入理论余项。

理论根包络仍为γ16 [0,∞)、γ64约[.009754,.229380]、γ256约[.044654,.064106]，条件于Tγ及原外部Muon输入。实验夹区间远窄于保守包络也不使该包络自动成为精确根定理。冻结中心与有限维区间的位置区别完整披露；不把极限e*≈.05015613当成各γ真值。

'''
 failures=[r for r in A if not yes(r['precision_pass']) or not yes(r['center_pass']) or (r['envelope_direction']!='unresolved' and not yes(r['envelope_direction_confirmed']))]
 text+='A未过格：'+('无。' if not failures else '见下表；失败不删去。')+'\n\n'
 if failures:text+=table(['d','γ','e/路径','精度通过','中心通过','包络方向','方向确认'],[[r['d'],r['gamma'],r['label'],r['precision_pass'],r['center_pass'],r['envelope_direction'],r['envelope_direction_confirmed']] for r in failures])+'\n\n'
 text+='完整78格均值/SE/区间/门槛见 `P1NUM_A_gates.csv`，Muon见 `P1NUM_muon_gates.csv`，边界及未分辨网格点见 `P1NUM_boundary.csv`。\n\n## 4. A4：固定物理 ε 与固定 e 对照\n\n'
 text+=f"固定 ε={design['design']['epsilon_physical']:.12g}，固定e对照为.06。两路径每种子配对；六格Bonferroni双侧t区间。与冻结预测差的1pp等效通过 {s['epsilon_contrast_pass']}/6。表中差为 R_phys−R_fixed，单位个百分点。\n\n"
 text+=table(['d','γ','物理路径e','配对差/pp','同时区间/pp','1pp等效'],[[r['d'],r['gamma'],fmt(r['physical_e']),pp(r['ratio_contrast']),f"[{pp(r['lower'])}, {pp(r['upper'])}]",r['pass_1pp']] for r in contr])+'\n\n'
 text+='锚点d128/γ64两法恒等，不计独立发现。其他格区间离0才确认路径不同。该人为可观测ε用于尺度对照，不是1e−8实用规模或语言模型结论。固定ε时e随√n增长，不能冒充固定e极限。\n\n## 5. P2：两个 d512 未分辨项\n\n'
 text+='保留旧四context校正：单侧.05/4确认两步减一步为负；相对M极限差的同时双侧区间全在±.01内才等效。均值、SE及等效界下表均以百分点给出，等效界必须≤1。\n\n'
 text+=table(['κ/指标','N/计划N','两步减一步±SE/pp','极限/pp','等效界/pp','方向确认','联合结论'],[[f"{r['kappa']}/{r['metric']}",f"{r['N']}/{r['planned_N']}",f"{pp(r['mean'])} ± {pp(r['SE'])}",pp(r['target']),pp(r['equivalence_bound']),r['direction_confirmed'],r['status']] for r in p2])+'\n\n'
 text+='功效使用旧同维24种子方差×2；假设相对极限偏差.003时80%需121/322种子。资源上限事前固定64/112，计划功效只有43.13%/16.10%，零偏差时65.21%/22.85%；不能因结果不显著倒推噪声。实际SE与MDE在 `P1NUM_M_gates.csv`。任何不足计划N的区间仅描述，不作注册通过判定；不合并旧种子或延长预算补跑。\n\n## 6. P3：六个代表、查询噪声与有限维残留\n\n'
 text+='选择规则在新数据前固定：原76项中37个未分辨项，按每κ×Adam/Sign家族取(name,metric)字典序首项，只补d256；每格512无放回查询和全部候选。六个新N=55/76/39/32/21/71。规划方差为2×max(旧D方差,旧条件查询方差均值)，在真实有符号均值0时±.005等效规划功效80.16%–83.11%；不承诺去噪A门槛的功效。\n\n'
 text+='下表D是真实减代理；均值/等效界为百分点，A为原始比例平方。通过须等效界≤0.5且A单侧上界≤2.5e−5。保留负A，不以逐维严格单调作本轮唯一判据。\n\n'
 prows=[]
 for r in p3:
  oldr=next(q for q in old if float(q['kappa'])==r['kappa'] and q['name']==r['name'] and q['metric']==r['metric'])
  prows.append([r['kappa'],'Adam16' if r['name'].startswith('Adam') else 'Sign2 α=.2',r['metric'],f"{r['N']}/{r['planned_N']}",f"{pp(r['mean'])} ± {pp(r['SE'])}",pp(r['equivalence_bound']),f"{r['A']:.3g}",f"{r['A_upper']:.3g}",r['status'],oldr['old240_noise_compatible']])
 text+=table(['κ','方法','指标','N','D±SE/pp','等效界/pp','A','A上界','新六项家族','原240校正相容'],prows)+'\n\n'
 text+=f'''新六项校正相容 {p3pass}/6；使用相同新数据、沿用原240项校正的敏感性分析相容 {olds}/6。两者差别是同时置信家族不同，不能将全部变化归因于增种子。显著小残留也可能同时位于0.5pp精度界内；A下界和D区间仍完整保留。A的显著正值只说明查询噪声以外存在完整数据集差，可能包含数据集随机环境，不能自动称为确定性的b/d或b/√d修正。种子t为有限样本近似。

全部37项的选择/未选择及理由在 `P1NUM_old37_disclosure.csv`；未选31项本轮继续未分辨，不能外推六代表为37项全部解决。没有重复已有P1/P4来增加通过数，没有重拟合旧b或从极限e*拟合边界。

## 7. 解释边界与产物

有限γ定位仍是两个有限d、三个固定γ的实验；Tγ、sign去平滑和Muon外部输入的数学状态沿用冻结文件，不因本轮门槛通过升级为定理。M的辅助矩阵/得分误差上界不自动给召回有符号首项，P2/P3结果也不能鉴定曲线修正速率。没有top1极值、增长T、长尾、实际多步微调或实用ε的外推。

结果与全部原始记录在 `experiments/p1num/`。建议阅读顺序：本报告→`P1NUM_M_gates.csv` / `P1NUM_A_gates.csv`→`P1NUM_boundary.csv` / `P1NUM_epsilon_contrast.csv`→`P1NUM_old37_disclosure.csv`→审计与metadata。图为 `P1NUM_boundary.png/.pdf` 和 `P1NUM_M_intervals.png/.pdf`（若绘图依赖可用）；连续线仅确定性积分预测。所有入口 exclusive-create；禁止覆盖重跑。新增脚本只处理本轮数据，原冻结实现和旧实验文件完整保留。
'''
 with (ROOT/'P1NUM_RESULTS.md').open('x') as f:f.write(text)
 put(OUT/'P1NUM_report_metadata.json',dict(seconds=time.monotonic()-st,resource_start=meta,resource_end=metadata()))
 print(json.dumps(dict(report='P1NUM_RESULTS.md',P2_pass=p2pass,P3_pass=p3pass,completed=s['completed_configurations'],planned=530)))
if __name__=='__main__':main()
