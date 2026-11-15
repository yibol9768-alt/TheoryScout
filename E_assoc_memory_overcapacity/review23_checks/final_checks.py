"""Finish review 23: reaggregate fresh data, independent budgets and integrity.

Only newly created review23 files may be edited by this script.
"""
import json
import math
import re
import hashlib
from pathlib import Path
from scipy.stats import beta, binom
import independent_review as own

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
fresh=json.loads((HERE/'fresh_results.json').read_text())
raw=[json.loads(x) for x in (HERE/'fresh_datasets.jsonl').read_text().splitlines()]
old=json.loads((ROOT/'ac2_checks/full/AC2_results.json').read_text())['rows']
comparisons=[]
for row in fresh['rows']:
    original=next(r for r in old if (r['method'],r['d'])==(row['method'],row['d']))
    records=[r for r in raw if (r['method'],r['d'])==(row['method'],row['d'])]
    assert len(records)==row['B']
    assert sorted(r['rep'] for r in records)==list(range(row['B']))
    for r in records:
        assert r['recall']==[k/r['n'] for k in r['counts']]
    comparisons.append(dict(method=row['method'],d=row['d'],mean_z=[
        (row['mean'][i]-original['mean'][i])/math.sqrt(row['SE'][i]**2+original['SE'][i]**2)
        for i in (0,1)],variance_ratio_new_over_old=[
        row['variance'][i]/original['dataset_variance'][i] for i in (0,1)]))
trends=[]
for method in own.METHODS:
    rr=sorted((r for r in fresh['rows'] if r['method']==method),key=lambda r:r['d'])
    for i,L in enumerate((1,5)):
        vv=[r['variance'][i] for r in rr]
        trends.append(dict(method=method,L=L,variances=vv,
                           strictly_decreasing=all(vv[k]>vv[k+1] for k in range(3)),
                           d64_over_d512=vv[0]/vv[-1]))

source_budget=json.loads((ROOT/'ac2_checks/AC2_root_budget.json').read_text())
budget=[]
for row in source_budget['rows']:
    n,d,L=row['n'],row['d'],row['L']; m=n-1
    neg=binom.cdf(L-1,m,.5)
    lever=beta.sf(.5,d/2,(n-d)/2)
    # A coordinate squared divided by m has Beta(1/2,(m-1)/2) law.
    # Uses survival probabilities rather than the reviewed betainc implementation.
    sphere=(m/L)*.5*beta.sf(.5,.5,(m-1)/2)
    value=min(1.,float(neg+lever+sphere))
    budget.append(dict(d=d,n=n,L=L,bound=value,
                       absolute_difference=abs(value-row['root_exception_union_bound'])))

before=json.loads((HERE/'inputs_before_sha256.json').read_text())
after=own.input_hashes()
changed=[p for p in before if before[p]!=after.get(p)]
assert not changed
updated=own.root_and_rate_checks()
prior=json.loads((HERE/'root_and_rates.json').read_text())
assert updated['rate_diagnostics']==prior['rate_diagnostics']
assert updated['root_cases'][2]['success_probability']==0.
assert updated['root_cases'][3]['success_probability']==0.
with (HERE/'root_and_rates_corrected.json').open('x') as f:
    json.dump(updated,f,indent=2,allow_nan=False)
summary=dict(fresh_datasets=len(raw),fresh_queries=sum(r['n'] for r in raw),
             fresh_all_one_rows=sum(r['counts']==[r['n'],r['n']] for r in raw),
             all_fresh_six_variances_decrease=all(t['strictly_decreasing'] for t in trends),
             fresh_trends=trends,comparisons=comparisons,
             maximum_absolute_combined_mean_z=max(abs(z) for c in comparisons for z in c['mean_z']),
             root_budget_cells=len(budget),root_budget_max=max(b['bound'] for b in budget),
             root_budget_max_abs_difference=max(b['absolute_difference'] for b in budget),
             checked_input_files=len(before),changed=changed,
             corrected_root_cases=updated['root_cases'],
             note='Fresh simulation completed before adding the explicit zero-U assertion; all 480 rows have recall below 1, so none has zero U. The initial whole-line integration used an infinite test point when no roots existed; the corrected finite point changes the H<0 example from 1 to 0. Both are covered by the reviewed [0,1] branch. Original output and failed validation log retained; rate diagnostics unchanged.')
with (HERE/'final_checks.json').open('x') as f: json.dump(summary,f,indent=2,allow_nan=False)

path=ROOT/'PROOF_REVIEW23.md'
text=path.read_text()
placeholder='独立结果与比较详见 [fresh_results.json](review23_checks/fresh_results.json) 及 [final_checks.json](review23_checks/final_checks.json)。本节最终统计和资源记录在计算结束后补齐。'
assert placeholder in text
lines=['独立新样本共直接排名 **1,541,600 个训练查询**；六条新方差序列也全部下降。表内方差均以完整数据集为单位、分母为 B−1。','',
       '| 方法 | d | B | 独立 top-1 / top-5 | 独立 Var(top-1) / Var(top-5) |',
       '|---|---:|---:|---|---|']
for row in fresh['rows']:
    lines.append(f"| {row['method']} | {row['d']} | {row['B']} | {row['mean'][0]:.6f} / {row['mean'][1]:.6f} | {row['variance'][0]:.6g} / {row['variance'][1]:.6g} |")
lines.extend(['',f"24 项新旧均值比较的最大绝对合并 SE 标准化差为 {summary['maximum_absolute_combined_mean_z']:.3f}。这只是描述性比较，没有用它宣布逐项等效。方差估计有明显抽样波动，尤其新 d=512 每格仅 16 份数据；新旧方差不需数值相等。完整数据见 [fresh_results.json](review23_checks/fresh_results.json)、[fresh_datasets.jsonl](review23_checks/fresh_datasets.jsonl)、[比较及最终核验](review23_checks/final_checks.json)。",'',
       f"独立重算全部 72 个 F5 预算，最大值 {summary['root_budget_max']:.8g}，与报告最大绝对差 {summary['root_budget_max_abs_difference']:.3g}；此处仍是浮点复算，不是机器舍入证书。"])
info=json.loads((HERE/'resources.json').read_text())
lines.extend(['',f"主核验墙钟 {info['wall_seconds']:.2f} 秒，CPU {info['cpu_seconds']:.2f} 秒，峰值 RSS {info['peak_RSS_KiB']} KiB；实测 Threads=1，memory.max={info['memory.max']}、memory.swap.max={info['memory.swap.max']}。核对 {len(before)} 个输入/被审代码/paper 文件哈希，变化列表为 []。"])
text=text.replace(placeholder,'\n'.join(lines))
text=text.replace('(review23_checks/root_and_rates.json)',
                  '(review23_checks/root_and_rates_corrected.json)')

# Normalize inline math delimiters in this new report, leaving references,
# Markdown link targets and display equations intact.
text=re.sub(r'\\{1,2}\((.*?)\\{1,2}\)',lambda m:'$'+m.group(1)+'$',text)
def normalize(line):
    out=[]; i=0
    while i<len(line):
        if line[i] in ('$', '`'):
            end=line.find(line[i],i+1)
            if end!=-1:
                out.append(line[i:end+1]); i=end+1; continue
        if line[i]!='(' or (i and line[i-1]==']'):
            out.append(line[i]); i+=1; continue
        level=1; j=i+1
        while j<len(line) and level:
            level+=(line[j]=='(')-(line[j]==')'); j+=1
        if level:
            out.append(line[i:]); break
        inside=line[i+1:j-1]
        mathy=bool(re.search(r'[\\_^=<>]',inside) or re.fullmatch(r'[A-Za-z](?:,[A-Za-z])*',inside))
        out.append('$'+inside+'$' if mathy else line[i:j]); i=j
    return ''.join(out)
new=[]; display=False; code=False
for line in text.splitlines():
    if line.startswith('```'): code=not code
    if line.strip()=='\\[': display=True
    new.append(line if display or code else normalize(line))
    if line.strip()=='\\]': display=False
text='\n'.join(new)+'\n'
# Literal underscores in table math are fine; vertical bars require escapes.
path.write_text(text)
print(json.dumps(summary,ensure_ascii=False))
