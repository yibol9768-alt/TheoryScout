"""Render existing deterministic outputs, exclusive creation only."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    assert os.environ.get(k)=='2'
import csv
from pathlib import Path
P=Path(__file__).resolve().parent
def read(n):
    with (P/n).open() as f:return list(csv.DictReader(f))
c=read('Y3_coefficients.csv');nn=read('Y3_X1_new_comparison.csv');cc=read('Y3_X1_contrasts.csv')
out=['# Y3 全数值表（候选预测，非确认）','',
     '比例均乘 100 后为百分点。B 是 recall 比例的 1/d 系数。数值系数来自确定性积分，没有使用 X1 观测拟合。','',
     '|r|方法|B_old|B_new|新匹配偏差预测 d512（百分点）|',
     '|---|---|---|---|---|']
for old in c:
    if old['task']!='old':continue
    new=next(a for a in c if a['ir']==old['ir'] and a['method']==old['method'] and a['task']=='new')
    out.append(f"|{old['r']}|{old['method']}|{float(old['B_recall']):.8f}|{float(new['B_recall']):.8f}|{100*float(new['B_recall'])/512:+.5f}|")
out+=['','## X1 新匹配偏差：观测 / 候选修正（百分点）','',
      '|r|方法|d128 观测|d128 预测|d256 观测|d256 预测|','|---|---|---|---|---|---|']
for old in nn:
    if old['d']!='128':continue
    new=next(a for a in nn if a['r']==old['r'] and a['method']==old['method'] and a['d']=='256')
    vals=[float(old['observed_bias']),float(old['proposed_bias']),float(new['observed_bias']),float(new['proposed_bias'])]
    out.append('|'+old['r']+'|'+old['method']+'|'+ '|'.join(f'{100*x:+.5f}' for x in vals)+'|')
out+=['','## X1 旧任务配对差（百分点）','',
      '|d|r|差|极限|候选修正|观测|观测−候选|','|---|---|---|---|---|---|---|']
for r in cc:
    vals=[float(r[k]) for k in ['population','proposed','observed','residual']]
    out.append('|'+r['d']+'|'+r['r']+'|'+r['a']+'−'+r['b']+'|'+ '|'.join(f'{100*x:.5f}' for x in vals)+'|')
out+=['','## d512 冻结的旧任务配对差（百分点）','',
      '|r|差|极限|候选修正|','|---|---|---|---|']
for r in cc:
    if r['d']!='128':continue
    pop=float(r['population']);pred=pop+float(r['B_difference'])/512
    out.append(f"|{r['r']}|{r['a']}−{r['b']}|{100*pop:.5f}|{100*pred:.5f}|")
with (P/'Y3_tables.md').open('x') as f:f.write('\n'.join(out)+'\n')
print('\n'.join(out))
