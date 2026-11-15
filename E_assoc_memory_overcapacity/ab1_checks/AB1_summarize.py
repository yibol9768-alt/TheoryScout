"""Create concise Q numerical receipts from outward result tables."""
import json
from pathlib import Path
from decimal import Decimal as D, Context, ROUND_CEILING
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'ab1_checks';ctx=Context(prec=9,rounding=ROUND_CEILING)
lines=['# AB1：Q 的可复核常数例值','','固定 T=2、κ=.25、ρ=.1；GD η=.5，Sign α=.6，Adam α=.6、e=1、β=(.9,.99)。',
       '下表 C_* 向上放宽；q 显示严格区间；维度由48位区间上端上整得到，不代表实际最小门槛。','',
       '| 方法 | q严格包围 | C_*上界 | 充分适用维度 | C_*d^(-r)<1 的充分维度 | 计算秒数 |',
       '|---|---|---|---|---|---|']
for method in ['gd','sign','adam']:
    suffix='_v2' if method=='adam' else ''
    data=json.loads((B/f'AB1_q_{method}_example{suffix}.json').read_text());c=data['constants']
    cup=ctx.create_decimal(c['C_star'][1]);q=c['q_integral_enclosure']
    lines.append('| '+' | '.join([method,'['+', '.join(q)+']',str(cup),str(data['sufficient_dimension']),str(data['sufficient_dimension_for_error_below_one']),f"{data['seconds']:.3f}"])+ ' |')
a=json.loads((B/'AB1_q_audit.json').read_text());lines+=['','独立90位 mpmath 诊断复算了真实 q，并在适用门槛和非平凡门槛各检查一次全部有限维条件和完整预算；三种方法均通过。该诊断不替代区间常数认证。','',
'| 方法 | 真实q诊断值 | 非平凡门槛处完整有限维预算(诊断) |','|---|---|---|']
for row in a['methods']:lines.append('| '+' | '.join([row['method'],row['q_independent'],row['budgets'][1]['finite_budget']])+' |')
lines+=['','这是纯公式计算，没有构造上述维度的训练数据。`d_*` 只保证可以使用非平凡公式，不保证该公式小于1；程序另给出保守的非平凡维度。',
        'Sign 的维度极大，反映最坏情形常数和 d^(-1/6) 的保守性。这里没有为两步 FT 的微小历史效应提供有限维保证。',
        '有向常数证据见 `ab1_checks/AB1_q_{gd,sign}_example.json`、`AB1_q_adam_example_v2.json`；诊断见 `AB1_q_audit.json`。']
with (ROOT/'AB1_Q_RESULTS.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print('\n'.join(lines))
