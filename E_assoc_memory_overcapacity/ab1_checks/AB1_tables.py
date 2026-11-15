"""Create human-readable directed envelopes and machine-readable Q formula table."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal as D, Context, ROUND_FLOOR, ROUND_CEILING
import json
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'ab1_checks'
LOW=Context(prec=8,rounding=ROUND_FLOOR);HIGH=Context(prec=8,rounding=ROUND_CEILING)
def fmt(x,up=False):
    f=F(x);ctx=HIGH if up else LOW
    return str(ctx.divide(D(f.numerator),D(f.denominator)))
def box(v):return '['+fmt(v[0])+', '+fmt(v[1],True)+']'
rows=json.loads((BASE/'AB1_m8_directed_envelopes.json').read_text());ends=json.loads((BASE/'AB1_m8_endpoint_envelopes.json').read_text())
out=['# AB1：M.8 实际有向区间包络','','此表的十进制下端向下、上端向上放宽到8位有效数；判定使用 JSON 内精确有理数。',
     'J=Jβ/(a p_j²)，H=(∂βc₀|c₁=u)/(a³p_j²)=(15/16)J/(√q Λ)，实际导数 D=a³p_j²H。',
     '每格均为闭区间。D 的包围在接触 a=0 的格上必含0；严格符号指格内所有 a>0，a=0 时 D=0。',
     '区域表来自已独立重积分的 Decimal 包围；整数原证书的独立包围另存 JSON。',
     '默认 verify_m8.py 再次从公式积分全部237格，并不把这张表当作积分输入。','',
     '| 路径 | e | 改变的参数 | a范围 | 固定另一参数 | 正幅度内导数 |',
     '|---|---|---|---|---|---|',
     '| P1 | .5 | β1∈[0,.9] | [0,.29] | β2=.99 | 负 |',
     '| P2 | .5 | β2∈[0,.99] | [0,.29] | β1=.9 | 正 |',
     '| P3 | 2 | β1∈[0,.9] | [0,.53] | β2=.99 | 负 |',
     '| P4 | 2 | β2∈[0,.99] | [0,.53] | β1=.9 | 负 |','',
     '## 四条路径的联合包络','',
     '| 路径 | J有向包围 | H有向包围 | Λ下界 | 右端c1下界 |',
     '|---|---|---|---|---|']
for p in range(1,5):
    rr=[r for r in rows if r['path']==p];ee=[r for r in ends if r['path']==p]
    hull=lambda key:[str(min(F(r[key][0]) for r in rr)),str(max(F(r[key][1]) for r in rr))]
    out.append(f"| P{p} | {box(hull('Jnormalized'))} | {box(hull('matched_derivative_over_a3p2'))} | {fmt(min(F(r['lambda_bound'][0]) for r in rr))} | {fmt(min(F(r['decimal_c1'][0]) for r in ee))} |")
out.extend(['','## 全部173个区域单元','','| 单元 | a | βj | J区间 | H区间 | 实际D区间 | 判定(a>0) |','|---|---|---|---|---|---|---|'])
for r in rows:out.append('| '+' | '.join([r['id'],'['+', '.join(r['a'])+']','['+', '.join(r['beta'])+']',box(r['Jnormalized']),box(r['matched_derivative_over_a3p2']),box(r['matched_derivative']),'PASS '+('正' if r['sign']>0 else '负')])+ ' |')
out.extend(['','## 全部64个右端记录','','| 单元 | βj | c1(amax)有向区间 | >.55 |','|---|---|---|---|'])
for r in ends:out.append('| '+' | '.join([r['id'],'['+', '.join(r['beta'])+']',box(r['decimal_c1']),'PASS'])+' |')
with (ROOT/'AB1_M8_DIRECTED_ENVELOPES.md').open('x') as f:f.write('\n'.join(out)+'\n')
formulas={
 'schema':'AB1-Q-constant-formulas-v1','conventions':{'n':'ceil(kappa*d^2)','xi':'d^(-r)','eta0':'d^(-r)','r':'GD/Adam:1/2; Sign:1/6','z':'Phi^-1(1-rho)','q':'E h_T(G)^2 > 0'},
 'algorithm':{'GD':{'k':'1-(1-eta)^T','K':'k','v_a':'2*k^2','v_q':'2*k^4','C_g':'eta^2 sum_{j=1}^{T-1} j*(1+3eta/2)^(j-1)','f':'2*(2*C_g+k)'},
              'Sign':{'K':'T*alpha','v_a':'K^2','v_q':'K^4','A_s':'2*T^2/sqrt(2*pi)','B_s':'sum_{t=0}^{T-1}(t*alpha+2)^2','f':'2*T*alpha*sqrt(A_s*(2*B_s/A_s)^(1/3)+B_s*(A_s/(2*B_s))^(2/3))'},
              'Adam':{'lambda':'1+2alpha/e','c0':'sqrt(1-beta2)','H_t':'e*(lambda^t-1)/(2*c0)','J_t':'lambda^(t-1)/c0','L_t':'(1+J_t)/e','A_0':'0','A_t':'A_(t-1)+alpha*L_t*sqrt(sum_{j=1}^t(2*A_(j-1)+H_(j-1)+2)^2)','K':'H_T','v_a':'K^2','v_q':'K^4','f':'A_T'}},
 'common':{'b_star':'sqrt(q/2)','Z_star':'z+1','M':'3/sqrt(kappa)','D4':'8*M^4+480','s_rho':'min(rho/2,(1/2-rho)/2)','D_rho':'1/phi(Phi^-1(1-rho+s_rho))','H_rho':'5sqrt(2)*D_rho+4sqrt(2)*z','r_star':'min(1/(64Z_star),1/(4sqrt(52Z_star)))','K_R':'12*(1+Z_star)+3sqrt(52Z_star)','K_n':'12*(1+Z_star)+3sqrt(2Z_star*(M^2+2))','Abar':'2log(9)+log(2)+r','Dbar':'4sqrt(Abar/kappa)+4Abar/kappa','Vbar':'2K sqrt(2Abar)','Qbar':'log(4)+r','Fbar':'f*Dbar^(2r)','Nbase':'max(128,(32Z_star)^2,8Z_star*(M^2+2),8sqrt(Z_star*D4))'},
 'd_star_max_terms':['2','3/kappa','(8Dbar)^2','(4Vbar/(b_star*r_star))^2','(2sqrt(v_q)/q)^(1/(1-r/2))','sqrt(Nbase/kappa)','25Qbar/(kappa*s_rho^2)','16Qbar/kappa','Qbar*H_rho^2/(kappa*min(z/2,1)^2)'],
 'C_star_sum_terms':['5','2/9','4Fbar/(b_star*r_star)','4Fbar/(b_star*sqrt(kappa))','2Dbar/sqrt(kappa)','sqrt(v_a)/(b_star*sqrt(kappa))','sqrt(v_q)/(b_star^2*sqrt(kappa))','12*(1+Z_star)*Dbar','(2K_R/b_star)*(Vbar+Fbar)','K_n/sqrt(kappa)','12sqrt(Z_star*D4)/kappa','H_rho*sqrt(Qbar/kappa)'],
 'C_star_fixed':'C_star+1/(2*kappa^(3/2))','valid_error':'min(1,C_star*d^(-r)) for integer d>=d_star',
 'failure_probabilities':{'scatter':'xi','G_frobenius':'2/(9d^2)','a_moment':'xi','q_moment':'xi','H_operator':'xi','trajectory_cutoff':'4*Lcal/(b*r_star)','sphere_order_threshold':'eta0'}}
with (ROOT/'AB1_Q_CONSTANT_TABLE.json').open('x') as f:json.dump(formulas,f,ensure_ascii=False,indent=2)
print('TABLES PASS',len(rows),len(ends))
