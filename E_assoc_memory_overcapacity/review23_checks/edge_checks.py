"""Independent deterministic counterexample and scope/centering checks."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import norm

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
d,n=3,17
x=np.zeros((d,n))
x[:,:d]=math.sqrt(n/d)*np.eye(d)
z=np.array([[.15,-.24,.31],[-.19,.28,-.34],[.22,.18,-.26]])
y=np.zeros((d,n))
y[:,:d]=z/math.sqrt(d)
s=(d/n)*(x@x.T)
actual_z=(d/math.sqrt(n))*(y@x.T)
u=np.zeros((d,d))
states=[]
for unused in range(2):
    u=u-.6*np.sign(u@s-actual_z)
    states.append(u.tolist())
assert np.array_equal(u,np.zeros_like(u))
score=y.T@u@x
strict_counts=(score>score.diagonal()[None,:]).sum(axis=0)
assert not strict_counts.any()
bad_recall=[float(np.mean(strict_counts<L)) for L in (1,5)]
uniform_tie=[L/n for L in (1,5)]
raw=[json.loads(line) for line in (ROOT/'ac2_checks/full/AC2_datasets.jsonl').read_text().splitlines()]
fresh=[json.loads(line) for line in (HERE/'fresh_datasets.jsonl').read_text().splitlines()]

# The open set ||S-I||_op<1/(12 sqrt(d)), 0.1<|Z_ab|<0.4
# also yields Sign2=0: ||.6 sign(Z)(S-I)||_max<.05.
# Gaussian data have full support, and this construction has S=I, so its
# inverse image contains a nonempty open neighborhood of (X,Y).
centering=[]
for k in (16.,64.,256.):
    a=1.; relative=k**(-.25)
    log_d=math.exp(k)
    log_n=2*log_d-math.log(2*a*k)+math.log1p(relative)
    tail=math.exp(-a*math.log(log_n))
    t=norm.isf(tail)
    mu=math.sqrt(2*a*k/(1+relative))
    centering.append(dict(log_log_d=k,relative_load_perturbation=relative,
                          mu0=mu,threshold=float(t),window_center=float(mu-t)))
output=dict(
    zero_Sign2=dict(d=d,n=n,S=s.tolist(),Z=actual_z.tolist(),states=states,
        strict_exceedance_recall=bad_recall,uniform_tie_expected_recall=uniform_tie,
        open_set_sufficient_op_radius=1/(12*math.sqrt(d))),
    published_all_one_rows=sum(r['counts']==[r['n'],r['n']] for r in raw),
    fresh_all_one_rows=sum(r['counts']==[r['n'],r['n']] for r in fresh),
    fresh_rows_so_far=len(fresh),
    leading_load_equivalence_alone_is_not_window_centering=centering,
    negative_control=dict(C='I',mu=0,rank_one_U='e1 e1^T',
        correct_score_limit='G1*G2',variance=1,fourth_moment=9,
        spread_U='I/sqrt(d)',spread_limit='N(0,1)',spread_fourth_moment=3,
        rank_one_S1_fails=True))
with (HERE/'edge_results.json').open('x') as f:
    json.dump(output,f,indent=2,allow_nan=False)
print(json.dumps(output,ensure_ascii=False))
