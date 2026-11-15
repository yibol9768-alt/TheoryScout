"""不导入被审计规划代码：重建训练核，独立聚合停止协议和冻结判决。"""
import csv
from collections import OrderedDict
import json
import numpy as np
from common import ROOT, dump_file, install_guards, resource_info

install_guards()
base = ROOT/'experiments/real/analysis_length'
tasks = list(csv.DictReader((base/'ANALYSIS_LENGTH_C_tasks.csv').open()))
steps = list(csv.DictReader((base/'ANALYSIS_LENGTH_C_steps.csv').open()))
index = {(r['batch'], r['id']): i for i, r in enumerate(tasks)}
counts = np.zeros((len(tasks), 2, 5), dtype=np.int64)
strata = OrderedDict()
for i, row in enumerate(tasks):
    strata.setdefault((row['batch'], row['length']), []).append(i)
for row in steps:
    if row['resolved'] != '1':
        continue
    i = index[row['batch'], row['id']]
    j = int(int(row['step']) >= 16)
    counts[i, j] += [1, int(row['fatal']), int(row['error']), int(row['r2'] != ''), int(row['r2'] or 0)]

rng = np.random.default_rng(116990004)
sampled = np.zeros((10000, 2, 5), dtype=np.int64)
for ids in strata.values():
    draw = rng.integers(0, len(ids), (10000, len(ids)))
    sampled += counts[np.asarray(ids)[draw]].sum(axis=1)
assert counts.sum(axis=0).tolist() == [[1463,34,333,72,38],[1339,14,289,56,42]]

design = json.loads((ROOT/'HOLDOUT_D_DESIGN.json').read_text())
claimed = json.loads((ROOT/'HOLDOUT_D_AMEND1_RESULTS.json').read_text())['scenarios']['parameter_resampling']
rng = np.random.default_rng(216990101)
size = (10000, 144)
alive = np.ones(size, dtype=bool)
completed = np.zeros(size, dtype=int)
cost = np.zeros(10000, dtype=int)
future = np.zeros((10000, 144, 2, 5), dtype=np.int64)
for segment, width in enumerate((16,48)):
    n, d, e, m, s = [sampled[:, segment, k, None] for k in range(5)]
    fatal_step = rng.geometric(d/n, size=size)
    exposures = alive * np.minimum(width, fatal_step)
    deaths = alive & (fatal_step <= width)
    correct_advances = exposures-deaths
    repaired_errors = rng.binomial(correct_advances, (e-d)/(n-d))
    second_success = rng.binomial(repaired_errors, (m-d)/(e-d))
    errors = repaired_errors+deaths
    second_reviews = second_success+deaths
    future[:,:,segment] = np.stack((exposures,deaths,errors,second_reviews,second_success),axis=-1)
    cost += (exposures+errors+second_reviews).sum(axis=1)
    completed += correct_advances
    alive &= ~deaths

w = np.asarray(design['comparison']['weights_log_D_over_A'])
cells = np.stack(((completed<8).sum(1), ((completed>=8)&(completed<16)).sum(1),
                  ((completed>=16)&(completed<32)).sum(1), ((completed>=32)&(completed<64)).sum(1),
                  (completed==64).sum(1)), axis=1)
primary = cells@w > .541
calibration = np.ones(10000,dtype=bool)
for L, bounds in zip((8,16,32,64), ((107,130),(85,113),(70,97),(46,74))):
    k = (completed>=L).sum(1)
    calibration &= (bounds[0]<=k)&(k<=bounds[1])
budget = cost<=9000
totals = future.sum(1)
eligible = (totals[:,:,0]>0).all(1)&(totals[:,:,1]>=10).all(1)&(totals[:,:,3]>=30).all(1)
flags = dict(potential_primary=primary,potential_calibration=calibration,
             potential_primary_calibration=primary&calibration,budget_complete=budget,
             primary=primary&budget,calibration=calibration&budget,joint=primary&calibration&budget,
             potential_count_eligibility=eligible,count_eligibility=eligible&budget)
rates = {k:float(v.mean()) for k,v in flags.items()}
matches = {k:rates[k]==claimed['rates'][k] for k in rates}

# 真正索引重采样核对前三个未来实验；不用被审计代码的频数矩阵乘法。
bootstrap = []
for i in range(3):
    ids = np.random.default_rng(116990005).integers(144,size=(10000,144))
    aggregate = np.zeros((10000,2,5),dtype=np.int64)
    for start in range(0,10000,100):
        aggregate[start:start+100] = future[i,ids[start:start+100]].sum(1)
    valid = (aggregate[:,:,0]>0).all(1)&(aggregate[:,:,3]>0).all(1)
    valid_values=aggregate[valid]
    h=valid_values[:,:,1]/valid_values[:,:,0]
    r=valid_values[:,:,4]/valid_values[:,:,3]
    ci_h=np.quantile(h[:,0]-h[:,1],[.025,.975],method='linear')
    ci_r=np.quantile(r[:,1]-r[:,0],[.025,.975],method='linear')
    bootstrap.append(dict(i=i,valid=int(valid.sum()),h_ci=ci_h.tolist(),r2_ci=ci_r.tolist(),
                          mechanisms=bool(eligible[i] and valid.sum()>=9900 and ci_h[0]>0 and ci_r[0]>0)))

result=dict(resources=resource_info(),training_counts=counts.sum(0).tolist(),strata={str(k):len(v) for k,v in strata.items()},
            simulations=10000,parameter_seed=116990004,future_seed=216990101,rates=rates,exact_matches=matches,
            counts={k:int(v.sum()) for k,v in flags.items()},budget_exhaustions=int((cost>9000).sum()),
            demand_quantiles={str(q):float(np.quantile(cost,q)) for q in (.5,.95,.99,.999)},
            demand_max=int(cost.max()),bootstrap_three=bootstrap)
dump_file('independent_planning.json',result)
print(json.dumps(result,ensure_ascii=False))
