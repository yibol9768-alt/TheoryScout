"""仅离线规划/状态展开；不创建 Generator、不抽训练数组或查询。"""
import ast
import csv
import itertools
import json
import math
from pathlib import Path
import resource
import shutil
import time
import numpy as np
from scipy.stats import nct, t
from common import ROOT, OUT, METHODS, PAIRS, DIMS, RS, dump, table, sha, environment, entropy, jobs, usage

resource.setrlimit(resource.RLIMIT_CPU, (240, 250))
env = environment()
start = usage()
inputs = {}
def read(path):
    p = ROOT / path
    inputs[path] = sha(p)
    return p

# 全部既有非日志文件的基线；不跟随链接、不修改。
baseline = {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and not p.is_symlink() and OUT not in p.parents
            and not p.name.startswith('Y3B_') and p.suffix != '.log'}
dump(OUT / 'existing_baseline.json', baseline)
pred_path = read('y3_checks/Y3_frozen_predictions.csv')
with pred_path.open() as f:
    pred = [{k: (v if k in ['method', 'task'] or v == '' else float(v)) for k, v in r.items()}
            for r in csv.DictReader(f)]
assert len(pred) == 90
with (OUT / 'predictions.csv').open('xb') as f:
    f.write(pred_path.read_bytes())
params = []
for ir, m in itertools.product(range(3), METHODS):
    p = next(r for r in pred if r['ir'] == ir and r['method'] == m)
    params.append({k: p[k] for k in ['ir', 'r', 'method', 'amplitude', 'sigma', 'q']})
dump(OUT / 'parameters.json', params)
by = {(int(p['d']), int(p['ir']), p['method'], p['task']): p for p in pred}

# 原始 X1 仅在此规划脚本解析成功指示；从不传给预测程序。
# 10维联合协方差保留旧新任务及五方法配对关系。
covs = {}
for d, ir in itertools.product([128, 256], range(3)):
    vals, ws = [], []
    for seed in range(192):
        rel = f'experiments/x1/X1_formal_d{d}_r{ir}_s{seed}.npz'
        with np.load(read(rel), allow_pickle=False) as f:
            v = np.column_stack([f[m + '_' + task + '_success'] for task in ['old', 'new'] for m in METHODS]).astype(float)
            assert v.shape == (2048, 10)
            vals.append(v.mean(axis=0))
            ws.append(np.cov(v, rowvar=False, ddof=1))
    covs[d, ir] = dict(C=np.cov(vals, rowvar=False, ddof=1), W=np.mean(ws, axis=0))
dump(OUT / 'planning_covariances.json', [dict(d=d, ir=ir, C2048=c['C'].tolist(), W=c['W'].tolist())
                                       for (d, ir), c in covs.items()])

# 方差外推：d256 的数据间剩余方差非负部分 + W/Q。
# 在单一目标 GD-Sign 上额外报告原始线性外推；不许用负方差获利。
vcontrast = np.zeros(10); vcontrast[0] = 1; vcontrast[4] = -1
c = covs[256, 2]
var2048 = float(vcontrast @ c['C'] @ vcontrast)
within = float(vcontrast @ c['W'] @ vcontrast)
floor = max(0., var2048 - within / 2048)
p0 = by[512, 2, 'GD', 'old']['recall_limit'] - by[512, 2, 'Sign', 'old']['recall_limit']
p1 = by[512, 2, 'GD', 'old']['recall_first_order'] - by[512, 2, 'Sign', 'old']['recall_first_order']
delta = p1 - p0
kernel = json.loads(read('review13_checks/kernel_budget.json').read_text())['rows']
# 固定成本模型：核心成本×1.5，加每格 0.035*d*n/1e6 CPU秒（3正态数组/混合/分配/保存）,
# 加 1 CPU秒/格；无新计时。只给目标格增加查询，其余512。
def cost(N, Q):
    total = 0.
    for row in kernel:
        d = row['d']
        for ir in range(3):
            q = Q if (d, ir) == (512, 2) else 512
            core = row['training_3_GEMM_cpu'] + row['candidate_10passes_cpu'] * q / 512
            total += 1.5 * core + .035 * d * row['n'] / 1e6 + 1.
    return N * total + 180.  # 启动/哈希/分析储备

grid = []
for Q, N in itertools.product([512, 1024, 2048, 4096, 8192], range(24, 385, 8)):
    sd = math.sqrt(floor + within / Q)
    nc = abs(delta) * math.sqrt(N) / (2 * sd)  # 损失改善需越过两预测的中点
    power = float(nct.sf(t.ppf(1 - .05 / 6, N-1), N-1, nc))
    # 1.25方差膨胀敏感性也须≥80%，避免规划在80%边缘。
    inflated = float(nct.sf(t.ppf(1 - .05 / 6, N-1), N-1, nc / math.sqrt(1.25)))
    grid.append(dict(N=N, Q_target=Q, Q_other=512, cpu_estimate=cost(N, Q),
                     wall_estimate=cost(N, Q)/1.8, power=power, power_variance_x1_25=inflated))
table(OUT / 'power_grid.csv', grid)
eligible = [r for r in grid if r['cpu_estimate'] <= 21000 and r['wall_estimate'] <= 12000
            and r['power_variance_x1_25'] >= .8]
# 在保守敏感性功效达标方案中选预计CPU最少，其次N较小/Q较小。规则在查表前固定于代码。
chosen = min(eligible, key=lambda r: (r['cpu_estimate'], r['N'], r['Q_target'])) if eligible else next(
    r for r in grid if r['N'] == 64 and r['Q_target'] == 512)
confirm = bool(eligible)
cfg = dict(blocks=chosen['N'], query_default=512, query_target=chosen['Q_target'],
           target_d=512, target_ir=2, target_pair=['GD', 'Sign'], d512_confirmatory=confirm,
           confirm_family=3 if confirm else 2, min_blocks=24, equivalence_band=.005,
           runner_cpu_soft=21100, runner_cpu_hard=21340, analysis_cpu_soft=45, analysis_cpu_hard=50,
           total_cpu_hard=21600, formal_cpu_hard=21400, preparation_cpu_reserve=200,
           runner_wall_soft=12300, runner_wall_hard=12525,
           analysis_wall_hard=60, total_wall_hard=12600, query_chunk=256, candidate_chunk=2048,
           rng_entropy_prefix=[20261106, 87410329, 83], bit_generator='PCG64', dtype='float64')
dump(OUT / 'config.json', cfg)
dump(OUT / 'power_summary.json', dict(target='d512,r=-.5,old GD-Sign squared-loss improvement',
     uncorrected=p0, corrected=p1, shift=delta, variance2048=var2048, within_variance=within,
     nonnegative_floor=floor, chosen=chosen, d512_confirmatory=confirm,
     model='truth=frozen corrected; d512 covariance extrapolated from X1 d256; not a guarantee',
     criterion='family3 two-sided simultaneous95% lower bound > 0; 1.25x variance sensitivity >=80%',
     cost_model='1.5*(train+scan*Q/512)+0.035*d*n/1e6+1 per cell; +180 total; wall=CPU/1.8',
     all_grid_preserved=True))
streams = [dict(d=d, ir=ir, seed=s, stream=k, entropy=entropy(d, ir, s, k))
           for d, ir, s in jobs(cfg['blocks']) for k in range(4)]
def state(e):
    return tuple(int(x) for x in np.random.SeedSequence(e).generate_state(8))
new = {state(r['entropy']): r['entropy'] for r in streams}
assert len(new) == len(streams)
dump(OUT / 'streams.json', streams)

# 重建所有可见旧熵及静态初始化器；同审查13的保守域，另显式加入Y3草案。
old = {}
def add(e, source):
    if isinstance(e, (list, tuple)) and e and all(isinstance(x, int) and x >= 0 for x in e):
        old.setdefault(tuple(e), set()).add(source)
def walk(obj, source):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if 'entropy' in k.lower():
                add(v, source)
                if isinstance(v, list):
                    for j in range(6): add(v + [j], source)
            walk(v, source)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)): walk(v, source)
inventory = []
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or p.is_symlink() or OUT in p.parents: continue
    rel = str(p.relative_to(ROOT))
    if p.suffix in ['.json', '.jsonl'] and p.stat().st_size < 25000000:
        txt = p.read_text()
        if 'entropy' in txt:
            read(rel)
            if p.suffix == '.json': walk(json.loads(txt), rel)
            else:
                for line in txt.splitlines():
                    if 'entropy' in line: walk(json.loads(line), rel)
    if p.suffix == '.py':
        txt = p.read_text(); read(rel)
        for i, line in enumerate(txt.splitlines(), 1):
            if any(s in line for s in ['SeedSequence', 'default_rng', 'RandomState', 'random.seed', 'manual_seed', 'Sobol(']):
                inventory.append(dict(file=rel, line=i, text=line.strip()))
        try: tree = ast.parse(txt)
        except SyntaxError: continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ['default_rng', 'SeedSequence', 'RandomState'] and node.args:
                try: v = ast.literal_eval(node.args[0])
                except (ValueError, TypeError): continue
                add([v] if isinstance(v, int) else v, rel)
for ph, d, ir, s, k in itertools.product([11,22,33], [16,128,256,512], range(3), range(192), range(4)):
    add([20261106,913752,ph,d,ir,s,k], 'X1 conservative')
for d, ir, s, k in itertools.product(DIMS, range(3), range(64), range(4)):
    add([20261106,30917401,71,d,ir,s,k], 'Y3 draft')
# 执行旧审查代码的域列表字面副本，绝不导入执行旧程序。
families = {
 'W4': ((20261106,440004,k,d,s,j) for k,d,s,j in itertools.product([1,2,9],[16,32,64,128,256,512],range(32),range(6))),
 'W35a': ((350611,10,d,s) for d,s in itertools.product([16,32,64,128],range(4))),
 'W35b': ((350611,20,d,g,s) for d,g,s in itertools.product([32,64,128,256],[16,64],range(5))),
 'W7': ((730619,c,d,s) for c,d,s in itertools.product(range(6),[32,64,128],range(4))),
 'W8a': ((20261106,880008,d,s) for d,s in itertools.product([64,128,256],range(64))),
 'W8b': ((880082,d,s) for d,s in itertools.product([32,64,96],range(16))),
 'W9a': ((900000+10000*d+s,) for d,s in itertools.product([64,128,256],range(8))),
 'W9b': ((19000000+s,) for s in range(24)),
 'X2': ((20261106,220099,d,s) for d,s in itertools.product([64,128,256,512],range(96))),
 'X3': ((202611063,g,s) for g,s in itertools.product([16,64],range(3))),
 'review6a': ((611063027,d,c,s) for d,c,s in itertools.product([64,128],range(6),range(16))),
 'review6b': [(611063104,)]+[(611064018,d,c,s) for d,c,s in itertools.product([64,128],[1,2],range(12))],
 'review8': ((8117903,d,s,c) for d,s,c in itertools.product([16,32,64,128,256,512,1024],range(64),range(4))),
 'smoke': ((100000*d+1000*k+s,) for d,k,s in itertools.product([32,64,128],[1,2,4],range(5))),
 'p01': ((2026110600+100000*d+1000*g+s,) for d,g,s in itertools.product([16,32,64,128,256,512],[2,4],range(64))),
 'review12': ((120000+d*10+s,) for d,s in itertools.product([256,512],range(3))),
 'integration': ((s,) for s in range(10000))}
for label, seqs in families.items():
    for ent in seqs: add(ent, label)
collisions = [dict(old=e, new=new[state(e)]) for e in old if state(e) in new]
assert not collisions
dump(OUT / 'rng_inventory.json', inventory)
dump(OUT / 'seed_audit.json', dict(planned_streams=len(new), old_entropy_count=len(old), collisions=collisions,
    limitation='可见记录/初始化器/保守域的256bit派生状态核验；非数学独立性证明，无法核查未记录执行',
    no_training_or_query_samples_generated=True))
for name in ['PROOF_REVIEW13.md', 'Y3_FINITE_D_NONLINEAR.md', 'Y3_PREREG_DRAFT.md',
             'y3_checks/Y3_inputs.json', 'experiments/x1/X1_population.json',
             'review13_checks/complete_seed_and_report_stats.py']:
    read(name)
dump(OUT / 'input_hashes.json', inputs)
dump(OUT / 'environment.json', env)
dump(OUT / 'prepare_status.json', dict(no_formal_sampling=True, start=start, end=usage()))
print(json.dumps(dict(chosen=chosen, d512_confirmatory=confirm, streams=len(new), old_entropies=len(old)), ensure_ascii=False))
