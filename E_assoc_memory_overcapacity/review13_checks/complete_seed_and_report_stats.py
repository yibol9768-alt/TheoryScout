"""Complete source-domain audit and report statistics, no data generation."""
import csv,ctypes,hashlib,itertools,json,math,os,resource,time
from pathlib import Path
import numpy as np
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_CPU,(60,65));start=time.process_time()
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:assert os.environ[name]=='2'
cg=next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
CP=Path('/sys/fs/cgroup')/cg.lstrip('/')
assert (CP/'memory.max').read_text().strip()=='3221225472' and (CP/'memory.swap.max').read_text().strip()=='0'
threads=[]
for path in {s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines() if 'openblas' in s and '.so' in s}:
    lib=ctypes.CDLL(path)
    for symbol in ['scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads','openblas_get_num_threads64_','openblas_get_num_threads']:
        if hasattr(lib,symbol):
            fn=getattr(lib,symbol);fn.restype=ctypes.c_int;threads.append(fn());break
assert threads and max(threads)<=2
def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False)
def state(ent):return tuple(int(v) for v in np.random.SeedSequence(list(ent)).generate_state(8))
planned=[(20261106,30917401,71,d,ir,s,stream) for d,ir,s,stream in itertools.product([128,256,512],range(3),range(64),range(4))]
new={state(ent):ent for ent in planned};families={}
def add(label,seqs):families[label]=set(tuple(ent) for ent in seqs)
add('W4 all kinds and dimensions conservative',((20261106,440004,k,d,s,stream) for k,d,s,stream in itertools.product([1,2,9],[16,32,64,128,256,512],range(32),range(6))))
add('W35 matching',((350611,10,d,s) for d,s in itertools.product([16,32,64,128],range(4))))
add('W35 stabilizer',((350611,20,d,g,s) for d,g,s in itertools.product([32,64,128,256],[16,64],range(5))))
add('W7 review FT',((730619,c,d,s) for c,d,s in itertools.product(range(6),[32,64,128],range(4))))
add('W8 simulation conservative',((20261106,880008,d,s) for d,s in itertools.product([64,128,256],range(64))))
add('W8 recall',((880082,d,s) for d,s in itertools.product([32,64,96],range(16))))
add('W9',((900000+10000*d+s,) for d,s in itertools.product([64,128,256],range(8))))
add('W9 focus',((19000000+s,) for s in range(24)))
add('X2 scatter conservative',((20261106,220099,d,s) for d,s in itertools.product([64,128,256,512],range(96))))
add('X3 matrix audit',((202611063,g,s) for g,s in itertools.product([16,64],range(3))))
add('review6 MR',((611063027,d,c,s) for d,c,s in itertools.product([64,128],range(6),range(16))))
add('review6 edges',[(611063104,)]+[(611064018,d,c,s) for d,c,s in itertools.product([64,128],[1,2],range(12))])
add('review8 conservative',((8117903,d,s,c) for d,s,c in itertools.product([16,32,64,128,256,512,1024],range(64),range(4))))
add('sanity',((s,) for s in range(5)))
add('smoke_v2',((100000*d+1000*k+s,) for d,k,s in itertools.product([32,64,128],[1,2,4],range(5))))
add('p01 scalar conservative',((2026110600+100000*d+1000*g+s,) for d,g,s in itertools.product([16,32,64,128,256,512],[2,4],range(64))))
add('review12',((120000+d*10+s,) for d,s in itertools.product([256,512],range(3))))
# Integration-only Sobol seeds are also domain-separated; use all explicit
# integer keyword seeds and documented variable ranges conservatively.
add('integration seed conservative',((s,) for s in range(10000)))
checks=[]
for label,ents in families.items():
    matches=[{'old':ent,'new':new[state(ent)]} for ent in ents if state(ent) in new]
    assert not matches
    checks.append({'family':label,'conservative_entropy_count':len(ents),'collisions':matches})
save('additional_seed_domains.json',{'families':checks,'total_additional_domain_checks':sum(len(v) for v in families.values()),
    'no_random_training_arrays_or_queries_drawn':True,'threads':threads,'cpu_seconds':time.process_time()-start})

with (OUT/'d512_power.csv').open() as f:power=list(csv.DictReader(f))
selected=[]
for task in ['old','new']:
    vals=[r for r in power if r['task']==task and r['N']=='64']
    selected.append({'task':task,'min_shift_pp':min(float(r['shift_pp']) for r in vals),'max_shift_pp':max(float(r['shift_pp']) for r in vals),
        'min_halfwidth72_pp':min(float(r['halfwidth72_pp']) for r in vals),'max_halfwidth72_pp':max(float(r['halfwidth72_pp']) for r in vals),
        'min_power72':min(float(r['approximate_two_sided_power72']) for r in vals),'max_power72':max(float(r['approximate_two_sided_power72']) for r in vals)})
pick=[r for r in power if r['N']=='64' and ((r['ir']=='2' and r['label'] in ['GD','Adam.1','GD-Sign','Adam.1-Sign']) or (r['ir']=='0' and r['label']=='GD-Sign'))]
crit=float(norm.ppf(1-.05/144));stats=[]
for r in pick:
    sd=float(r['projected_sd512_queries_pp'])/100;shift=float(r['shift_pp'])/100
    stats.append({**r,'normal_N_80pct_family72':((crit+norm.ppf(.8))*sd/abs(shift))**2})
save('report_statistics.json',{'d512_cell_ranges':selected,'selected_cells':stats,
    'normal_family72_critical':crit,'expected_loss_dimension_shares':{'d128':16/21,'d256':4/21,'d512':1/21},
    'note':'Power is approximate and X1-covariance-based, with d512 variance extrapolated from d256; scientific truth not assumed known.'})

source=(ROOT/'y3_checks/Y3_coefficients.py').read_text().splitlines()
interesting=[]
for no,line in enumerate(source,1):
    if any(s in line for s in ['POP =','COEFF =','OBS=[]','CON=[]','FORM=','def coefficients','def response','X1_new_matching.csv','X1_primary.csv']):interesting.append({'line':no,'text':line.strip()})
save('coefficient_dependency_audit.json',{'static_source_order':interesting,
    'optimizer_calls_found':[line for line in source if any(s in line for s in ['curve_fit','least_squares','polyfit','minimize(','lstsq('])],
    'interpretation':'No direct numeric fitting path from observed X1 outcomes into COEFF; choosing formula/closure after X1 remains a researcher degree of freedom.'})
print(json.dumps({'additional_seed_checks':sum(len(v) for v in families.values()),'d512_cell_ranges':selected,'cpu_seconds':time.process_time()-start},ensure_ascii=False))
