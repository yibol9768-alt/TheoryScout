import experiment as c
from experiment import np,json,time
from assess import adjudicate
from run import grid,identifier
# All source and old input hashes are recorded only after implementation smoke.
assert json.loads((c.OUT/'smoke.json').read_text())['budget_feasible']
A,p=c.data(7,901)
checks=[]
for beta in (16.,256.,4096.):
    S=beta*(p['U']@p['V'].T)
    native,info=c.root_bias(S,A)
    reference,ri=c.root_bias_reference(S,A)
    Y=2.*A-1
    residual=float(np.max(np.abs((-Y*c.expit(-Y*(S+native[:,None]))).sum(1))))
    ln=float(np.logaddexp(0,-Y*(S+native[:,None])).mean())
    lr=float(np.logaddexp(0,-Y*(S+reference[:,None])).mean())
    assert residual<1e-9 and abs(ln-lr)<1e-10
    checks.append(dict(beta=beta,residual=residual,loss_error=abs(ln-lr)))
# Run the actual adjudicator and independent terminal-metric replay on two smoke seeds.
records={}
for config in grid((901,902)):
    d,s,m,pa=config
    r=json.loads((c.OUT/'smoke'/f'{identifier(*config)}.json').read_text())
    # Mapping smoke seeds to two reserved slots tests missing-run treatment without inference.
    records[identifier(d,20+(s-901),m,pa)]=r
smoke_assessment=adjudicate(records,[])
assert not smoke_assessment['decisions']['H1']
c.save(c.OUT/'validation.json',dict(root_reference=checks,smoke_assessment=smoke_assessment))
files=[c.ROOT/'PREREG_ROUND4.md',c.ROOT/'PREREG_ROUND4.sha256']+[
 c.OUT/name for name in ('experiment.py','run.py','smoke.py','assess.py','freeze.py','bias_solver.c','bias_solver.so','smoke.json','validation.json')]
files+=sorted((c.ROOT/'experiments/round3/weights').glob('d[67]_s*_*.npz'))
files=[p for p in files if not (p.suffix=='.npz' and 'sigmoid_lift' in p.name)]
c.save(c.OUT/'manifest.json',dict(hashes={str(p.relative_to(c.ROOT)):c.digest(p) for p in files},
    frozen_epoch=time.time(),interpreter=c.sys.executable,numpy=np.__version__,steps=c.T,threads=dict(OMP=3,BLAS=1),device='CPU'))
print(json.dumps(dict(validation='passed',files=len(files),estimate=json.loads((c.OUT/'smoke.json').read_text())['estimated_serial_seconds'])))
