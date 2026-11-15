import experiment as c
files=['PREREG_ROUND5.md','PREREG_ROUND5.sha256']
files += ['experiments/round5/'+f for f in ('experiment.py','bias_solver.c','bias_solver.so','run.py','smoke.py','assess.py','validate.py','smoke.json')]
assert not (c.OUT/'manifest.json').exists()
expected=(c.ROOT/'PREREG_ROUND5.sha256').read_text().split()[0]
assert c.digest(c.ROOT/'PREREG_ROUND5.md')==expected
c.save(c.OUT/'manifest.json',dict(frozen_epoch=c.time.time(),hashes={f:c.digest(c.ROOT/f) for f in files},copied_sources={f:c.digest(c.ROOT/f) for f in ('experiments/round4/experiment.py','experiments/round4/bias_solver.c')},grid_main=100,grid_holdout=24,threads=dict(OMP=1,BLAS=1,MKL=1,NUMEXPR=1,workers=2),memory_max='4G',memory_swap_max=0))
