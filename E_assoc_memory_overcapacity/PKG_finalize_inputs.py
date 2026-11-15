from pathlib import Path
import json
p=Path('supplement/statements/adam_history.md');s=p.read_text();s=s.replace('# history：','# ')
s='\n'.join(line for line in s.splitlines() if not line.startswith('2026-11-06。对应'))
s=s.replace('FT-R（`paper/appendix/K_ft_rank.tex`）','双任务训练秩定理')
p.write_text(s+'\n')
p=Path('supplement/statements/manuscript_hashes.json');d=json.loads(p.read_text());names={'F':'exact_polar_variance','L_multistep':'two_step_adam','fine_tuning':'fine_tuning_main','K_ft_rank':'joint_training_rank','M_nonlinear_finite_d':'nonlinear_finite_dimension','experiments':'experiments_main'};p.write_text(json.dumps({names[k]:v for k,v in d.items()},indent=2)+'\n')
p=Path('supplement/two_step_adam_certificate/check_adam.py');s=p.read_text()
s=s.replace('    def inverse(self):','    def square(self):return Dual(self.x**2,2*self.x*self.da,2*self.x*self.de)\n    def inverse(self):')
s=s.replace('    def root(self):\n        v=[self.c[0].root()]','    def square(self):\n        value=self*self;value.c[0]=self.c[0].square();return value\n    def root(self):\n        v=[self.c[0].root()]')
s=s.replace('residual*residual','residual.square()').replace('linear=p*weight;squared=p*p*weight','linear=p*weight;squared=p.square()*weight')
p.write_text(s)
