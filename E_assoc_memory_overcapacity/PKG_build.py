from pathlib import Path
import hashlib, json, re, shutil
root=Path.cwd(); out=root/'supplement'; out.mkdir(exist_ok=False)
for name in ['polar_certificate','two_step_adam_certificate','experiments_replay','statements','logs']:(out/name).mkdir()
def put(name,text): (out/name).write_text(text)
def read(name): return (root/name).read_text()
def sanitize(s):
    replacements={'X3_POLAR_TOOL.md':'README.md','Y2_MULTISTEP.md':'two_step_adam.tex','Z3_ADAM_HISTORY.md':'adam_history.md','w6_checks/adam_certificate.py':'integer_intervals.py','y2_checks/interval_core.py':'integer_intervals.py','W35':'analytic','w35':'analytic','Y2':'two-step','X3':'polar','Z3':'history'}
    for a,b in replacements.items():s=s.replace(a,b)
    return s
originals=['paper/appendix/F.tex','paper/appendix/L_multistep.tex','paper/sections/fine_tuning.tex','paper/appendix/K_ft_rank.tex','paper/appendix/M_nonlinear_finite_d.tex','paper/sections/experiments.tex','x3_checks/polar_tool.py','y2_checks/interval_core.py','y2_checks/certify.py','y2_checks/certify_region.py','y2_checks/certify_points.py','y2_checks/certify_history.py','z3_checks/certify.py','experiments/x1/X1_run_v3.py','y3d/run.py','y3d/analyze.py']
(root/'PKG_original_sources.json').write_text(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in originals},indent=2))
put('statements/polar_variance.tex',sanitize(read('paper/appendix/F.tex')))
put('statements/two_step_adam.tex',sanitize(read('paper/appendix/L_multistep.tex')))
put('statements/adam_history.md',sanitize(read('Z3_ADAM_HISTORY.md').split('## 8.')[0]))
put('statements/finite_dimension.tex',sanitize(read('paper/appendix/M_nonlinear_finite_d.tex')))
put('statements/matched_retrieval.tex',read('paper/sections/experiments.tex'))
put('statements/manuscript_hashes.json',json.dumps({Path(p).stem:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in originals if p.startswith('paper/')},indent=2))
common='''def sources():
    base=Path(__file__).resolve().parents[1]
    paths=sorted(base.rglob('*.py'))+sorted((base/'statements').glob('*'))
    return {str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def resources():
    return dict(maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                thread_limits={k:os.environ.get(k) for k in
                  ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
'''
core=read('y2_checks/interval_core.py')
put('two_step_adam_certificate/integer_intervals.py',core+'\n'+common)
s=read('y2_checks/certify.py').replace('from interval_core import *','from integer_intervals import *')
s=s[:s.index("if __name__=='__main__':")]
s=s[:s.index('def resources():')]+common
s=s.replace('Core arithmetic is copied, unchanged, from w6_checks/adam_certificate.py.','Fixed-point integer arithmetic with outward rounding.')
put('two_step_adam_certificate/adam_integrals.py',sanitize(s))
s=read('y2_checks/certify_region.py').replace('from certify import *','from adam_integrals import *')
a=s.index('INPUTS='); b=s.index('def enc',a);s=s[:a]+'hashes=sources\n'+s[b:]
s=s.replace("default='y2_checks/region.json'","default='two_step_adam_certificate/stabilizer_ordering.json'")
s=re.sub(r"source_hashes=\{p:.*?\},seconds=",'source_hashes=sources(),seconds=',s)
put('two_step_adam_certificate/stabilizer_ordering.py',s)
s=read('y2_checks/certify_points.py').replace('from certify import *','from adam_integrals import *').replace("default='y2_checks/points.json'","default='two_step_adam_certificate/matched_chain.json'")
s=re.sub(r"source_hashes=\{p:.*?\},seconds=",'source_hashes=sources(),seconds=',s)
put('two_step_adam_certificate/matched_chain.py',s)
s=read('y2_checks/certify_history.py').replace('from certify import *','from adam_integrals import *')
s=s.replace("with open('y2_checks/points.json')", "with open(args.points)")
s=s.replace("start=time.monotonic();out=[]", "parser=argparse.ArgumentParser();parser.add_argument('--points',default='two_step_adam_certificate/matched_chain.json');parser.add_argument('--output',default='two_step_adam_certificate/moment_history.json');args=parser.parse_args()\n    start=time.monotonic();out=[]")
s=s.replace("open('y2_checks/history.json','x')","open(args.output,'x')")
s=re.sub(r"source_hashes=\{p:.*?\},seconds=",'source_hashes=sources(),seconds=',s)
put('two_step_adam_certificate/moment_history.py',s)
s=read('z3_checks/certify.py').replace('from interval_core import *','from integer_intervals import *')
a=s.index('def resources():'); b=s.index('def F(x):',a);s=s[:a]+common+'\n'+s[b:]
s=s.replace('Reuses a verbatim copy of the existing y2 integer arithmetic in this folder.','Uses fixed-point integer outward arithmetic.')
s=s.replace("default='z3_checks/certificate.json'","default='two_step_adam_certificate/history_reversal.json'")
a=s.index('    source_names ='); b=s.index('    before = hashes()',a);s=s[:a]+'    hashes = sources\n'+s[b:]
s=s.replace('st, _, _ = endpoint(ai, e, b1, b2, r, 64, 4)','st, raw, err = endpoint(ai, e, b1, b2, r, 64, 4)')
s=s.replace("N=enc(st['N'])))", "N=enc(st['N']), integrals=[enc(v) for v in raw], errors=[enc(v) for v in err])))")
s=s.replace('ends.append(enc(H))',"ends.append(dict(H=enc(H), integrals=[enc(v) for v in endpoint(F(x), e, b1, b2, r)[1]]))")
put('two_step_adam_certificate/history_reversal.py',s)
s=read('x3_checks/polar_tool.py')
s=s.replace('import time','import time\nimport os, resource\n\nRAW_EFFICIENCY=[]\nRAW_KERNEL=[]\n'+common)
s=s.replace('acc = acc+I(vl, vh).sum()/I(n*n)',"contribution=I(vl, vh).sum()/I(n*n)\n        RAW_KERNEL.append(dict(gamma=gamma,start=start,stop=min(start+batch,n),integral=contribution.out()))\n        acc = acc+contribution")
s=s.replace('return answers[0]/answers[1].sqrt()',"RAW_EFFICIENCY.append(dict(e_numerator=eps_num,e_denominator=eps_den, A=answers[0].out(), Q=answers[1].out(), cells=len(cells)))\n    return answers[0]/answers[1].sqrt()")
# Check the original argument names before inserting raw records.
m=re.search(r'def efficiency_interval\(([^)]*)\)',s); print('efficiency signature',m.group(1))
s=s.replace('eps_num','num').replace('eps_den','den')
a=s.index('    inputs = ['); b=s.index('    return {"method"',a);s=s[:a]+s[b:]
s=re.sub(r'"inputs_sha256": \{p:.*?for p in inputs\},','"source_hashes": sources(), "raw_efficiency_integrals":RAW_EFFICIENCY, "raw_kernel_batches":RAW_KERNEL,',s)
s=s.replace('default=2048)','default=4096)')
s=s.replace('    assert out.parent.resolve() == (Path.cwd()/"x3_checks").resolve()\n','')
put('polar_certificate/polar_variance.py',sanitize(s))
# Package only seed-level measurements, never the already computed test statistics.
rows=[json.loads(l) for l in read('experiments/x1/X1_formal_rows.jsonl').splitlines()]
for r in rows:
    for k in ['seconds_config','phase']:r.pop(k,None)
put('experiments_replay/matched_retrieval_measurements.jsonl',''.join(json.dumps(r)+'\n' for r in rows))
for src,dst in [('y3d/formal/analysis/all_completed.csv','finite_dimension_measurements.csv'),('y3d/predictions.csv','finite_dimension_predictions.csv'),('y3d/parameters.json','population_parameters.json'),('y3d/config.json','confirmation_design.json')]:
    put('experiments_replay/'+dst,read(src))
# Population values are numerical inputs, regenerated and checked by the replay program.
pop=json.loads(read('experiments/x1/X1_population.json'));pop.pop('meta',None)
put('experiments_replay/population_predictions.json',json.dumps(pop,indent=2))
