"""只在新建镜像中重跑原合成测试，不修改原测试输出。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(40,21600))
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit_y3b_checks'
target=OUT/'synthetic_mirror'/'y3b';target.mkdir(parents=True)
copied={}
for p in [*(ROOT/'y3b').glob('*.py'),*[ROOT/'y3b'/n for n in ['config.json','predictions.csv','parameters.json']]]:
    dest=target/p.name
    with dest.open('xb') as f:f.write(p.read_bytes())
    copied[p.name]=hashlib.sha256(dest.read_bytes()).hexdigest()
with (OUT/'synthetic_mirror_sources.json').open('x') as f:json.dump(copied,f,indent=2)
with (OUT/'synthetic_rerun.log').open('x') as f:
    p=subprocess.run([sys.executable,'-B',str(target/'test_synthetic.py'),'--name','synthetic_independent_review'],
        cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=45)
assert p.returncode==0,p.returncode
result=json.loads((target/'synthetic_independent_review_tests.json').read_text())
print(json.dumps(dict(original_sources_unchanged=True,passed=result['passed'],checks=len(result['checks']))))
