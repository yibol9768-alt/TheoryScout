"""原受审测试仅重定向写入位置；原实现和原反例内容不改。"""
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'audit_y3c_checks'
sys.path.insert(0,str(ROOT/'y3c'))
kind=sys.argv[1]
if kind=='protocol':
    path=ROOT/'y3c/test_protocol.py';source=path.read_text()
    old="TEST=OUT/(sys.argv[2] if len(sys.argv)==3 and sys.argv[1]=='--name' else 'protocol_tests')"
    new="TEST=ROOT/'audit_y3c_checks'/'protocol_replay'"
    assert source.count(old)==1
    source=source.replace(old,new)
elif kind=='synthetic':
    path=ROOT/'y3c/test_synthetic.py';source=path.read_text()
    old='TEST=OUT/name;TEST.mkdir()'
    assert source.count(old)==1
    source=source.replace(old,"TEST=ROOT/'audit_y3c_checks'/'synthetic_replay';TEST.mkdir()")
    old="dump(OUT/(name+'_tests.json'),"
    assert source.count(old)==1
    source=source.replace(old,"dump(ROOT/'audit_y3c_checks'/'synthetic_results.json',")
else: raise ValueError(kind)
with (AUDIT/(kind+'_executed.py')).open('x') as f:f.write(source)
with (AUDIT/(kind+'_adaptation.json')).open('x') as f:
    json.dump({'source':str(path.relative_to(ROOT)),
               'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
               'executed_sha256':hashlib.sha256(source.encode()).hexdigest(),
               'adaptation':'仅改TEST和合成结果JSON的输出路径，禁写原目录'},f,indent=2,ensure_ascii=False)
sys.argv=[str(path)]
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
