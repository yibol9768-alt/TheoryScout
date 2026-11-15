"""保留首次审计失败；精确披露非输入资源账本的并发变化，再核科学边界。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(25,30))
from common_v2 import ROOT,OUT,read,dump,sha,usage,environment

prior=read(OUT/'resume_audit.json')
assert prior['errors']==[{'existing_changed':['PKG_continuation_resources.json']}]
assert prior['collisions']==0 and not prior['new_rng_sources'] and prior['formal_never_started']
baseline=read(OUT/'resume_baseline.json');inputs=read(OUT/'input_hashes.json')
changed={p:dict(before=h,after=sha(ROOT/p)) for p,h in baseline.items() if sha(ROOT/p)!=h}
assert set(changed)<= {'PKG_continuation_resources.json'},changed
assert 'PKG_continuation_resources.json' not in inputs
assert all(sha(ROOT/p)==h for p,h in inputs.items())
assert environment()==read(OUT/'environment.json')
assert not (OUT/'formal').exists()
assert read(OUT/'resume_tests.json')['passed'] and read(OUT/'credential_tests.json')['passed']
dump(OUT/'resume_final_audit.json',dict(passed=True,initial_audit_preserved='z4/resume_audit.json',
    concurrent_noninput_changes=changed,scientific_inputs_unchanged=True,z4_existing_files_unchanged=True,
    seed_audit_reused=prior,training_or_query_draws=0,formal_sampling=False,usage=usage(),
    boundary='未向该资源账本写入；其变化在本轮只新增路径的写操作之外。不能据此宣称整个项目字节均不变。'))
print(dict(passed=True,concurrent_noninput_changes=list(changed),usage=usage()))
