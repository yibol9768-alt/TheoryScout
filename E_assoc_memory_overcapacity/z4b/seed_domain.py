"""纯整数 SeedSequence 逆映射：不构造 Generator、不抽训练/查询。

算法依据 NumPy v2.5.4 bit_generator.pyx，MIT 许可；本文件独立实现正逆映射。
https://raw.githubusercontent.com/numpy/numpy/v2.5.4/numpy/random/bit_generator.pyx
"""
import ast
import json
import subprocess
import numpy as np
from common_v2 import ROOT,OUT,read,dump,sha,usage

MASK=(1<<32)-1
A=0x931e8875;INIT=0x43b0d7e5;L=0xca01f9dd;R=0x4973f715
def unxor(x):return (x^(x>>16))&MASK
def hash_at(x,c):
    z=((x^c)*((c*A)&MASK))&MASK
    return unxor(z)
def forward(words):
    words=list(words);c=INIT;pool=[]
    for i in range(4):
        pool.append(hash_at(words[i] if i<len(words) else 0,c));c=(c*A)&MASK
    for src in range(4):
        for dst in range(4):
            if src!=dst:
                pool[dst]=unxor((L*pool[dst]-R*hash_at(pool[src],c))&MASK);c=(c*A)&MASK
    for w in words[4:]:
        for dst in range(4):
            pool[dst]=unxor((L*pool[dst]-R*hash_at(w,c))&MASK);c=(c*A)&MASK
    return pool

def inverse_four(pool):
    # 对固定四个输入词，16次 hash 的常数序列完全确定。
    constants=[];c=INIT
    for _ in range(16):constants.append(c);c=(c*A)&MASK
    moves=[];k=4
    for src in range(4):
        for dst in range(4):
            if src!=dst:moves.append((src,dst,constants[k]));k+=1
    p=list(map(int,pool));invL=pow(L,-1,1<<32)
    for src,dst,c in reversed(moves):
        p[dst]=((unxor(p[dst])+R*hash_at(p[src],c))*invL)&MASK
    return [((unxor(p[i])*pow((constants[i]*A)&MASK,-1,1<<32))&MASK)^constants[i] for i in range(4)]

def main():
    start=usage();streams=read(OUT/'streams.json');inverse=[]
    for e in streams:
        pool=list(map(int,np.random.SeedSequence(e).pool))
        assert forward(e)==pool
        words=inverse_four(pool)
        assert forward(words)==pool
        assert list(map(int,np.random.SeedSequence(words).pool))==pool
        # 若四词唯一原像的末词非零，任何<=3词(含scalar<2**96)熵均不可能产生此池。
        assert words[3]!=0
        inverse.append(words)
    cases=0
    for i in range(4096):
        w=[(i*0x9e3779b9)&MASK,(i*i*0x85ebca6b)&MASK,(i*137+41)&MASK,(i*991+7)&MASK]
        assert forward(w)==list(map(int,np.random.SeedSequence(w).pool))
        assert inverse_four(forward(w))==w;cases+=1
    for bits in [0,1,(1<<32)-1,1<<32,(1<<62)-1,(1<<64)-1,(1<<96)-1]:
        w=[];x=bits
        while x:w.append(x&MASK);x>>=32
        w=w or [0]
        pool=list(map(int,np.random.SeedSequence(bits).pool))
        assert forward(w)==pool and inverse_four(pool)==w+[0]*(4-len(w));cases+=1
    # 捕获准备后并发新增源；只登记源与熵，不运行其中任一科学程序。
    inventory=read(OUT/'rng_inventory.json');known={r['file'] for r in inventory}
    paths=subprocess.check_output(['rg','--files','--hidden','-g','*.py','-g','!z4b/**','-g','!Z4B_*','-g','!.git/**'],cwd=ROOT,text=True).splitlines()
    tokens=['SeedSequence','default_rng','RandomState','random.seed','manual_seed','Sobol(']
    inputs=read(OUT/'input_hashes.json');old={tuple(x) for x in read(OUT/'old_entropies.json')};added=[]
    reviewed={'aa1_checks/AA1_bound_audit.py','review19_checks/independent_checks.py','review19_checks/trajectory_checks.py',
              'review20_checks/independent_check.py','review20_checks/summarize_and_audit.py'}
    for rel in paths:
        p=ROOT/rel
        if p.is_symlink():continue
        txt=p.read_text(errors='replace')
        hits=[dict(file=rel,line=i,text=line.strip()) for i,line in enumerate(txt.splitlines(),1) if any(t in line for t in tokens)]
        if not hits or rel in known:continue
        assert rel in reviewed,('unreviewed added source',rel)
        inputs[rel]=sha(p);inventory.extend(hits);added.append(rel)
        for node in ast.walk(ast.parse(txt)):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['SeedSequence','default_rng','RandomState'] and node.args:
                try:v=ast.literal_eval(node.args[0])
                except (ValueError,TypeError):continue
                if type(v) is int and v>=0:old.add((v,))
    # Review19 固定SEED，Review20根三词域和由其抽得的未记录scalar噪声种子：
    # 全部由<=3词结构证明覆盖，不需要重放生成矩阵来恢复noise_seed。
    old.update([(202611060102,),(202611061919,),(202611061920,),(42679,)])
    old.update((202620,d,s) for d in [16,32,64,128,256,512,1024,2048,4096] for s in range(256))
    states={tuple(map(int,np.random.SeedSequence(e).generate_state(8))) for e in streams}
    extra=[e for e in old if len(e)<=3]
    assert not any(tuple(map(int,np.random.SeedSequence(e).generate_state(8))) in states for e in extra)
    for name in ['seed_audit.json','rng_inventory.json','input_hashes.json','old_entropies.json']:
        (OUT/name).rename(OUT/('initial_'+name))
    dump(OUT/'rng_inventory.json',inventory);dump(OUT/'input_hashes.json',inputs);dump(OUT/'old_entropies.json',sorted(map(list,old)))
    previous=read(OUT/'initial_seed_audit.json');previous.update(old_unique=len(old),extension='seed_extension.json',
        all_scalar_entropy_below_2pow96_and_all_at_most_3_uint32_words_disjoint=True,
        new_sources_reviewed=previous['new_sources_reviewed']+added)
    dump(OUT/'seed_audit.json',previous)
    dump(OUT/'seed_preimages.json',inverse)
    dump(OUT/'seed_extension.json',dict(passed=True,training_or_query_draws=0,new_streams=len(streams),
         old_explicit_unique=len(old),added_sources=added,deterministic_roundtrip_cases=cases,
         every_new_stream_unique_4word_preimage_last_word_nonzero=True,
         excluded_entire_domains=['unspawned SeedSequence(pool_size=4), scalar 0<=entropy<2**96',
             'unspawned SeedSequence(pool_size=4), <=3 uint32 entropy words; includes [202620,d,seed] and 0<=noise_seed<2**62'],
         binary_path=np.random.bit_generator.__file__,binary_sha256=sha(np.random.bit_generator.__file__),
         numpy=np.__version__,source_url='https://raw.githubusercontent.com/numpy/numpy/v2.5.4/numpy/random/bit_generator.pyx',
         qualifications='证明池/派生初态不相同，不证明整个周期不相交或数学独立；任意未记录长熵/spawn域不在证明范围',
         usage=usage(),start=start))
    print(dict(passed=True,old_unique=len(old),all_new_streams=len(streams),roundtrips=cases,added_sources=added,usage=usage()))

if __name__=='__main__':main()
