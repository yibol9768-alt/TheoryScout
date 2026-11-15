import json,base64,collections
ev=[json.loads(l) for l in open('HOLDOUT_D_live/HOLDOUT_D_WAL.jsonl')]
res={e['call']:e for e in ev if e['kind']=='reserve'}
acts={e['call']:e['record'] for e in ev if e['kind']=='action'}
ok=bad=skip=0; kinds=collections.Counter(); shown=0
for c,r in res.items():
    b=json.loads(base64.b64decode(r['body'])); u=json.loads(b['messages'][-1]['content'])
    kinds[tuple(sorted(u.get('operation',{}).keys()))]+=1
    a=acts[c]; op=u.get('operation',{})
    if set(op)!={'instructions'} or u.get('table'): skip+=1; 
    if set(op)!={'instructions'}: continue
    st=dict(u['checkpoint']) 
    for ins in op['instructions']:
        A,B,C=f"r{ins['a']}",f"r{ins['b']}",f"r{ins['c']}"
        st[A]=(st[A]+ins['m']*st[B]-st[C]+ins['d'])%997; st[C]=(st[C]+st[A])%997
    st['F0']=(st['r0']+2*st['r1']-st['r2'])%997; st['F1']=(3*st['F0']+st['r3']-st['r4'])%997; st['F2']=(st['F1']+2*st['r5'])%997
    if st==a['expected']: ok+=1
    else:
        bad+=1
        if shown<2: shown+=1; print('diff',c,u.get('step'),a['review'],{k:(st[k],a['expected'].get(k)) for k in st if st[k]!=a['expected'].get(k)}, 'sys:',b['messages'][0]['content'][-300:])
print('ops',kinds,'recomputed ok',ok,'mismatch',bad)
# does before match checkpoint in prompt?
cm=sum(1 for c,r in res.items() if json.loads(json.loads(base64.b64decode(r['body']))['messages'][-1]['content']).get('checkpoint')==acts[c]['before'])
print('checkpoint==before',cm,'/',len(res))
# show one review prompt
for c,r in res.items():
    if acts[c]['review']==1:
        b=json.loads(base64.b64decode(r['body'])); print('REVIEW PROMPT user:',b['messages'][-1]['content'][:700]); break
