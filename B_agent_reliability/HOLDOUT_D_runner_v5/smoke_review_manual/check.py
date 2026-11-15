import json,hashlib,base64,collections,sys
def dump(o): return json.dumps(o,separators=(',',':'),ensure_ascii=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
lines=open('HOLDOUT_D_live/HOLDOUT_D_WAL.jsonl').read().splitlines()
ev=[json.loads(l) for l in lines]
head='0'*64; ok=True
for i,e in enumerate(ev,1):
    d=e.pop('digest')
    if e['seq']!=i or e['prev']!=head or sha(dump(e).encode())!=d: ok=False;print('chain break',i);break
    e['digest']=d;head=d
print('chain ok',ok,len(ev))
t=json.load(open('HOLDOUT_D_live/HOLDOUT_D_SMOKE_REVIEW_TEMPLATE.json'))['snapshot']
pre=ev[:t['wal_seq']]
print('wal_sha256 match',sha(''.join(dump(e)+'\n' for e in pre).encode())==t['wal_sha256'], 'digest match',pre[-1]['digest']==t['wal_digest'])
res={e['call']:e for e in ev if e['kind']=='response'}
reserve={e['call']:e for e in ev if e['kind']=='reserve'}
models=collections.Counter();rmodels=collections.Counter();fin=collections.Counter();bad=0
for c,r in reserve.items():
    b=base64.b64decode(r['body']); assert sha(b)==r['body_sha256']; models[json.loads(b)['model']]+=1
    x=res[c]; raw=base64.b64decode(x['raw']); assert sha(raw)==x['raw_sha256']
    j=json.loads(raw); rmodels[j.get('model')]+=1; fin[j['choices'][0]['finish_reason']]+=1
    if x['status']!=200: bad+=1
print('req models',models,'resp models',rmodels,'finish',fin,'non200',bad)
acts=[e for e in ev if e['kind']=='action']
for s in sorted({a['record']['seed'] for a in acts}):
    A=[a['record'] for a in acts if a['record']['seed']==s]
    print('== seed',s,'calls',len(A))
    mism=0
    for a in A:
        try: w=json.loads(a['content'])['writes']
        except Exception: w=None
        if a['action'] and w is not None and a['action']['writes']!=w: mism+=1
        # independent: correct iff after==expected
        if a['correct']!=(a['after']==a['expected']): print(' correct-flag mismatch step',a['step'])
    errs=[(a['step'],a['review'],a['error_type'],a['action']['detect'] if a['action'] else None) for a in A if not a['correct']]
    print(' content/action mismatch',mism,'errors',len(errs)); print(' ',errs[:30])
    print(' reviews>0:',sorted(collections.Counter(a['review'] for a in A).items()), 'last step',A[-1]['step'])
