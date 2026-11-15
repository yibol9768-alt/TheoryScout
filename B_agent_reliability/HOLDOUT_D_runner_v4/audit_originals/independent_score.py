"""独立状态/五格/机制参考；评分逻辑不调用被审计的 Task 或 score 函数。"""
import base64, math
from common import *
D=json.loads((ROOT/'HOLDOUT_D_DESIGN.json').read_text())
SEEDS=list(range(116064001,116064145));KEYS=['r0','r1','r2','r3','r4','r5','F0','F1','F2']
def compact(z):return json.dumps(z,ensure_ascii=False,separators=(',',':'))
def integer_step(state,op):
    x=dict(state)
    for ins in op['instructions']:
        a='r'+str(ins['a']);b='r'+str(ins['b']);c='r'+str(ins['c'])
        x[a]=(x[a]+ins['m']*x[b]-x[c]+ins['d'])%997
        x[c]=(x[c]+x[a])%997
    x['F0']=(x['r0']+2*x['r1']-x['r2'])%997
    x['F1']=(3*x['F0']+x['r3']-x['r4'])%997
    x['F2']=(x['F1']+2*x['r5'])%997
    return x
def parse_content(s):
    def unique(pairs):
        z={}
        for k,v in pairs:
            if k in z:raise ValueError('duplicate')
            z[k]=v
        return z
    try:
        z=json.loads(s,object_pairs_hook=unique)
        assert type(z) is dict and set(z)=={'writes','detect'} and type(z['detect']) is bool
        assert type(z['writes']) is dict and set(z['writes'])==set(KEYS)
        assert all(type(v) is int and 0<=v<=996 for v in z['writes'].values())
        return z
    except (ValueError,AssertionError,TypeError):return None
def apply_ref(st,body,response,seed):
    msgs=json.loads(body)['messages'];p=json.loads(msgs[1]['content'])
    if st is None:
        assert p['step']==0
        st=dict(c=0,review=0,state=p['checkpoint'].copy(),gold=p['checkpoint'].copy(),counts=[[0]*5,[0]*5],group=[],terminal=False,status='missing')
    assert not st['terminal'] and p['step']==st['c']
    assert len(msgs)==(2 if st['review']==0 else 4)
    if st['review']==0:
        assert p['checkpoint']==st['state'];st['gold']=integer_step(st['gold'],p['operation'])
    gold=st['gold'];before=st['state'].copy();content=response['choices'][0]['message']['content'];a=parse_content(content)
    if a:st['state'].update(a['writes'])
    errors=sum(st['state'].get(k)!=gold[k] for k in KEYS)+int(a is None);correct=errors==0
    st['group'].append(correct)
    record=dict(seed=seed,step=st['c'],review=st['review'],correct=correct,before=before,after=st['state'].copy(),expected=gold.copy(),n_after=errors)
    if correct or st['review']==2:
        z=st['counts'][0 if st['c']<16 else 1]
        z[0]+=1;z[1]+=int(not correct);z[2]+=int(not st['group'][0]);z[3]+=int(len(st['group'])==3);z[4]+=int(len(st['group'])==3 and correct)
        st['group']=[]
        if correct:
            st['c']+=1;st['review']=0
            if st['c']==64:st.update(terminal=True,status='success')
        else:st.update(terminal=True,status='unrepaired_after_2_reviews')
    else:st['review']+=1
    return st,record
def read_events(path):
    raw=Path(path).read_bytes();assert raw.endswith(b'\n');prev='0'*64;events=[]
    for line in raw.splitlines():
        z=json.loads(line);d=z.pop('digest');assert z['seq']==len(events)+1 and z['prev']==prev and digest(compact(z).encode())==d
        z['digest']=d;assert compact(z).encode()==line;events.append(z);prev=d
    return events
def independent_wal(path):
    events=read_events(path);states={s:None for s in SEEDS};requests={};actions={};expected={}
    pending={};f=events[0].get('fixture') or {};success=f.get('success',0);attempts=f.get('attempts',success);halt=None
    for ev in events:
        if ev['kind']=='reserve':
            assert ev['call']==attempts+1;attempts+=1;requests[ev['call']]=ev;pending[ev['call']]=ev
            assert success+len(pending)<=9000 and attempts<=12000 and len(pending)<=2
        elif ev['kind']=='response':
            req=pending.pop(ev['call']);raw=base64.b64decode(ev['raw']);assert digest(raw)==ev['raw_sha256']
            if 200<=ev['status']<300:
                success+=1;body=base64.b64decode(req['body']);seed=req['seed']
                try:response=json.loads(raw);response['choices'][0]['message']['content']
                except (ValueError,KeyError,IndexError):continue
                states[seed],expected[ev['call']]=apply_ref(states[seed],body,response,seed)
            elif ev['status'] not in (0,408) and not 500<=ev['status']<=599:halt='other_http'
        elif ev['kind']=='action':
            for k,v in expected[ev['call']].items():assert ev['record'][k]==v,(k,ev['call'])
            assert ev['call'] not in actions;actions[ev['call']]=ev['record']
        elif ev['kind']=='halt':halt=ev['reason']
        elif ev['kind']=='reconcile' and halt=='unknown_inflight':halt=None
    assert set(expected)==set(actions)
    return describe_ref(states,pending,halt),states
def describe_ref(states,pending,halt):
    cells=[0]*5;bounds={};per=[];details=[]
    for seed,st in states.items():
        if st is None:st=dict(c=0,terminal=False,status='missing',counts=[[0]*5,[0]*5])
        if st['terminal']:
            c=st['c'];j=0 if c<8 else 1 if c<16 else 2 if c<32 else 3 if c<64 else 4;cells[j]+=1
        per.append(st['counts']);details.append(dict(seed=seed,c=st['c'],status=st['status'],terminal=st['terminal']))
    for L in (8,16,32,64):
        K=sum(z['c']>=L for z in details);F=sum(z['terminal'] and z['c']<L for z in details);U=144-K-F
        lo,hi=D['predictions'][str(L)]['count99_D'];pred=D['predictions'][str(L)]['S_D']
        acc=[k for k in range(lo,hi+1) if abs(k/144-pred)<=.1]
        bounds[str(L)]=dict(K=K,F=F,U=U,success_count_bound=[K,144-F],effective_acceptance=[min(acc),max(acc)],robust_incompatible=K>max(acc) or 144-F<min(acc))
    complete=sum(cells)==144;T=sum(cells[j]*D['comparison']['weights_log_D_over_A'][j] for j in range(5)) if complete else None
    return dict(five_cells=cells,terminal=sum(cells),prefix_bounds=bounds,resolved_per_task=per,tasks=details,T=T,G1=complete and not pending and halt is None)
def mechanism_ref(per):
    import numpy as np
    x=np.asarray(per,dtype=np.int64);tot=x.sum(axis=0)
    if min(tot[:,3])<30 or min(tot[:,1])<10:return dict(eligible=False,valid=0,passed=None,status='信息不足',intervals=None)
    rng=np.random.default_rng(116990005);values=[]
    # Deliberately use scalar replicates, different from score.py's batches.
    for i in range(10000):
        z=x[rng.integers(0,144,144)].sum(axis=0)
        if min(z[:,0])<=0 or min(z[:,3])<=0:continue
        values.append([z[0,1]/z[0,0]-z[1,1]/z[1,0],z[1,4]/z[1,3]-z[0,4]/z[0,3]])
    if len(values)<9900:return dict(eligible=False,valid=len(values),passed=None,status='信息不足',intervals=None)
    # Explicit linear order statistic interpolation, not np.quantile.
    cis=[]
    for col in range(2):
        ordered=sorted(v[col] for v in values);interval=[]
        for q in (.025,.975):
            pos=(len(ordered)-1)*q;lo=int(pos);frac=pos-lo
            interval.append(ordered[lo]+(ordered[min(lo+1,len(ordered)-1)]-ordered[lo])*frac)
        cis.append(interval)
    passed=all(p[0]>0 for p in cis)
    return dict(eligible=True,valid=len(values),passed=passed,status='通过' if passed else '未达到机制门槛',intervals=cis)
