"""只从冻结源 AST 提取纯函数/常量，不导入旧客户端。"""
import ast, hashlib, json, random
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'experiments/real/reliability_core.py'
MODEL='gpt-6-luna'
def dump(x): return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def sha(b): return hashlib.sha256(b).hexdigest()
_tree=ast.parse(SOURCE.read_text())
_nodes=[x for x in _tree.body if (isinstance(x,ast.FunctionDef) and x.name in ('task','transition','parse','required')) or
        (isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('COMMON','RULES') for t in x.targets))]
_ns={'random':random,'json':json}
exec(compile(ast.Module(body=_nodes,type_ignores=[]),str(SOURCE),'exec'),_ns)

def oracle(state,op):
    s=state.copy()
    for ins in op['instructions']:
        a,b,c=[f'r{ins[k]}' for k in ('a','b','c')]
        value=(s[a]+ins['m']*s[b]-s[c]+ins['d'])%997
        s[a]=value;s[c]=(s[c]+value)%997
    for key,value in (('F0',(s['r0']+2*s['r1']-s['r2'])%997),): s[key]=value
    s['F1']=(3*s['F0']+s['r3']-s['r4'])%997
    s['F2']=(s['F1']+2*s['r5'])%997
    return s

class Task:
    def __init__(self,seed):
        self.seed=seed;self.t=_ns['task'](dict(family='sheet',level=0,knob=2,seed=seed,length=64))
        s=self.t['initial'].copy()
        for i,op in enumerate(self.t['operations']):
            s=oracle(s,op)
            if s!=self.t['prefix'][i+1]: raise ValueError('independent_oracle_mismatch')
        self.state=self.t['initial'].copy();self.step=0;self.review=0;self.last=None;self.start=None
        self.terminal=False;self.status='missing';self.counts=[[0,0,0,0,0],[0,0,0,0,0]]
    @property
    def rid(self): return f'HOLDOUT_D_{self.seed}_{self.step}_{self.review}'
    def body(self):
        if self.terminal: raise ValueError('terminal_task_request')
        op=self.t['operations'][self.step];keys=_ns['required'](self.t,op)
        start=self.state.copy() if self.review==0 else self.start
        p=dict(step=self.step,operation=op,required_writes=keys,checkpoint=start.copy(),table=self.t['table'])
        messages=[{'role':'system','content':_ns['COMMON']+_ns['RULES']['sheet']},{'role':'user','content':dump(p)}]
        if self.review:
            repair=dict(mode='validator_triggered_review',instruction='The applied transaction failed validation. Recompute and correct the required writes. State whether you detect a previous error.',
                        step=self.step,operation=op,required_writes=keys,applied_state=self.state.copy(),checkpoint=start.copy(),table=self.t['table'])
            messages.extend([{'role':'assistant','content':self.last or ''},{'role':'user','content':dump(repair)}])
        return dump(dict(model=MODEL,messages=messages,max_completion_tokens=4096)).encode()
    def apply(self,response):
        if self.terminal: raise ValueError('terminal_task_apply')
        choice=response['choices'][0];content=choice['message']['content'];finish=choice['finish_reason']
        expected=self.t['prefix'][self.step+1];before=self.state.copy()
        action,error=_ns['parse'](content,list(expected))
        if action: self.state.update(action['writes'])
        n=sum(self.state.get(k)!=v for k,v in expected.items())+int(action is None);correct=n==0
        if not correct and error is None:error='numeric'
        if finish=='length' and not correct:error='truncated_'+str(error)
        record=dict(seed=self.seed,step=self.step,review=self.review,action=action,content=content,before=before,
                    after=self.state.copy(),expected=expected,n_before=0 if self.review==0 else sum(before.get(k)!=v for k,v in expected.items()),
                    n_after=n,correct=correct,error_type=error)
        z=self.counts[int(self.step>=16)]
        if self.review==0:
            self.start=before.copy();z[0]+=1;z[2]+=int(not correct)
        if self.review==2:z[3]+=1;z[4]+=int(correct)
        if correct:
            self.step+=1;self.review=0;self.last=None;self.start=None
            if self.step==64:self.terminal=True;self.status='success'
        elif self.review==2:
            z[1]+=1;self.terminal=True;self.status='unrepaired_after_2_reviews'
        else:self.review+=1;self.last=content
        return record

class GuardError(Exception): pass

def guard(raw):
    try:
        z=json.loads(raw)
        if type(z) is not dict or z.get('model')!=MODEL: raise ValueError()
        u=z['usage']
        if type(u) is not dict or any(type(u.get(k)) is not int or u[k]<0 for k in ('prompt_tokens','completion_tokens','total_tokens')):raise ValueError()
        if u['prompt_tokens']>16000 or u['completion_tokens']>4096 or u['total_tokens']!=u['prompt_tokens']+u['completion_tokens']:raise ValueError()
        choices=z['choices']
        if type(choices) is not list or len(choices)!=1:raise ValueError()
        if type(choices[0]['message']['content']) is not str or type(choices[0]['finish_reason']) is not str:raise ValueError()
        return z
    except (ValueError,KeyError,TypeError,IndexError,UnicodeError):raise GuardError('envelope_model_usage_token_guard') from None
