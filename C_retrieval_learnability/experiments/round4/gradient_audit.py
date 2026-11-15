"""Additional off-optimum gradient audit; zero-length numerical checks, no training."""
import experiment as c
from experiment import np,json

def main():
    errors={}; chain=[]
    for d in (6,7):
      for mode in c.MODES:
        A,p=c.data(d,903); beta=23.; b,_=c.root_bias(beta*(p['U']@p['V'].T),A)
        bias=np.linspace(-3.,4.,len(A))
        if mode in ('b','b_legacy'): p['b']=bias.copy()
        if mode=='theta': p['theta']=-bias/beta
        p['a']=np.array([np.log(beta)])
        _,g=c.obj(p,A,mode,beta,b)
        _,fastg=c.obj(p,A,mode,beta,b,need_loss=False)
        assert all(np.array_equal(g[k],fastg[k]) for k in g)
        worst=0.
        coordinates=[('U',(3,1)),('U',(100,0)),('V',(2,0)),('V',(12,3)),('a',(0,))]
        if mode in ('b','b_legacy'): coordinates += [('b',(3,)),('b',(511,))]
        if mode=='theta': coordinates += [('theta',(3,)),('theta',(511,))]
        for key,index in coordinates:
            values=[]
            for sign in (1,-1):
                q={k:v.copy() for k,v in p.items()}; q[key][index]+=sign*1e-5
                qb=float(np.exp(q['a'][0])); bb=b
                if mode=='solve': bb,_=c.root_bias(qb*(q['U']@q['V'].T),A,b)
                values.append(c.obj(q,A,mode,qb,bb)[0])
            fd=(values[0]-values[1])/2e-5
            worst=max(worst,abs(fd-g[key][index]))
        errors[f'd{d}_{mode}']=worst
      A,p=c.data(d,903); p['a']=np.array([np.log(beta)]); p['b']=bias.copy()
      _,gb=c.obj(p,A,'b',beta)
      q={k:v.copy() for k,v in p.items() if k!='b'}; q['theta']=-bias/beta
      _,gt=c.obj(q,A,'theta',beta)
      err=abs(gt['a'][0]-gb['a'][0]-(bias*gb['b']).sum())
      assert err<1e-12
      chain.append(dict(d=d,chain_error=float(err),nonzero_chain_term=float((bias*gb['b']).sum())))
    assert max(errors.values())<1e-7
    c.save(c.OUT/'gradient_audit.json',dict(off_optimum_errors=errors,temperature_chain=chain,pass_validation=True))
    print(json.dumps(dict(errors=errors,chain=chain)))
if __name__=='__main__': main()
