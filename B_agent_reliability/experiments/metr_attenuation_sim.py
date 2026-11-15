"""How much does noise in log(human minutes) attenuate the fitted log-logistic shape k?
Use METR's actual task-length distribution; simulate success from true shape k_true at true lengths,
fit with noisy lengths ln t_obs = ln t_true + N(0, sigma^2). Report fitted k / k_true."""
import json, sys, numpy as np
from scipy.optimize import minimize
rs=[json.loads(l) for l in open(sys.argv[1])]
lt_all=np.log(np.array(sorted({r['task_id']:r['human_minutes'] for r in rs if r.get('human_minutes',0)>0}.values())))
rng=np.random.default_rng(1)
def fit(lt,y):
    def nll(p):
        z=p[0]-p[1]*lt; return np.sum(y*np.logaddexp(0,-z)+(1-y)*np.logaddexp(0,z))
    return minimize(nll,[1,.5],method='L-BFGS-B').x
print(f"tasks={len(lt_all)} sd(ln t)={lt_all.std():.2f}")
print("k_true h50  sigma  mean_k_fit  ratio")
for k_true in [1.0,1.3,1.6]:
  for h50 in [30.0,240.0]:
    for sigma in [0.0,0.5,0.8,1.2]:
        ks=[]
        for rep in range(40):
            lt=np.repeat(lt_all,8)  # 8 runs per task, like METR
            p=1/(1+np.exp(k_true*(lt-np.log(h50))))
            y=(rng.random(len(lt))<p).astype(float)
            lto=lt+rng.normal(0,sigma,len(lt))
            ks.append(fit(lto,y)[1])
        print(f"{k_true:5.1f} {h50:5.0f} {sigma:5.1f} {np.mean(ks):10.2f} {np.mean(ks)/k_true:6.2f}",flush=True)
