"""Float64 retrieval model, analytic backpropagation and exact-SVD directions."""
import math
import numpy as np

H, D, DH, L = 4, 32, 8, 8
SCALE = math.sqrt(DH)


def make_bundle(seed, condition, steps=600):
    # Reset streams for both conditions: same latent Gaussians, indices, init.
    data_rng, init_rng, batch_rng = [np.random.default_rng(s) for s in
                                     np.random.SeedSequence(seed).spawn(3)]
    coord = np.ones(D) if condition == 'isotropic' else 4.0**(-np.arange(D)/31)
    def dataset(n):
        keys = data_rng.normal(size=(n, L, D)) * coord
        labels = data_rng.integers(L, size=n)
        query = keys[np.arange(n), labels].copy() + .1*data_rng.normal(size=(n,D))*coord
        query *= math.sqrt(D)/np.linalg.norm(query, axis=-1, keepdims=True)
        keys *= math.sqrt(D)/np.linalg.norm(keys, axis=-1, keepdims=True)
        target = np.full((n,L), .1/L)
        target[np.arange(n), labels] += .9
        return {'x': query, 'keys': keys, 'labels': labels, 'target': target}
    bundle = {split: dataset(n) for split,n in [('train',4096),('test',512),('probe',128)]}
    bundle['initial'] = [init_rng.normal(size=(D,D))*.1/math.sqrt(D) for _ in range(2)]
    batches = []
    while len(batches) < steps:
        batches.extend(batch_rng.permutation(4096).reshape(-1,32))
    bundle['batches'] = np.asarray(batches[:steps])
    return bundle


def subset(data, ix):
    return {k:v[ix] for k,v in data.items()}


def norm_forward(x, intervention):
    if intervention not in ('layer_norm','head_norm'):
        return x, None
    shape = x.shape
    y = x.reshape(*shape[:-2],32) if intervention == 'layer_norm' else x
    r = np.sqrt(np.mean(y*y, axis=-1, keepdims=True)+1e-8)
    return (y/r).reshape(shape), (y,r,shape)


def norm_backward(g, cache):
    if cache is None:
        return g
    x,r,shape = cache
    g = g.reshape(x.shape)
    return (g/r-x*np.sum(x*g,axis=-1,keepdims=True)/(x.shape[-1]*r**3)).reshape(shape)


def forward(weights, data, intervention='base', grad=False):
    wq,wk = weights
    x,keys,target = data['x'],data['keys'],data['target']
    n = x.shape[0]
    q = (x@wq.T).reshape(n,H,DH)
    k = (keys.reshape(-1,D)@wk.T).reshape(n,L,H,DH)
    q,qcache = norm_forward(q, intervention)
    k,kcache = norm_forward(k, intervention)
    logits = np.einsum('bhd,bjhd->bhj',q,k)/SCALE
    shifted = logits-logits.max(axis=-1,keepdims=True)
    logp = shifted-np.log(np.exp(shifted).sum(axis=-1,keepdims=True))
    loss = float(-np.sum(target[:,None,:]*logp)/(n*H))
    result = {'loss': loss, 'logits':logits, 'q':q, 'k':k,
              'accuracy':float(np.mean(logits.argmax(axis=-1)==data['labels'][:,None])),
              'ensemble_accuracy':float(np.mean(np.exp(logp).mean(axis=1).argmax(axis=-1)==data['labels']))}
    if grad:
        e = (np.exp(logp)-target[:,None,:])/(n*H)
        dq = np.einsum('bhj,bjhd->bhd',e,k)/SCALE
        dk = np.einsum('bhj,bhd->bjhd',e,q)/SCALE
        dq,dk = norm_backward(dq,qcache),norm_backward(dk,kcache)
        result['grad'] = [dq.reshape(n,D).T@x, dk.reshape(-1,D).T@keys.reshape(-1,D)]
    return result


class Optimizer:
    def __init__(self, name):
        self.name = name
        self.t = 0
        self.m = [np.zeros((D,D)),np.zeros((D,D))]
        self.v = [np.zeros((D,D)),np.zeros((D,D))]
    def direction(self, gradients):
        self.t += 1
        out, dropped = [], []
        for i,g in enumerate(gradients):
            if self.name.startswith('muon'):
                beta = .9 if self.name == 'muon09' else 0.
                self.m[i] = beta*self.m[i]+(1-beta)*g
                u,s,vt = np.linalg.svd(self.m[i],full_matrices=False)
                keep = s>1e-12*s[0] if s[0]>0 else np.zeros(D,dtype=bool)
                out.append(u[:,keep]@vt[keep,:])
                dropped.append(int((~keep).sum()))
            elif self.name == 'adamw':
                self.m[i] = .9*self.m[i]+.1*g
                self.v[i] = .99*self.v[i]+.01*g*g
                out.append((self.m[i]/(1-.9**self.t))/(np.sqrt(self.v[i]/(1-.99**self.t))+1e-8))
                dropped.append(0)
            else:
                out.append(np.sign(g))
                dropped.append(0)
        return out,dropped


def freeze(weights):
    wq,wk = [w.reshape(H,DH,D) for w in weights]
    frames = []
    for h in range(H):
        _,e = np.linalg.eigh(wq[h]@wq[h].T+wk[h]@wk[h].T)
        u = e[:,-1]
        if u[np.argmax(np.abs(u))]<0:
            u = -u
        v,w = wq[h].T@u,wk[h].T@u
        if min(np.linalg.norm(v),np.linalg.norm(w)) == 0:
            frames.append(None)
        else:
            frames.append([u,v/np.linalg.norm(v),w/np.linalg.norm(w)])
    return frames


def project(matrices, frames):
    ans = np.full((H,2),np.nan)
    for h,f in enumerate(frames):
        if f is not None:
            u,v,w = f
            ans[h] = [u@matrices[0][h*DH:(h+1)*DH]@v,
                      u@matrices[1][h*DH:(h+1)*DH]@w]
    return ans


def measure(weights,data,directions,k0,intervention,frames=None,reference=None):
    result = forward(weights,data,intervention)
    logits = result['logits']
    q,k = result['q'],result['k']
    ans = {'ce':result['loss'],'accuracy':result['accuracy'],
           'ensemble_accuracy':result['ensemble_accuracy'],
           'Mplus':float(max(0,logits.max())), 'Mabs':float(np.abs(logits).max()),
           'q999':float(np.quantile(logits,.999)),
           'abs_q999':float(np.quantile(np.abs(logits),.999)),
           'head_Mabs':np.abs(logits).max(axis=(0,2)),
           'q_rms':np.sqrt(np.mean(q*q,axis=(0,2))),
           'k_rms':np.sqrt(np.mean(k*k,axis=(0,1,3))),
           'weight_op':[float(np.linalg.norm(w,2)) for w in weights],
           'head_product_op':[float(np.linalg.norm(weights[0][h*DH:(h+1)*DH].T@
                                                weights[1][h*DH:(h+1)*DH],2)) for h in range(H)],
           'direction_fro':[float(np.linalg.norm(d)/k) for d,k in zip(directions,k0)],
           'direction_op':[float(np.linalg.norm(d,2)/k) for d,k in zip(directions,k0)]}
    if frames is not None:
        ab = project(weights,frames)
        residual,explained,mode_mse = [],[],[]
        for h,f in enumerate(frames):
            if f is None:
                residual.append(np.nan); explained.append(np.nan); mode_mse.append(np.nan)
                continue
            u,v,w = f
            product = weights[0][h*DH:(h+1)*DH].T@weights[1][h*DH:(h+1)*DH]
            approximate = ab[h,0]*ab[h,1]*np.outer(v,w)
            residual.append(float(np.linalg.norm(product-approximate)/(np.linalg.norm(product)+1e-12)))
            mode_logits = np.einsum('bd,d,bjn,n->bj', data['x'],v,data['keys'],w)*np.prod(ab[h])/SCALE
            mode_mse.append(float(np.mean((mode_logits-logits[:,h])**2)))
            if reference is None:
                explained.append(np.nan)
            else:
                full_delta = logits[:,h]-reference['logits'][:,h]
                mode_delta = mode_logits-reference['mode_logits'][:,h]
                explained.append(float(1-np.sum((full_delta-mode_delta)**2)/(np.sum(full_delta**2)+1e-30)))
        ans.update({'ab':ab,'residual':residual,'explained_change':explained,'mode_logit_mse':mode_mse})
    return ans


def reference_at_freeze(weights,data,frames,intervention):
    logits = forward(weights,data,intervention)['logits']
    ab = project(weights,frames)
    ml = np.zeros_like(logits)
    for h,f in enumerate(frames):
        if f is not None:
            _,v,w = f
            ml[:,h] = np.einsum('bd,d,bjn,n->bj',data['x'],v,data['keys'],w)*np.prod(ab[h])/SCALE
    return {'logits':logits,'mode_logits':ml}
