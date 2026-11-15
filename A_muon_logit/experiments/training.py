"""Sequential paired retrieval runs and interventions, with raw JSON directions."""
import gzip
import json
import math
import time
import numpy as np
from common import OUT, save, serial, rss_bytes
from model import (make_bundle, subset, forward, Optimizer, freeze, project,
                   measure, reference_at_freeze, H, DH)
from diagnostics import analyze_modes


def run_training(seed,condition,optimizer,eta,intervention,steps,budget,tag='e1',diagnose=True):
    started = time.monotonic()
    budget.check(new_run=True)
    bundle = make_bundle(seed,condition,steps)
    weights = [w.copy() for w in bundle['initial']]
    opt = Optimizer(optimizer)
    first_batch = subset(bundle['train'],bundle['batches'][0])
    initial_direction,_ = Optimizer(optimizer).direction(forward(weights,first_batch,intervention,True)['grad'])
    k0 = np.array([np.linalg.norm(d)/math.sqrt(32) for d in initial_direction])
    assert np.all(k0>0)
    alpha = 0. if intervention == 'no_decay' else .1*eta
    runid = f'{tag}_{condition}_{optimizer}_eta{eta:g}_seed{seed}_{intervention}'
    direction_path = OUT/'directions'/f'{runid}.jsonl.gz'
    direction_path.parent.mkdir(parents=True,exist_ok=True)
    history = np.zeros((steps,2,32,32),dtype=np.float32)
    ab_history = np.full((steps,H,2),np.nan)
    frames,reference = None,None
    probes,tests,train = [],[],[]
    step_stats,clip_stats = [],[]
    probes.append({'step':0,**measure(weights,bundle['probe'],initial_direction,k0,intervention)})
    tests.append({'step':0,**measure(weights,bundle['test'],initial_direction,k0,intervention)})
    train.append({'step':0,'ce':forward(weights,bundle['train'],intervention)['loss']})
    with gzip.open(direction_path,'wt',encoding='utf-8',compresslevel=1) as output:
        for t in range(1,steps+1):
            data = subset(bundle['train'],bundle['batches'][t-1])
            loss_grad = forward(weights,data,intervention,True)
            directions,dropped = opt.direction(loss_grad['grad'])
            for w,d,k in zip(weights,directions,k0):
                w *= 1-alpha
                w -= eta*d/k
            events = np.zeros(H,bool)
            if intervention == 'qk_clip':
                fresh = forward(weights,data)['logits']
                pre = np.abs(fresh).max(axis=(0,2))
                events = pre>8
                scales = np.sqrt(np.minimum(1,8/(pre+1e-30)))
                for h,s in enumerate(scales):
                    for w in weights:
                        w[h*DH:(h+1)*DH] *= s
                post = np.abs(forward(weights,data)['logits']).max(axis=(0,2))
                assert np.max(post)<=8+1e-10
                clip_stats.append({'step':t,'pre_max_by_head':pre,'post_max_by_head':post,
                                   'triggered':events,'scale':scales})
            history[t-1] = directions
            output.write(json.dumps({'step':t,'D_Q':history[t-1,0].tolist(),
                                     'D_K':history[t-1,1].tolist()},separators=(',',':'))+'\n')
            step_stats.append({'step':t,'minibatch_ce_before_update':loss_grad['loss'],
                               'direction_fro':[float(np.linalg.norm(d)/k) for d,k in zip(directions,k0)],
                               'direction_op':[float(np.linalg.norm(d,2)/k) for d,k in zip(directions,k0)],
                               'dropped_singular_values':dropped})
            if t==50:
                frames = freeze(weights)
                reference = reference_at_freeze(weights,bundle['probe'],frames,intervention)
            if frames is not None:
                ab_history[t-1] = project(weights,frames)
            if t%10 == 0:
                probes.append({'step':t,**measure(weights,bundle['probe'],directions,k0,intervention,frames,reference)})
            if t%50 == 0:
                # Test mode residual is defined; changes are evaluated on probes only.
                tests.append({'step':t,**measure(weights,bundle['test'],directions,k0,intervention,frames)})
                train.append({'step':t,'ce':forward(weights,bundle['train'],intervention)['loss']})
                budget.check()
    xi = np.asarray([project([-d/k for d,k in zip(row,k0)],frames) for row in history])
    # Raw bounded polar coefficients allow checking the effective-step convention.
    xi_unscaled = xi*k0[None,None,:]
    diagnostics = analyze_modes(xi,ab_history,alpha,eta,seed,probes) if diagnose else {}
    result = {'run_id':runid,'seed':seed,'condition':condition,'optimizer':optimizer,
              'eta':eta,'alpha':alpha,'intervention':intervention,'steps':steps,
              'batch_size':32,'dtype':'float64','k0':k0,
              'effective_eta':eta/k0,'effective_lambda':alpha/(eta/k0),
              'data_rng':'SeedSequence(seed).spawn(3): data, initialization, minibatch',
              'directions_file':str(direction_path.relative_to(OUT)),
              'direction_storage_dtype':'float32 serialized exactly as JSON numbers',
              'initial_weights':bundle['initial'],'final_weights':weights,
              'frozen_frames':frames,'mode_ab_each_step':ab_history,
              'xi_scaled_each_step':xi,'xi_raw_polar_each_step':xi_unscaled,
              'probe':probes,'test':tests,'train':train,'steps_raw':step_stats,
              'clip':clip_stats,'diagnostics':diagnostics,
              'elapsed_seconds':time.monotonic()-started,'peak_RSS_bytes':rss_bytes()}
    save(OUT/'runs'/f'{runid}.json',result)
    return serial(result)


def gradient_checks():
    bundle = make_bundle(11,'isotropic',50)
    weights = [w.copy() for w in bundle['initial']]
    data = subset(bundle['train'],bundle['batches'][0])
    rows = []
    for intervention in ['base','layer_norm','head_norm']:
        g = forward(weights,data,intervention,True)['grad']
        maxerr,maxrel = 0.,0.
        for i in range(2):
            for ix in [(0,0),(13,7),(31,31)]:
                orig,eps = weights[i][ix],1e-6
                weights[i][ix] = orig+eps
                plus = forward(weights,data,intervention)['loss']
                weights[i][ix] = orig-eps
                minus = forward(weights,data,intervention)['loss']
                weights[i][ix] = orig
                est = (plus-minus)/(2*eps)
                maxerr = max(maxerr,abs(est-g[i][ix]))
                maxrel = max(maxrel,abs(est-g[i][ix])/(abs(est)+abs(g[i][ix])+1e-12))
        assert maxerr<1e-7,(intervention,maxerr)
        rows.append({'intervention':intervention,'coordinates':6,'max_absolute_error':maxerr,
                     'max_symmetric_relative_error':maxrel})
    # Independent autograd check of entire gradient arrays, only CPU float64.
    import torch
    for intervention in ['base','layer_norm','head_norm']:
        tw = [torch.tensor(w,dtype=torch.float64,requires_grad=True) for w in weights]
        tx,tk,ty = [torch.tensor(data[key],dtype=torch.float64) for key in ['x','keys','target']]
        q = (tx@tw[0].T).reshape(32,H,DH)
        k = (tk@tw[1].T).reshape(32,8,H,DH)
        if intervention == 'layer_norm':
            q = (q.reshape(32,32)/torch.sqrt((q.reshape(32,32)**2).mean(-1,keepdim=True)+1e-8)).reshape_as(q)
            k = (k.reshape(32,8,32)/torch.sqrt((k.reshape(32,8,32)**2).mean(-1,keepdim=True)+1e-8)).reshape_as(k)
        elif intervention == 'head_norm':
            q = q/torch.sqrt((q*q).mean(-1,keepdim=True)+1e-8)
            k = k/torch.sqrt((k*k).mean(-1,keepdim=True)+1e-8)
        logits = torch.einsum('bhd,bjhd->bhj',q,k)/math.sqrt(DH)
        loss = -(ty[:,None,:]*torch.log_softmax(logits,dim=-1)).sum()/(32*H)
        loss.backward()
        analytic = forward(weights,data,intervention,True)
        err = max(float(np.max(np.abs(w.grad.numpy()-g))) for w,g in zip(tw,analytic['grad']))
        assert err<1e-12,(intervention,err)
        rows.append({'intervention':intervention,'autograd_max_absolute_error':err,
                     'loss_absolute_error':abs(float(loss.detach())-analytic['loss']),
                     'device':'cpu'})
    save(OUT/'gradient_checks.json',rows)
    return rows


def constructions():
    rows = []
    for heads in [1,2,4,8]:
        q = np.zeros((heads,8)); q[0,0] = math.sqrt(heads*8)
        layer = float(np.max(np.sum(q*q,axis=1))/math.sqrt(8))
        q[0,0] = math.sqrt(8)
        head = float(np.max(np.sum(q*q,axis=1))/math.sqrt(8))
        rows.append({'heads':heads,'dh':8,'layer_max':layer,'head_max':head,
                     'layer_bound':heads*math.sqrt(8),'head_bound':math.sqrt(8)})
    a = np.array([.1,.3,.7]); b = np.array([.2,.4,.9]); p = np.array([.8,.15,.05])
    maxerr = 0.
    trace = []
    muon,sign = [a.copy(),b.copy()],[a.copy(),b.copy()]
    for t in range(100):
        ga = -p*muon[1]/(math.sqrt(3)*(1+np.exp(muon[0]*muon[1]/math.sqrt(3))))
        gb = -p*muon[0]/(math.sqrt(3)*(1+np.exp(muon[0]*muon[1]/math.sqrt(3))))
        for i,g in enumerate([ga,gb]):
            u,_,vt = np.linalg.svd(np.diag(g),full_matrices=False)
            polar = u@vt
            maxerr = max(maxerr,float(np.max(np.abs(polar-np.diag(np.sign(g))))))
            muon[i] = .996*muon[i]-.02*np.diag(polar)
            sign[i] = .996*sign[i]-.02*np.sign(g)
        trace.append({'step':t+1,'muon':np.array(muon),'sign':np.array(sign)})
    coherent = 2.5
    coherent_counts = 0
    for _ in range(100):
        z = .996*coherent+.02
        coherent_counts += z>2.5
        coherent = min(z,2.5)
    result = {'P6':rows,'diagonal_max_direction_error':maxerr,
              'diagonal_max_trajectory_error':float(np.max(np.abs(np.array(muon)-np.array(sign)))),
              'diagonal_trace':trace,'coherent_clip_triggers_out_of_100':coherent_counts}
    save(OUT/'constructions.json',result)
    return result
