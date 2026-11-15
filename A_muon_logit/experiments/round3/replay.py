"""Reconstruct saved trajectories, never train or advance using new gradients."""
import argparse
import gzip
import json
import time
import numpy as np
from util import (HERE, ROOT, DATA, make_bundle, forward, subset, SCALE, sha,
                  save, verify_prereg, metadata, check_time)


def directions(path):
    with gzip.open(path, 'rt') as f:
        rows = [json.loads(line) for line in f]
    assert [r['step'] for r in rows] == list(range(1,601))
    return np.asarray([[r['D_Q'],r['D_K']] for r in rows], dtype=np.float64)


def trajectory(d, dr):
    weights = np.empty((601,2,32,32),dtype=np.float64)
    weights[0] = d['initial_weights']
    for t in range(600):
        weights[t+1] = (1-d['alpha'])*weights[t] - d['eta']*dr[t]/np.asarray(d['k0'])[:,None,None]
    error = float(np.max(np.abs(weights[-1]-d['final_weights'])))
    assert error <= 1e-5, ('float32 replay final weight error',d['run_id'],error)
    return weights,error


def frames_at_150(weights, bundle, old):
    f = forward(weights,bundle['probe'])
    pairs,frames,x,y = [],[],[],[]
    for h in range(4):
        flat = np.argsort(-np.abs(f['logits'][:,h,:]).ravel(),kind='stable')[:4]
        pp = [tuple(map(int,np.unravel_index(v,(128,8)))) for v in flat]
        uu = []
        for i,j in pp:
            q,k = f['q'][i,h],f['k'][i,j,h]
            _,v = np.linalg.eigh(np.outer(q,q)+np.outer(k,k))
            u = v[:,-1]
            if u[np.argmax(np.abs(u))]<0:
                u = -u
            uu.append(u)
        if old is not None:
            assert pp == [tuple(p) for p in old['heads'][h]['pairs']]
            assert np.max(np.abs(np.asarray(uu)-old['heads'][h]['frames'])) < 1e-10
        pairs.append(pp); frames.append(uu)
        x.append([bundle['probe']['x'][i] for i,j in pp])
        y.append([bundle['probe']['keys'][i,j] for i,j in pp])
    return np.asarray(pairs),np.asarray(frames),np.asarray(x),np.asarray(y)


def project(matrices, frames, x, y):
    return np.stack([np.einsum('hcd,hde,hce->hc',frames,matrices[0].reshape(4,8,32),x),
                     np.einsum('hcd,hde,hce->hc',frames,matrices[1].reshape(4,8,32),y)],axis=-1)


def full_gradient_inputs(weights, bundle, frames, xproj, keyproj, k0):
    """Mean raw gradient and SD over 128 fixed contiguous 32-example blocks.

    Each sample is the actual 8-key smoothed CE gradient (including 1/H),
    not a binary margin surrogate. Minus sign makes g a descent projection.
    """
    data = bundle['train']
    f = forward(weights,data)
    logits = f['logits']
    e = np.exp(logits-logits.max(-1,keepdims=True))
    e /= e.sum(-1,keepdims=True)
    e -= data['target'][:,None,:]
    dq = np.einsum('nhj,njhd->nhd',e,f['k'])/(4*SCALE)
    sampleq = -np.einsum('nhd,hcd->nhc',dq,frames)*xproj/k0[0]
    qp = np.einsum('nhd,hcd->nhc',f['q'],frames)
    samplek = -qp*np.einsum('nhj,njhc->nhc',e,keyproj)/(4*SCALE*k0[1])
    samples = np.stack([sampleq,samplek],axis=-1).reshape(4096,-1)
    return samples.mean(0),samples.reshape(128,32,-1).mean(1).std(0,ddof=1),samples,f


def polar(g):
    u,s,vt = np.linalg.svd(g,full_matrices=False)
    keep = s>1e-12*s[0] if s[0]>0 else np.zeros(32,dtype=bool)
    return u[:,keep]@vt[keep]


def masking(g, frame, vector, head):
    u,s,vt = np.linalg.svd(g,full_matrices=False)
    r = np.zeros((32,32))
    r[head*8:(head+1)*8] = np.outer(frame,vector)
    contribution = (frame@u[head*8:(head+1)*8])*(vt@vector)
    j = int(np.argmax(np.abs(contribution)))
    weak = s[j]*np.outer(u[:,j],vt[j])
    mask = float(np.sum(np.abs(r)*(np.abs(g-weak)>np.abs(weak)))/np.sum(np.abs(r)))
    ratio = float(s[j]/s[0]) if s[0]>0 else 0.
    return dict(index=j,singular_ratio=ratio,mask_fraction=mask,
                selected=bool(ratio<=.1 and mask>=.9),
                score=mask*(-np.log(max(ratio,1e-15))))


def checkpoint_directions(w, gradients, adam, k0s, frames, x, y, d, step):
    raw = {'muon0':np.asarray([polar(g) for g in gradients]),
           'signsgd':np.sign(gradients),'adamw':adam}
    fnorm = np.linalg.norm(raw['muon0'],axis=(1,2))
    states = project(w,frames,x,y)
    q = np.einsum('hde,hce->hcd',w[0].reshape(4,8,32),x)
    k = np.einsum('hde,hce->hcd',w[1].reshape(4,8,32),y)
    ell = (q*k).sum(-1)/SCALE
    masks = [[(masking(gradients[0],frames[h,c],x[h,c],h),
               masking(gradients[1],frames[h,c],y[h,c],h)) for c in range(4)] for h in range(4)]
    result = []
    max_identity_error = 0.
    modes = {}
    for mode in ('original_k0','matched_fro'):
        byopt = {}
        for opt,dr in raw.items():
            scale = (1/np.asarray(k0s[opt]) if mode=='original_k0' else
                     fnorm/np.linalg.norm(dr,axis=(1,2)))
            delta = -dr*scale[:,None,None]
            xi = project(delta,frames,x,y)
            dq = np.einsum('hde,hce->hcd',delta[0].reshape(4,8,32),x)
            dk = np.einsum('hde,hce->hcd',delta[1].reshape(4,8,32),y)
            c = 1-d['alpha']; eta=d['eta']
            ell_next = c*c*ell + c*eta*((q*dk).sum(-1)+(k*dq).sum(-1))/SCALE + eta*eta*(dq*dk).sum(-1)/SCALE
            actualq = np.einsum('hde,hce->hcd',(c*w[0]+eta*delta[0]).reshape(4,8,32),x)
            actualk = np.einsum('hde,hce->hcd',(c*w[1]+eta*delta[1]).reshape(4,8,32),y)
            err = float(np.max(np.abs((actualq*actualk).sum(-1)/SCALE-ell_next)))
            max_identity_error = max(max_identity_error,err)
            byopt[opt] = dict(amplitude=np.linalg.norm(xi,axis=-1),xi=xi,
                             logit_next=ell_next,logit_delta=ell_next-ell,
                             abs_logit_delta=np.abs(ell_next)-np.abs(ell),
                             update_fro=np.linalg.norm(delta,axis=(1,2)))
        modes[mode] = byopt
    for h in range(4):
        for c in range(4):
            mq,mk = masks[h][c]
            row = dict(step=step,head=h,candidate=c,a=states[h,c,0],b=states[h,c,1],logit=ell[h,c],
                       masking_Q=mq,masking_K=mk,weak_masked=mq['selected'] and mk['selected'],
                       masking_score=(mq['score']+mk['score'])/2,comparisons={})
            for mode,byopt in modes.items():
                vals = {}
                for opt,v in byopt.items():
                    vals[opt] = {key:(vv[h,c] if key!='update_fro' else vv) for key,vv in v.items()}
                for other in ('signsgd','adamw'):
                    a0=vals['muon0']['amplitude']; a1=vals[other]['amplitude']
                    vals['ratio_'+other] = float(a0/a1) if a1>0 else None
                    vals['zero_'+other] = bool(a1==0)
                row['comparisons'][mode] = vals
            result.append(row)
    assert max_identity_error<=1e-10,max_identity_error
    return result,max_identity_error


def replay_one(path, out, index, spatial, primary=True, head_limit=None):
    start=time.monotonic()
    d=json.loads(path.read_text())
    bundle=make_bundle(d['seed'],d['condition'])
    assert np.array_equal(np.asarray(bundle['initial']),np.asarray(d['initial_weights']))
    drpath=DATA/d['directions_file']; dr=directions(drpath)
    weights,weight_error=trajectory(d,dr)
    old=spatial.get(d['run_id'])
    pairs,frames,x,y=frames_at_150(weights[150],bundle,old)
    if head_limit:
        # Smoke computes one head but still verifies the complete saved path.
        pairs,frames,x,y=pairs[:head_limit],frames[:head_limit],x[:head_limit],y[:head_limit]
    # Diagnostic formulae expect four heads; full raw reconstruction remains
    # small even in smoke. Restrict output/test family later, never reselect.
    if head_limit:
        pairs,frames,x,y=frames_at_150(weights[150],bundle,old)
    cum_actual=np.concatenate([[0.],np.cumsum(np.sqrt(np.sum(np.diff(weights,axis=0)**2,axis=(1,2,3))))])
    cum_direction=np.concatenate([[0.],np.cumsum(d['eta']*np.sqrt(np.sum((dr/np.asarray(d['k0'])[None,:,None,None])**2,axis=(1,2,3))))])
    dense=[]
    recorded={r['step']:r for r in d['probe']}
    probe_error=0.
    for t in range(601):
        if t<=50:
            dense.append(dict(step=t,ce=forward(weights[t],bundle['train'])['loss'],
                              Mabs=float(np.abs(forward(weights[t],bundle['probe'])['logits']).max()),
                              cumulative_actual_length=cum_actual[t],cumulative_direction_length=cum_direction[t]))
        if t%10==0:
            maxlog=float(np.abs(forward(weights[t],bundle['probe'])['logits']).max())
            probe_error=max(probe_error,abs(maxlog-recorded[t]['Mabs']))
    assert probe_error<=1e-4,(d['run_id'],probe_error)
    ce_error=max(abs(r['ce']-next(z['ce'] for z in d['train'] if z['step']==r['step'])) for r in dense if r['step'] in (0,50))
    assert ce_error<1e-5,ce_error
    for r in d['train']:
        if r['step']>50:
            t=r['step']
            dense.append(dict(step=t,ce=r['ce'],Mabs=recorded[t]['Mabs'],
                              cumulative_actual_length=cum_actual[t],cumulative_direction_length=cum_direction[t]))
    k0s={}
    for opt in ('muon0','signsgd','adamw'):
        p=DATA/'runs'/f'e1_{d["condition"]}_{opt}_eta{d["eta"]:g}_seed{d["seed"]}_base.json'
        k0s[opt]=json.loads(p.read_text())['k0']
    moments=np.zeros((2,32,32)); variances=moments.copy()
    checkpoints=[];identity_error=0.
    for t in range(451):
        # This advances diagnostic moments only. Weights always come from dr.
        grads=np.asarray(forward(weights[t],subset(bundle['train'],bundle['batches'][t]),grad=True)['grad'])
        moments=.9*moments+.1*grads; variances=.99*variances+.01*grads*grads
        if t in (150,300,450):
            adam=(moments/(1-.9**(t+1)))/(np.sqrt(variances/(1-.99**(t+1)))+1e-8)
            rows,err=checkpoint_directions(weights[t],grads,adam,k0s,frames,x,y,d,t)
            checkpoints.extend(rows);identity_error=max(identity_error,err)
    arrays={}
    recurrence_error=0.; bound_excess=0.; gradient_check_error=0.
    if primary:
        ab=np.asarray([project(w,frames,x,y) for w in weights[150:601]])
        xi=np.asarray([-project(d0/np.asarray(d['k0'])[:,None,None],frames,x,y) for d0 in dr[150:600]])
        recurrence_error=float(np.max(np.abs(ab[1:]-(1-d['alpha'])*ab[:-1]-d['eta']*xi)))
        assert recurrence_error<=1e-10,recurrence_error
        bound=np.stack([np.linalg.norm(x,axis=-1)/d['k0'][0],np.linalg.norm(y,axis=-1)/d['k0'][1]],axis=-1)
        bound_excess=float(np.max(np.abs(xi)-bound))
        assert bound_excess<1e-5,bound_excess
        g=[];noise=[];mini=[]
        xproj=np.einsum('nd,hcd->nhc',bundle['train']['x'],x)
        keyproj=np.einsum('njd,hcd->njhc',bundle['train']['keys'],y)
        for t in range(150,600):
            mean,sd,samples,f=full_gradient_inputs(weights[t],bundle,frames,xproj,keyproj,d['k0'])
            g.append(mean); noise.append(sd);mini.append(samples[bundle['batches'][t]].mean(0))
            if t==150:
                grads=np.asarray(forward(weights[t],bundle['train'],grad=True)['grad'])
                check=-project(grads/np.asarray(d['k0'])[:,None,None],frames,x,y).reshape(-1)
                gradient_check_error=float(np.max(np.abs(check-mean)))
                assert gradient_check_error<1e-12,gradient_check_error
            if t%50==0:
                print(f'{d["run_id"]} full-gradient step={t+1} elapsed={time.monotonic()-start:.1f}s',flush=True)
        # Save raw arrays for independent reanalysis; no future refitting.
        arrays=dict(ab=ab,xi=xi.reshape(450,32),g=np.asarray(g),noise_sd=np.asarray(noise),
                    minibatch_g=np.asarray(mini),bound=bound.reshape(32),pairs=pairs,frames=frames,
                    frozen_x=x,frozen_y=y)
        if head_limit:
            for name in ('xi','g','noise_sd','minibatch_g'):
                arrays[name]=arrays[name][:,:8]
            arrays['bound']=arrays['bound'][:8]
            arrays['ab']=arrays['ab'][:,:1]
        np.savez_compressed(out/'arrays'/f'{d["run_id"]}.npz',**arrays)
    report=dict(run_id=d['run_id'],index=index,seed=d['seed'],condition=d['condition'],eta=d['eta'],optimizer=d['optimizer'],
                primary=primary,k0=d['k0'],alpha=d['alpha'],pairs=pairs,frames=frames,
                dense_CE=dense,checkpoints=checkpoints,
                same_step=[dict(**recorded[t],cumulative_actual_length=cum_actual[t],
                                cumulative_direction_length=cum_direction[t]) for t in (150,300,450)],
                validation=dict(final_weight_max_error=weight_error,probe_Mabs_max_error=probe_error,
                                dense_CE_replay_error=ce_error,scalar_recurrence_error=recurrence_error,
                                max_xi_bound_excess=bound_excess,full_gradient_projection_error=gradient_check_error,
                                logit_identity_error=identity_error),
                source_sha256={str(path.relative_to(ROOT)):sha(path),str(drpath.relative_to(ROOT)):sha(drpath)},
                resources=metadata(start))
    save(out/'replays'/f'{d["run_id"]}.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    verify_prereg(); start=time.monotonic()
    out=HERE/('smoke' if args.smoke else 'full')
    (out/'arrays').mkdir(parents=True,exist_ok=True)
    paths=sorted((DATA/'runs').glob('e1_*_base.json'))
    primary=[p for p in paths if '_muon0_' in p.name]
    controls=[p for p in paths if '_muon0_' not in p.name]
    assert len(primary)==18 and len(controls)==54
    spatial={r['run_id']:r for r in json.loads((DATA/'v2_offline/diagnostics.json').read_text())['runs']}
    sequence=primary[:1] if args.smoke else primary+controls
    rows=[]
    for index,path in enumerate(sequence):
        result=replay_one(path,out,index,spatial,path in primary,1 if args.smoke else None)
        rows.append(dict(run_id=result['run_id'],validation=result['validation'],resources=result['resources']))
        save(out/'replay_manifest.json',dict(completed=rows,resources=metadata(start),script_sha256=sha(__file__)))
        print(f'COMPLETED {len(rows)}/{len(sequence)} {path.stem} total={time.monotonic()-start:.1f}s',flush=True)
        check_time(start)


if __name__=='__main__':
    main()
