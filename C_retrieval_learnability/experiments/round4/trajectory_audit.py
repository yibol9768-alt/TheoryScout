"""Descriptive decomposition of observed endpoints and sampled last hits only."""
import experiment as c
from experiment import np,json

def main():
    rows=[]
    for path in sorted((c.OUT/'raw').glob('*.json')):
        r=json.loads(path.read_text()); g=np.array(r['final']['gamma']); z=np.array(r['final']['zeta'])
        hits=[h['step'] for h in r['history'] if h['all_sign']]
        final_rank=g>1e-10; final_sign=z>1e-10
        rows.append(dict(id=path.stem,**r['config'],rank_wrong=int((~final_rank).sum()),
                         sorted_but_threshold_wrong=int((final_rank&~final_sign).sum()),
                         first_all_sign=r['first']['all_sign'],last_sampled_all_sign=max(hits) if hits else None,
                         first_beta=r['history'][0]['beta'],final_beta=r['history'][-1]['beta']))
    c.save(c.OUT/'trajectory_audit.json',rows)
    print(json.dumps([r for r in rows if r['d']==7 and r['mode']=='b' and r['schedule']=='fixed']))
if __name__=='__main__': main()
