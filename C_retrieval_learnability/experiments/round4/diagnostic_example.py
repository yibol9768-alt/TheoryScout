"""Extract one observed finite-beta counterexample from existing fixed-geometry output."""
import experiment as c
from experiment import np,json

def main():
    A,_=c.data(7,15)
    name='d7_s15_nce_16'
    with np.load(c.OUT/'fixed_geometry.npz') as z:
        gamma=z[name+'_gamma']; zstar=z[name+'_star_zeta']; bias=z[name+'_star_b']
    rows=np.flatnonzero((gamma>0)&(zstar<=0))
    row=int(rows[np.argmax(gamma[rows])])
    with np.load(c.ROOT/'experiments/round3/weights/d7_s15_nce.npz') as z:
        scores=(z['U']@z['V'].T)[row]
    low=float(scores[A[row]].min()); high=float(scores[~A[row]].max()); mid=(low+high)/2
    y=2.*A[row]-1
    midloss=float(np.logaddexp(0,-y*16*(scores-mid)).mean())
    starloss=float(np.logaddexp(0,-y*(16*scores+bias[row])).mean())
    example=dict(source='experiments/round3/weights/d7_s15_nce.npz',row=row,beta=16.,positive_indices=np.flatnonzero(A[row]).tolist(),
        scores=scores.tolist(),gamma=float(gamma[row]),theta_mid=mid,theta_star=float(-bias[row]/16),
        midpoint_margin=float(gamma[row]/2),logistic_star_margin=float(zstar[row]),
        midpoint_row_loss=midloss,logistic_star_row_loss=starloss)
    c.save(c.OUT/'diagnostic_example.json',example); print(json.dumps(example))
if __name__=='__main__': main()
