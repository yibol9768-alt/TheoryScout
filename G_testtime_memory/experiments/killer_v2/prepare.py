"""Deterministic predictions; run only after freezing, before smoke data."""
import json
import time
import numpy as np
from design import *
from common import OUT, begin, finish, prior_elapsed

def main():
    state = begin('prepare')
    assert not (OUT/'predictions.json').exists()
    predictions = []
    max_quadrature_difference = 0.
    for family in FAMILIES:
        for d in DIMS:
            for a in ALPHAS:
                for load in LOADS:
                    n = d*load
                    ids = query_ids(n)
                    masks = [bins(n,ids)==b for b in range(4)]
                    for method in METHODS:
                        m,Q,Q0 = feedback(d,load,a,family,method)
                        moment = {'m_bins':[float(m[x].mean()) for x in masks],
                                  'Q_bins':[float(Q[x].mean()) for x in masks]}
                        for readout in READOUTS:
                            p = value_dim(n,d,readout)
                            # 96-point quadrature is frozen; 64 points is an audit,
                            # never a reason to alter the predictor or its gates.
                            ap = recall(m,Q,n,p,96)
                            check = recall(m,Q,n,p,64)
                            diff = abs(float(ap.mean()-check.mean()))
                            max_quadrature_difference = max(max_quadrature_difference,diff)
                            row = {'family':family,'d':d,'load':load,'alpha':a,
                                   'method':method,'readout':readout,'p':p,
                                   'predicted':float(ap.mean()),'quadrature_delta':diff,
                                   'predicted_bins':[float(ap[x].mean()) for x in masks], **moment}
                            if method != 'hebb':
                                # Leave-out/no-own-write-feedback diagnostic only.
                                # m/sqrt(Q0) may exceed one: this diagnostic is
                                # invalid in that case, and is never clipped to fit.
                                row['no_feedback_valid'] = bool(np.all(m*m <= Q0))
                                row['no_feedback_Q_bins'] = [float(Q0[x].mean()) for x in masks]
                                row['no_feedback_predicted'] = (float(recall(m,Q0,n,p,96).mean())
                                                                if row['no_feedback_valid'] else None)
                            predictions.append(row)
                    print(f'predictions {family} d={d} a={a} load={load} elapsed={time.monotonic()-state[0]:.1f}s',flush=True)
                    assert time.monotonic()-state[0]+prior_elapsed('prepare') < 3300
    (OUT/'predictions.json').write_text(json.dumps(predictions))
    finish(state, prediction_rows=len(predictions), max_quadrature_difference=max_quadrature_difference)

if __name__ == '__main__': main()
