"""Independent checks of the scientific statistics, before full execution."""
import time
import numpy as np
from util import HERE, save, max_acf, fit_beta, predict, rng, metadata, verify_prereg


def direct_maxacf(x):
    result=0.
    for lo in (50,150,250,350):
        z=x[lo:lo+100]-x[lo:lo+100].mean(0)
        den=(z*z).mean(0)
        for lag in range(1,21):
            numerator=(z[lag:]*z[:-lag]).mean(0)
            a=np.divide(numerator,den,out=np.zeros_like(den),where=den>0)
            result=max(result,float(np.abs(a).max()))
    return result


def main():
    start=time.monotonic();verify_prereg();gen=rng(100,0)
    examples=gen.normal(size=(7,450,8))
    examples[0,:,0]=0
    errors=[abs(float(max_acf(x))-direct_maxacf(x)) for x in examples]
    assert max(errors)<1e-12,errors
    g=gen.normal(size=(150,8));bound=np.arange(1,9)
    truth=np.arange(8)/10
    y=predict(g,truth,bound)
    fitted=fit_beta(g,y,bound)
    assert np.max(np.abs(truth-fitted))<1e-12
    # Vectorized bootstrap fit must equal individual refits exactly.
    ys=gen.normal(size=(5,150,8))
    vector=fit_beta(g,ys,bound)
    individual=np.asarray([fit_beta(g,z,bound) for z in ys])
    assert np.max(np.abs(vector-individual))<1e-12
    from calibrate import simulate
    m=predict(np.tile(g,(3,1)),truth,bound)
    vals=simulate(np.tile(g,(3,1)),m,bound,np.eye(8),gen,32,rho=.2)
    assert vals.shape==(32,) and np.isfinite(vals).all()
    save(HERE/'smoke'/'numerical_checks.json',dict(max_ACF_FFT_vs_direct_error=max(errors),
         constrained_LS_max_error=float(np.max(np.abs(truth-fitted))),
         vectorized_refit_error=float(np.max(np.abs(vector-individual))),
         simulation_shape=list(vals.shape),resources=metadata(start)))
    print('Independent ACF, constrained fit and bootstrap checks passed',flush=True)


if __name__=='__main__':
    main()
