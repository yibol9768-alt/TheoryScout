"""Auxiliary checks only; the analytic proof is Z1_G1_PROOF.md.

Run from the project root, with the command in Z1_RESULTS.md.
Does not import project code or call any API. Writes only a new z1_checks/Z1_*.json.
"""
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.integrate import quad
import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
OUT = (ROOT / (sys.argv[1] if len(sys.argv) > 1 else "z1_checks/Z1_output.json")).resolve()
assert OUT.parent == ROOT / "z1_checks" and OUT.name.startswith("Z1_") and OUT.suffix == ".json"
if OUT.exists():
    raise FileExistsError("Refusing to overwrite an existing output")
assert Path.cwd().resolve() == ROOT
assert sys.executable == "/home/liuyibo/venvs/theory/bin/python"

# Read system resource metadata, not any other project's files.
status = Path("/proc/self/status").read_text()
threads = int(next(s.split()[1] for s in status.splitlines() if s.startswith("Threads:")))
cgroup_rel = next(s.split(":", 2)[2] for s in Path("/proc/self/cgroup").read_text().splitlines()
                  if s.startswith("0::"))
cgroup_path = Path("/sys/fs/cgroup") / cgroup_rel.lstrip("/")
memory_max = (cgroup_path / "memory.max").read_text().strip()
swap_max = (cgroup_path / "memory.swap.max").read_text().strip()
assert memory_max == str(3 * 1024**3), memory_max
assert swap_max == "0", swap_max
assert threads <= 2, threads
result = {"resources": {"python": sys.executable, "threads": threads,
          "memory_max": memory_max, "swap_max": swap_max,
          "numpy": np.__version__, "scipy": scipy.__version__, "sympy": sp.__version__}}

# Exact symbolic coefficient checks using S=I+t[[0,1],[1,0]].
# This family has tr(E)=0, tr(E^2)/d=t^2, E_bb=0.
t, r, alpha, be, bd, Q, Qd, r0, r1 = sp.symbols("t r alpha beta betadot Q Qdot r0 r1", real=True)
sig2 = 2 * (1-r)
b = 3-2*r
S = sp.Matrix([[1,t],[t,1]])
K = S**3 - 2*r*S**2 + S
J = r*S-S**2
B = (be + bd*b*t*t)*sp.eye(2)
P = r1*S-r0*S**2
mean = sp.trace(r0*S+alpha*B*P)/2
mean_expected = alpha*((r1-r0)*b*bd-r0*be)
Mh = sp.Matrix([[Q+Qd*b*t*t, be*be*(2*sig2*t+t**3)],
                [be*be*(2*sig2*t+t**3), Q+Qd*b*t*t]])
energy = sp.trace(S*(S+alpha*(J*B+B*J)+alpha*alpha*Mh))/2
energy_expected = 1+2*alpha*(be*(r-3)+(r-1)*b*bd)+alpha**2*(2*sig2*be**2+b*Qd)
symchecks = {
    "K_expansion": sum((sp.expand(x) for x in K-(sig2*sp.eye(2)+2*sig2*(S-sp.eye(2))+b*(S-sp.eye(2))**2+(S-sp.eye(2))**3)), sp.Integer(0)),
    "correct_mean_coefficient": sp.expand(mean).coeff(t, 2)-mean_expected,
    "row_energy_coefficient": sp.expand(energy).coeff(t, 2)-energy_expected,
    "energy_constant": energy.subs(t,0)-(1-alpha*sig2*be+alpha**2*Q),
    "sign_old_mean": mean_expected.subs({r0:1,r1:r,bd:-be/(2*sig2)})+alpha*be*(1+2*r)/4,
    "sign_new_mean": mean_expected.subs({r0:r,r1:1,bd:-be/(2*sig2)})+alpha*be*(3+2*r)/4,
}
assert all(sp.simplify(v) == 0 for v in symchecks.values())
result["symbolic_residuals"] = {k:str(sp.simplify(v)) for k,v in symchecks.items()}

# Exact sphere fourth moments, independent of Gaussian trace formulas.
def sphere_moment(powers):
    if any(p % 2 for p in powers):
        return sp.Integer(0)
    total = sum(powers)//2
    numer = sp.Integer(1)
    for p in powers:
        numer *= sp.factorial2(p-1) if p else 1
    denom = sp.Integer(1)
    for j in range(total):
        denom *= len(powers)+2*j
    return numer/denom

v = sp.symbols("v0:4")
T = sp.Matrix([[2,3,-1,4],[-2,1,5,0],[1,-3,2,7],[4,1,0,-2]])/7
poly = sp.Poly(sp.expand((sp.Matrix(v).dot(T*sp.Matrix(v)))**2), *v)
sphere_direct = sum(coef*sphere_moment(powers) for powers,coef in poly.terms())
sphere_formula = (sp.trace(T)**2+sum(x*x for x in T)+sp.trace(T*T))/(4*6)
assert sp.simplify(sphere_direct-sphere_formula) == 0
result["sphere_fourth_moment_residual"] = str(sp.simplify(sphere_direct-sphere_formula))

rng = np.random.default_rng(2026110601)

# Finite data algebra; these checks are exact up to floating roundoff.
max_errors = {"all_candidate_energy": 0., "wrong_candidate_deletion": 0.,
              "trace_T": 0., "frobenius_T": 0., "conditional_contraction": 0.}
for trial in range(10):
    d, n = 3, 29
    rr = [-.5,0.,.5][trial % 3]
    sigma = math.sqrt(2*(1-rr))
    X, Y0, YE = [rng.normal(size=(d,n))/math.sqrt(d) for _ in range(3)]
    Y1 = rr*Y0+math.sqrt(1-rr*rr)*YE
    SS = d/n*X@X.T
    Z0, Z1 = [d/math.sqrt(n)*Y@X.T for Y in (Y0,Y1)]
    HH = Z1-Z0@SS
    KK = SS@SS@SS-2*rr*SS@SS+SS
    for method in ("gd","sign","adam"):
        aa = .37
        response = {"gd": lambda z:z, "sign":np.sign,
                    "adam":lambda z:z/(np.abs(z)+.8*sigma)}[method]
        UU = Z0+aa*response(HH)
        # Direct 1D density integrals, with fixed physical epsilon.
        betas = []
        for vv in np.diag(KK):
            val = quad(lambda g: 2*g*float(response(math.sqrt(vv)*g))*math.exp(-g*g/2)/math.sqrt(2*math.pi),
                       0, np.inf, epsabs=2e-12, epsrel=2e-12)[0]/math.sqrt(vv)
            betas.append(val)
        for kk, YY in enumerate((Y0,Y1)):
            CC = d/n*YY@YY.T
            TT = YY.T@UU@X
            correct = np.diag(TT)
            wrong = (np.sum(TT*TT, axis=0)-correct*correct)/(n-1)
            vC = np.trace(UU.T@CC@UU@SS)/d**2
            max_errors["all_candidate_energy"] = max(max_errors["all_candidate_energy"],abs(np.mean(TT*TT)-vC))
            max_errors["wrong_candidate_deletion"] = max(max_errors["wrong_candidate_deletion"],abs(np.mean(wrong)-(n*vC-np.mean(correct*correct))/(n-1)))
            ZZ = (Z0,Z1)[kk]
            max_errors["trace_T"] = max(max_errors["trace_T"],abs(np.trace(TT)-math.sqrt(n)/d*np.trace(ZZ.T@UU)))
            max_errors["frobenius_T"] = max(max_errors["frobenius_T"],abs(np.sum(TT*TT)-n*n/d**2*np.trace(UU.T@CC@UU@SS)))
            rk0,rk1 = ((1,rr),(rr,1))[kk]
            Ck = rk0*np.eye(d)+aa*np.diag(betas)@(rk1*np.eye(d)-rk0*SS)
            ej = Ck@X/math.sqrt(n)
            direct = d/n*sum(ej[:,j]@SS@ej[:,j] for j in range(n))
            trace = np.trace(SS@Ck@SS@Ck.T)/n
            max_errors["conditional_contraction"] = max(max_errors["conditional_contraction"],abs(direct-trace))
assert max(max_errors.values()) < 2e-11, max_errors
result["finite_algebra_max_abs_error"] = max_errors
result["finite_algebra_cases"] = 60

# Sign: exact arcsin covariance tests the scalar Hermite bound.
hermite_ratios = []
for rho in np.linspace(-.95,.95,39):
    if abs(rho) < 1e-8:
        continue
    rem = 2/math.pi*(math.asin(rho)-rho)
    ratio = abs(rem)/(abs(rho)**3)
    assert ratio <= 1+1e-13
    hermite_ratios.append(ratio)
result["sign_scalar_hermite_max_ratio"] = max(hermite_ratios)

# Multivariate even-block lemma for centered squared norms: cov=2||R||_F^2.
even_ratios = []
for _ in range(50):
    R = rng.normal(size=(2,2))
    R *= .9/max(1.,np.linalg.norm(R,2))
    exact_cov = 2*np.sum(R*R)
    bound = 4*np.linalg.norm(R,2)**2  # both L2 norms are sqrt(4)
    assert exact_cov <= bound+1e-12
    even_ratios.append(exact_cov/bound)
result["even_block_quadratic_max_ratio"] = max(even_ratios)

# Independent exact Sign/GD covariance on a 2x2 family; no fitted coefficients.
family_rows = []
rr = .3
sigma2 = 2*(1-rr)
aa = .37
for method in ("gd","sign"):
    beta0 = 1. if method == "gd" else math.sqrt(2/math.pi/sigma2)
    betadot = 0. if method == "gd" else -beta0/(2*sigma2)
    q0 = sigma2 if method == "gd" else 1.
    qdot = 1. if method == "gd" else 0.
    energy_c2 = 1+2*aa*(beta0*(rr-3)+(rr-1)*(3-2*rr)*betadot)+aa**2*(2*sigma2*beta0**2+(3-2*rr)*qdot)
    qpop = 1-aa*sigma2*beta0+aa**2*q0
    for tt in (.08,.04,.02,.01):
        SS = np.array([[1.,tt],[tt,1.]])
        KK = SS@SS@SS-2*rr*SS@SS+SS
        JJ = rr*SS-SS@SS
        BB = np.eye(2) if method == "gd" else np.diag(np.sqrt(2/math.pi/np.diag(KK)))
        MMh = KK if method == "gd" else 2/math.pi*np.arcsin(KK/np.sqrt(np.outer(np.diag(KK),np.diag(KK))))
        MU = SS+aa*(JJ@BB+BB@JJ)+aa**2*MMh
        energy_actual = np.trace(SS@MU)/2
        family_rows.append({"method":method,"t":tt,"energy_c2":energy_c2,
                            "scaled_residual_t4":(energy_actual-qpop-energy_c2*tt*tt)/tt**4})
    assert max(abs(row["scaled_residual_t4"]) for row in family_rows if row["method"] == method) < 100
result["two_by_two_exact_covariance_checks"] = family_rows

# Conditional row-energy variances, evaluated by Monte Carlo at fixed X.
# These are diagnostics, NOT verification of a dimension-free universal constant.
variance_rows = []
rr, aa, samples = .3, .37, 40000
for d in (8,16,32):
    n = d*d
    X = rng.normal(size=(d,n))
    SS = X@X.T/n
    eigval,eigvec = np.linalg.eigh(SS)
    sqrtS = (eigvec*np.sqrt(eigval))@eigvec.T
    g0,gE = [rng.normal(size=(samples,d)) for _ in range(2)]
    z0 = g0@sqrtS
    z1 = (rr*g0+math.sqrt(1-rr*rr)*gE)@sqrtS
    hh = z1-z0@SS
    eps = np.linalg.norm(SS-np.eye(d),2)
    for method in ("gd","sign","adam"):
        response = {"gd":lambda z:z, "sign":np.sign,
                    "adam":lambda z:z/(np.abs(z)+.8*math.sqrt(2*(1-rr)))}[method]
        uu = z0+aa*response(hh)
        R0 = np.sum(uu*uu,axis=1)/d
        R = np.sum((uu@SS)*uu,axis=1)/d
        var0,varR = np.var(R0,ddof=1),np.var(R,ddof=1)
        # Empirical version of Var(A+B) <= 2 Var(A)+2 E[B^2].
        deterministic_bound = 2*var0+2*np.mean((R-R0)**2)
        assert varR <= deterministic_bound*(1+1e-4)
        variance_rows.append({"d":d,"n":n,"method":method,"samples":samples,
                              "epsilon":float(eps),"var_R0":float(var0),"var_R":float(varR),
                              "d_times_var_R0":float(d*var0),
                              "var_R_over_1d_plus_epsilon2":float(varR/(1/d+eps*eps))})
result["conditional_variance_diagnostics"] = variance_rows

result["all_checks_passed"] = True
with OUT.open("x") as output_file:
    output_file.write(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"all_checks_passed":True,"symbolic_checks":len(symchecks),
                  "sphere_fourth_moment_residual":"0","finite_max_errors":max_errors,
                  "threads":threads,"memory_max":memory_max,"swap_max":swap_max,
                  "output":str(OUT.relative_to(ROOT))},ensure_ascii=False,indent=2))
