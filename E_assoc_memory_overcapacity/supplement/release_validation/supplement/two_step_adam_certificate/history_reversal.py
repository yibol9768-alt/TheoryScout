"""Supplement to the analytic proof: integer intervals, no floating decisions.

Uses fixed-point integer outward arithmetic.
The proof of the single sign transition does not use this computation.
"""
from integer_intervals import *


def sources():
    base=Path(__file__).resolve().parents[1]
    folder=Path(__file__).resolve().parent
    names=['integer_intervals.py','adam_integrals.py','stabilizer_ordering.py','matched_chain.py','moment_history.py','history_reversal.py'] if folder.name=='two_step_adam_certificate' else ['polar_variance.py']
    paths=[folder/name for name in names]+sorted((base/'statements').glob('*'))
    return {str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def resources():
    return dict(maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                thread_limits={k:os.environ.get(k) for k in
                  ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})

def F(x):
    f = Fraction(str(x))
    return rat(f.numerator, f.denominator)


def enc(x):
    return dict(lo=str(x[0]), hi=str(x[1]), power=P,
                display_only=[x[0]/S, x[1]/S])


def gaussian_jet(z0, n):
    d = [density(z0)]
    for k in range(1, n+1):
        d.append(neg(div(add(mul(z0, d[k-1]), d[k-2] if k >= 2 else Z), rat(k))))
    return Jet(d)


def moment_jets(z0, n, e):
    z = Jet([z0, O]+[Z]*(n-1))
    den = z+Jet.const(e, n)
    value = z.square()*gaussian_jet(z0, n)
    out = []
    for _ in range(4):
        value = value/den
        out.append(value)
    return out


class AD:
    """Exact first a derivative of a Taylor jet in z."""
    def __init__(self, v, a=None):
        self.v = v
        self.a = Jet.const(Z, len(v.c)-1) if a is None else a
    def __add__(self, other): return AD(self.v+other.v, self.a+other.a)
    def __neg__(self): return AD(-self.v, -self.a)
    def __sub__(self, other): return self+-other
    def __mul__(self, other): return AD(self.v*other.v, self.a*other.v+self.v*other.a)
    def __truediv__(self, other):
        v = self.v/other.v
        return AD(v, (self.a-v*other.a)/other.v)
    def square(self): return AD(self.v.square(), (self.v*self.a).scale(rat(2)))
    def root(self):
        v = self.v.root()
        return AD(v, self.a/v.scale(rat(2)))
    def scale(self, x): return AD(self.v.scale(x), self.a.scale(x))


def endpoint_jets(z0, n, a, e, b1, b2):
    def C(x): return AD(Jet.const(x, n))
    z = AD(Jet([z0, O]+[Z]*(n-1)))
    aa = C(a)
    aa.a = Jet.const(O, n)
    t = aa/(z+C(e))
    f = C(O)-t
    L = ((f.square()+C(b2)).scale(div(O, add(O, b2)))).root()
    pp = C(O)/(z+C(e))+(C(O)-t.scale(div(O, add(O, b1))))/(z*L+C(e))
    weight = z.v.square()*gaussian_jet(z0, n)
    pp2 = pp.square()
    return [v*weight for v in (pp.v, pp2.v, pp.a, pp2.a)]


def integrate(jets, cells=128, order=10, count=4):
    out = [Z]*count
    errors = [Z]*count
    for i in range(cells):
        center = rat(6*(i*i+(i+1)**2), cells*cells)
        whole = (rat(12*i*i, cells*cells)[0], rat(12*(i+1)**2, cells*cells)[1])
        h = rat(6*(2*i+1), cells*cells)
        powers = [O]
        for _ in range(order+1): powers.append(mul(powers[-1], h))
        mid, full = jets(center, order-1), jets(whole, order)
        for j in range(count):
            value = sum_iv([mul(mid[j].c[k], div(mul(rat(2), powers[k+1]), rat(k+1)))
                            for k in range(0, order, 2)])
            err = mul((abmax(full[j].c[order]),)*2,
                      div(mul(rat(2), powers[order+1]), rat(order+1)))
            out[j] = add(out[j], add(value, (-err[1], err[1])))
            errors[j] = add(errors[j], err)
    # All uses have e>=1/2 and 0<=a/e<=1/2. See MD for the tail bound.
    tail = rat(1, 1 << 60)
    return [add(v, (-tail[1], tail[1])) for v in out], errors


def endpoint(a, e, b1, b2, r, cells=128, order=10):
    vals, err = integrate(lambda z, n: endpoint_jets(z, n, a, e, b1, b2), cells, order)
    A, Q, Aa, Qa = vals
    B, V = mul(a, A), mul(sq(a), Q)
    Ba = add(A, mul(a, Aa))
    Va = add(mul(mul(rat(2), a), Q), mul(sq(a), Qa))
    s = sub(O, r)
    q = add(O, mul(mul(rat(2), s), sub(V, B)))
    newnum, oldnum = add(r, mul(s, B)), sub(O, mul(s, B))
    N = sub(mul(Ba, add(q, newnum)), mul(newnum, Va))
    return dict(B=B, V=V, q=q, N=N, newnum=newnum,
                c0=div(oldnum, sqrt(q))), vals, err


def point_comparison(e_text, r_text, u_text):
    e, r, u = map(F, (e_text, r_text, u_text))
    # Obtain root hints with a float-free bisection driven exclusively by intervals.
    rows = []
    for b2 in (Z, F('99/100')):
        b1 = F('9/10')
        al, ar = Fraction(0), Fraction(e_text)/2
        for _ in range(31):
            am = (al+ar)/2
            st, _, _ = endpoint(F(am), e, b1, b2, r, 96, 8)
            H = sub(sq(st['newnum']), mul(sq(u), st['q']))
            assert st['newnum'][0] > 0
            if H[1] < 0: al = am
            elif H[0] > 0: ar = am
            else: raise AssertionError(('ambiguous bisection', e_text, str(am), enc(H)))
        # The final root interval includes the first root: certify derivative on
        # the entire earlier amplitude range, rather than just near the root.
        monotonic = []
        for j in range(20):
            ai = (F(ar*j/20)[0], F(ar*(j+1)/20)[1])
            st, raw, err = endpoint(ai, e, b1, b2, r, 64, 4)
            assert st['N'][0] > 0, ('first-root check', e_text, j, enc(st['N']))
            monotonic.append(dict(a=[str(ar*j/20), str(ar*(j+1)/20)], N=enc(st['N']), integrals=[enc(v) for v in raw], errors=[enc(v) for v in err]))
        ends = []
        for x, sign in ((al, -1), (ar, 1)):
            st, _, _ = endpoint(F(x), e, b1, b2, r)
            H = sub(sq(st['newnum']), mul(sq(u), st['q']))
            assert st['newnum'][0] > 0 and (H[1] < 0 if sign < 0 else H[0] > 0)
            ends.append(dict(H=enc(H), integrals=[enc(v) for v in endpoint(F(x), e, b1, b2, r)[1]]))
        ai = (F(al)[0], F(ar)[1])
        st, vals, err = endpoint(ai, e, b1, b2, r)
        rows.append(dict(beta1='9/10', beta2='0' if b2 == Z else '99/100',
                         root=[str(al), str(ar)], c0=enc(st['c0']),
                         endpoints_H=ends, monotonic=monotonic,
                         root_integrals=[enc(x) for x in vals], errors=[enc(x) for x in err]))
    gap = sub((int(rows[1]['c0']['lo']), int(rows[1]['c0']['hi'])),
              (int(rows[0]['c0']['lo']), int(rows[0]['c0']['hi'])))
    assert gap[0] > 0 if e_text == '1/2' else gap[1] < 0
    return dict(e=e_text, r=r_text, u=u_text, rows=rows, history_gap=enc(gap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='two_step_adam_certificate/history_reversal.json')
    args = parser.parse_args()
    resources()
    start = time.monotonic()
    hashes = sources
    before = hashes()
    threshold = []
    for text_e, sign in [('21/20', 1), ('11/10', -1)]:
        vals, err = integrate(lambda z, n: moment_jets(z, n, F(text_e)))
        M1, M2, M3, M4 = vals
        D = sub(mul(M1, M3), sq(M2))
        T = sub(D, mul(F(text_e), sub(mul(M1, M4), mul(M2, M3))))
        assert T[0] > 0 if sign > 0 else T[1] < 0
        threshold.append(dict(e=text_e, moments=[enc(x) for x in vals],
                              errors=[enc(x) for x in err], D=enc(D), T=enc(T)))
        print('threshold', text_e, enc(T)['display_only'], flush=True)
    examples = [point_comparison(e, '1/4', '1/2') for e in ('1/2', '2')]
    after = hashes()
    assert before == after, {p: (before[p], after[p]) for p in before if before[p] != after[p]}
    result = dict(status='PASS', threshold=threshold, examples=examples,
                  source_hashes=before, hashes_unchanged=True, resources=resources(),
                  seconds=time.monotonic()-start)
    with open(args.output, 'x') as out: json.dump(result, out, indent=2)
    print('PASS', args.output, 'seconds', result['seconds'], result['resources'], flush=True)
    for example in examples:
        print('example', example['e'], 'c0', [row['c0']['display_only'] for row in example['rows']],
              'gap', example['history_gap']['display_only'], flush=True)
