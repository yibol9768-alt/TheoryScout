"""Integer outward interval certificate for the F6 four-step Adam matching roots.

Only the standard library is used. All proof decisions use integer endpoints,
never approximate displays. Run under the resource scope in W6_ADAM_ROOT.md.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from fractions import Fraction

P = 160
S = 1 << P

def rat(n, d=1):
    return (n*S//d, -((-n*S)//d))

Z = rat(0)
O = rat(1)

def add(a, b): return (a[0]+b[0], a[1]+b[1])
def neg(a): return (-a[1], -a[0])
def sub(a, b): return add(a, neg(b))
def hull(a, b): return (min(a[0], b[0]), max(a[1], b[1]))
def mul(a, b):
    v = [x*y for x in a for y in b]
    return (min(v)//S, -((-max(v))//S))
def sq(a):
    lo = 0 if a[0] <= 0 <= a[1] else min(a[0]**2, a[1]**2)//S
    return (lo, -((-max(a[0]**2, a[1]**2))//S))
def div(a, b):
    assert b[0] > 0, ('nonpositive denominator', b)
    return (min(x*S//y for x in a for y in b),
            max(-((-x*S)//y) for x in a for y in b))
def sqrt(a):
    assert a[0] >= 0, ('negative sqrt', a)
    lo = math.isqrt(a[0]*S)
    hi = math.isqrt(a[1]*S)
    return (lo, hi+(hi*hi < a[1]*S))
def abmax(a): return max(abs(a[0]), abs(a[1]))
def expneg(x):
    # exp(-x/256): alternating Taylor, even partial sum 60 above, 61 below.
    assert 0 <= x[0] <= x[1] <= 72*S
    y = div(x, rat(256))
    acc = O
    term = O
    for k in range(1, 62):
        term = div(mul(term, y), rat(k))
        acc = sub(acc, term) if k % 2 else add(acc, term)
        if k == 60: upper = acc[1]
    ans = (acc[0], upper)
    for _ in range(8): ans = sq(ans)
    return ans
def atan(x):
    term = x
    acc = Z
    for k in range(122):
        v = div(term, rat(2*k+1))
        acc = sub(acc, v) if k % 2 else add(acc, v)
        if k == 120: upper = acc[1]
        term = mul(term, sq(x))
    return (acc[0], upper)

PI = sub(mul(rat(16), atan(rat(1,5))), mul(rat(4), atan(rat(1,239))))
DEN = sqrt(div(rat(2), PI))
TAIL = rat(1, 1 << 60)
B1, B2 = rat(9,10), rat(99,100)
BC1 = [None]+[rat(10**t-9**t, 10**t) for t in range(1,5)]
BC2 = [None]+[rat(100**t-99**t, 100**t) for t in range(1,5)]

def density(z): return mul(DEN, expneg(div(sq(z), rat(2))))

def natural(a, e, z):
    """p=D/z and its a derivative; factors remove the apparent z=0 singularity."""
    p = pa = m = ma = v = va = Z
    for t in range(1,5):
        r = sub(p, O)
        m = add(mul(B1,m), mul(rat(1,10),r))
        ma = add(mul(B1,ma), mul(rat(1,10),pa))
        v = add(mul(B2,v), mul(rat(1,100),sq(r)))
        va = add(mul(B2,va), mul(rat(1,50),mul(r,pa)))
        mh, mah = div(m,BC1[t]), div(ma,BC1[t])
        root = sqrt(div(v,BC2[t]))
        den = add(mul(z,root),e)
        root_a = div(div(va,BC2[t]),mul(rat(2),root))
        dr = div(mh,den)
        dra = div(sub(mah,mul(dr,mul(z,root_a))),den)
        pa = sub(pa,add(dr,mul(a,dra)))
        p = sub(p,mul(a,dr))
    z2 = sq(z)
    return [mul(rat(2),mul(z2,mul(sub(p,O),pa))),
            mul(z2,pa), mul(rat(2),mul(z2,mul(p,pa)))]

def natural_integrals(a, e, cells=512):
    sums = [Z,Z,Z]
    for i in range(cells):
        z = (rat(12*i,cells)[0], rat(12*(i+1),cells)[1])
        den = (density((z[1],z[1]))[0], density((z[0],z[0]))[1])
        vals = natural(a,e,z)
        for k in range(3): sums[k] = add(sums[k],mul(mul(vals[k],den),rat(12,cells)))
    return [add(s,hull(neg(TAIL),TAIL)) for s in sums]

class Jet:
    """Interval Taylor coefficients f^(k)/k!, including interval input centers."""
    def __init__(self, c): self.c = c
    @staticmethod
    def const(x, n): return Jet([x]+[Z]*n)
    def __add__(self, other): return Jet([add(a,b) for a,b in zip(self.c,other.c)])
    def __neg__(self): return Jet([neg(a) for a in self.c])
    def __sub__(self, other): return self+-other
    def __mul__(self, other):
        return Jet([sum_iv([mul(self.c[j],other.c[k-j]) for j in range(k+1)])
                    for k in range(len(self.c))])
    def square(self):
        out = self*self
        out.c[0] = sq(self.c[0])
        return out
    def scale(self, x): return Jet([mul(a,x) for a in self.c])
    def __truediv__(self, other):
        c = []
        for k,a in enumerate(self.c):
            rem = sum_iv([mul(other.c[j],c[k-j]) for j in range(1,k+1)])
            c.append(div(sub(a,rem),other.c[0]))
        return Jet(c)
    def root(self):
        c = [sqrt(self.c[0])]
        for k in range(1,len(self.c)):
            rem = sum_iv([mul(c[j],c[k-j]) for j in range(1,k)])
            c.append(div(sub(self.c[k],rem),mul(rat(2),c[0])))
        return Jet(c)

def sum_iv(values):
    out = Z
    for v in values: out = add(out,v)
    return out

def jet_values(a,e,z0,n):
    z = Jet([z0,O]+[Z]*(n-1))
    one = Jet.const(O,n)
    p = m = v = Jet.const(Z,n)
    for t in range(1,5):
        r = p-one
        m = m.scale(B1)+r.scale(rat(1,10))
        v = v.scale(B2)+r.square().scale(rat(1,100))
        mh = m.scale(div(O,BC1[t]))
        vh = v.scale(div(O,BC2[t]))
        p = p-(mh/(z*vh.root()+Jet.const(e,n))).scale(a)
    # The Gaussian coefficient recurrence is valid also for an interval center.
    dc = [density(z0)]
    for k in range(1,n+1):
        prev = dc[k-2] if k >= 2 else Z
        dc.append(neg(div(add(mul(z0,dc[k-1]),prev),rat(k))))
    weight = z.square()*Jet(dc)
    return [(p-one).square()*weight, p*weight, p.square()*weight]

def validated_integrals(a,e,cells=192,order=10):
    sums = [Z,Z,Z]
    remainder = [Z,Z,Z]
    for i in range(cells):
        # Quadratic mesh resolves the short e=.1 length scale near zero.
        center = rat(6*(i*i+(i+1)**2),cells*cells)
        cell = (rat(12*i*i,cells*cells)[0],rat(12*(i+1)**2,cells*cells)[1])
        radius = rat(6*(2*i+1),cells*cells)
        powers = [O]
        for _ in range(order+1): powers.append(mul(powers[-1],radius))
        mid = jet_values(a,e,center,order-1)
        full = jet_values(a,e,cell,order)
        for j in range(3):
            integral = sum_iv([mul(mid[j].c[k],div(mul(rat(2),powers[k+1]),rat(k+1)))
                               for k in range(0,order,2)])
            err = mul((abmax(full[j].c[order]),)*2,
                      div(mul(rat(2),powers[order+1]),rat(order+1)))
            sums[j] = add(sums[j],add(integral,(-err[1],err[1])))
            remainder[j] = add(remainder[j],err)
    return [add(s,hull(neg(TAIL),TAIL) if j==1 else hull(Z,TAIL))
            for j,s in enumerate(sums)], remainder

def show(v):
    return {'lo':str(v[0]),'hi':str(v[1]),'power':P,
            'display_only':[v[0]/S,v[1]/S]}

INPUTS = ['ICML_REVIEW_SIM2.md','W35_MATCHING.md','W35_FIXES.md',
          'PROOF_REVIEW4.md','experiments/ft/FT_run.py',
          'experiments/ft/FT_forecasts.json','paper/figures/make_figures.py']

def hashes():
    return {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in INPUTS}

def primitive_checks():
    # Independently compare the outward arithmetic to exact Fraction endpoints.
    intervals = []
    for i in range(-4,5):
        for j in range(i,5):
            intervals.append((Fraction(i,7),Fraction(j,7)))
    def enc(x): return hull(rat(x[0].numerator,x[0].denominator),
                            rat(x[1].numerator,x[1].denominator))
    def contains(a,x): return Fraction(a[0],S)<=x<=Fraction(a[1],S)
    for x in intervals:
        a = enc(x)
        for xx in x: assert contains(sq(a),xx*xx)
        if x[0]<=0<=x[1]: assert contains(sq(a),Fraction(0))
        for y in intervals:
            b = enc(y)
            for xx in x:
                for yy in y:
                    assert contains(add(a,b),xx+yy)
                    assert contains(mul(a,b),xx*yy)
                    if y[0]>0: assert contains(div(a,b),xx/yy)
        if a[0]>=0:
            s = sqrt(a)
            assert s[0]*s[0] <= a[0]*S
            assert s[1]*s[1] >= a[1]*S
    # Exact weighted Cauchy certificate for the global direction bound |A_t|<2.
    weight_sums = []
    for t in range(1,5):
        aa = [Fraction(9,10)**(t-k)/sum(Fraction(9,10)**j for j in range(t))
              for k in range(1,t+1)]
        bb = [Fraction(99,100)**(t-k)/sum(Fraction(99,100)**j for j in range(t))
              for k in range(1,t+1)]
        c2 = sum(a*a/b for a,b in zip(aa,bb))
        assert c2 < 4
        weight_sums.append({'t':t,'direction_bound_squared':str(c2)})
    assert Fraction(2)*sum(Fraction(11,2)**j for j in range(4)) < 512
    return weight_sums

def as_iv_float(x):
    n,d = x.as_integer_ratio()
    return rat(n,d)

def exact_search(root, m, M):
    # Input nodes and arithmetic reproduce FT_run.py's binary64 linspace/midpoints.
    # Floats supply only exact dyadic inputs; every decision is made with integers.
    grid = [j*((.6/4)/32) for j in range(33)]
    grid[-1] = .6/4
    scan = []
    cache = []
    minimum_gap = None
    for x in grid:
        a = as_iv_float(x)
        if a[1]<root[0]: sign = 1; distance = sub(root,a)
        elif a[0]>root[1]: sign = -1; distance = sub(a,root)
        else: raise AssertionError(('root band insufficient for decision',x))
        gap = mul(m,distance)
        minimum_gap = gap if minimum_gap is None else (min(minimum_gap[0],gap[0]),min(minimum_gap[1],gap[1]))
        row = {'a':show(a),'sign':sign,'residual_abs_lower':show(gap)}
        scan.append(row); cache.append(x)
        if sign<0:
            lo,hi = grid[len(scan)-2],x
            break
    bracket = [as_iv_float(lo),as_iv_float(hi)]
    decisions = []
    for k in range(24):
        x = (lo+hi)/2
        a = as_iv_float(x)
        if a[1]<root[0]: sign = 1; distance = sub(root,a); lo=x
        elif a[0]>root[1]: sign = -1; distance = sub(a,root); hi=x
        else: raise AssertionError(('root band insufficient for midpoint',k,x))
        gap = mul(m,distance)
        minimum_gap = (min(minimum_gap[0],gap[0]),min(minimum_gap[1],gap[1]))
        decisions.append({'a':show(a),'sign':sign,'residual_abs_lower':show(gap)})
        cache.append(x)
    left,right = as_iv_float(lo),as_iv_float(hi)
    width = sub(right,left)
    assert width[1] < rat(28,10**11)[0]  # 2.8e-10, outward rational display bound.
    return {'scan':scan,'initial_bracket':[show(v) for v in bracket],
            'decisions':decisions,'minimum_decision_gap_lower':show(minimum_gap),
            'final_bracket':[show(left),show(right)],'final_width':show(width),
            'root_distance_upper':show((0,width[1])),
            'loss_residual_upper':show(mul(M,(0,width[1]))),
            'msr_error_upper_all_five_cases':show(mul(rat(400),(0,width[1])))}, (left,right)

def msrs(b0,v0,r):
    sig2 = mul(rat(2),sub(O,r))
    b,v = mul(sig2,b0),mul(sig2,v0)
    q = add(sub(O,b),v)
    return [div(sub(O,div(b,rat(2))),sqrt(q)),
            div(add(r,div(b,rat(2))),sqrt(q))]

def resource_info():
    out = {'executable':sys.executable,'python':sys.version,
           'maxrss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
           'thread_environment':{k:os.environ.get(k) for k in
               ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('Threads:'): out['os_threads'] = int(line.split()[1])
    cg = Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
    out['cgroup'] = str(cg)
    for name in ('memory.max','memory.swap.max','memory.peak','memory.events'):
        path = cg/name
        out[name] = path.read_text().strip() if path.exists() else None
    assert out['memory.max']=='2147483648' and out['memory.swap.max']=='0'
    assert out['os_threads']<=2
    return out

def certify(output,start):
    before = hashes()
    weights = primitive_checks()
    results = []
    for en,ed,root_n in [(1,10,4819026078294),(1,1,8287875576148)]:
        e = rat(en,ed)
        slabs = []
        global_derivatives = None
        for j in range(32):
            a = hull(rat(3*j,640),rat(3*(j+1),640))
            derivatives = natural_integrals(a,e,256)
            assert derivatives[0][0] > rat(-7)[1]
            assert derivatives[0][1] < rat(-2)[0]
            for d in derivatives[1:]: assert abmax(d)<4*S
            global_derivatives = derivatives if global_derivatives is None else [
                hull(a,b) for a,b in zip(global_derivatives,derivatives)]
            slabs.append({'a':show(a),'Fprime_Bprime_Vprime':[show(v) for v in derivatives]})
        print('monotonicity certified e=',en,'/',ed,
              [show(v)['display_only'] for v in global_derivatives],flush=True)
        root = hull(rat(root_n,10**14),rat(root_n+1,10**14))
        endpoints = []
        for n in (root_n,root_n+1):
            a = rat(n,10**14)
            vals,rem = validated_integrals(a,e)
            residual = sub(vals[0],rat(3,4))
            assert (residual[0]>0 if n==root_n else residual[1]<0)
            assert rem[0][1] < rat(1,10**15)[0]
            endpoints.append({'a':show(a),'F_B_V':[show(v) for v in vals],
                              'residual':show(residual),'Taylor_remainder':[show(v) for v in rem]})
        print('root endpoints certified e=',en,'/',ed,flush=True)
        search, final = exact_search(root,rat(2),rat(7))
        final_values = []
        for a in final:
            vals,rem = validated_integrals(a,e)
            final_values.append(vals)
        residuals = [sub(v[0],rat(3,4)) for v in final_values]
        assert residuals[0][0]>0 and residuals[1][1]<0
        absolute = [residuals[0],neg(residuals[1])]
        if absolute[0][1]<absolute[1][0]: chosen=0
        elif absolute[1][1]<absolute[0][0]: chosen=1
        else: raise AssertionError('best-cache gap not separated')
        gap = sub(absolute[1-chosen],absolute[chosen])
        assert gap[0]>0
        search['final_residuals'] = [show(v) for v in residuals]
        search['chosen_endpoint'] = chosen
        search['chosen_a'] = show(final[chosen])
        search['best_cache_gap_lower'] = show(gap)
        # A stronger strict gap against ALL earlier points follows from monotonicity:
        # each earlier point other than the final pair is at least one final width
        # beyond its same-side endpoint. Check this using the dyadic lattice.
        all_nodes = search['scan']+search['decisions']
        left,right = final
        extra_distances = []
        for row in all_nodes:
            a = (int(row['a']['lo']),int(row['a']['hi']))
            if a==left or a==right: continue
            d = sub(left,a) if a[1]<left[0] else sub(a,right)
            assert d[0]>0
            extra_distances.append(d)
        earlier_gap = min(mul(rat(2),d)[0] for d in extra_distances)
        search['all_cache_argmin_gap_lower'] = show((min(gap[0],earlier_gap),min(gap[0],earlier_gap)))
        # Enclose the moments at the root itself, using the whole root interval.
        vals,rem = validated_integrals(root,e)
        comparisons = []
        gd_k = sub(O,sqrt(rat(3,4)))
        for rn,rd in [(0,1),(1,2),(-1,2)]:
            r = rat(rn,rd)
            adam = msrs(vals[1],vals[2],r)
            gd = msrs(gd_k,sq(gd_k),r)
            gap_old = sub(gd[0],adam[0])
            assert gap_old[0]>rat(1,10000)[1]
            comparisons.append({'r':str(Fraction(rn,rd)),
                'adam_old_new':[show(v) for v in adam],
                'gd_old_new':[show(v) for v in gd],
                'gd_minus_adam_old':show(gap_old)})
        results.append({'e':str(Fraction(en,ed)),'beta1':'9/10','beta2':'99/100',
                        'slabs':slabs,'global_derivatives':[show(v) for v in global_derivatives],
                        'root':show(root),'root_endpoints':endpoints,'search24':search,
                        'moments_at_root':[show(v) for v in vals],
                        'root_moment_Taylor_remainder':[show(v) for v in rem],
                        'comparisons':comparisons})
        print('24-step decisions, cache argmin, MSR comparisons certified e=',en,'/',ed,flush=True)
    after = hashes()
    assert before==after
    info = resource_info()
    result = {'status':'PASS','arithmetic':'integer outward dyadic intervals',
              'power':P,'pi':show(PI),'half_normal_prefactor':show(DEN),
              'tail_bound_each_integral':show(TAIL),'direction_weights':weights,
              'natural_z_cells':256,'a_slabs':32,'z_truncation':12,
              'Taylor_order':10,'Taylor_quadratic_cells':192,'results':results,
              'inputs_before':before,'inputs_after':after,'inputs_unchanged':True,
              'certificate_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'seconds':time.monotonic()-start,'resources':info}
    with open(output,'x') as f: json.dump(result,f,indent=2,ensure_ascii=False)
    print('PASS',output,'seconds',result['seconds'],'resources',info,flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['probe','certify'], default='probe')
    parser.add_argument('--output', default='w6_checks/certificate.json')
    args = parser.parse_args()
    start = time.monotonic()
    if args.mode == 'certify':
        certify(args.output,start)
        return
    if args.mode == 'probe':
        for en,ed in [(1,10),(1,1)]:
            e = rat(en,ed)
            for an,ad in [(0,1),(48,1000),(83,1000),(145,1000)]:
                a = hull(rat(an,ad),add(rat(an,ad),rat(3,640)))
                val = natural_integrals(a,e,256)
                print('natural',en,ed,an,ad,[show(x)['display_only'] for x in val],flush=True)
            a = rat(48190260783,10**12) if ed==10 else rat(82878755761,10**12)
            vals,rem = validated_integrals(a,e,192,10)
            print('validated',en,ed,[show(x)['display_only'] for x in vals],
                  'remainder',[show(x)['display_only'] for x in rem],flush=True)
        print('seconds',time.monotonic()-start,flush=True)

if __name__ == '__main__': main()
