"""Cover [0,1/2] x [1/10,1] by exact rational rectangles."""
from certify import *
INPUTS=['ICML_REVIEW_SIM4.md','paper/sections/fine_tuning.tex','paper/appendix/K_ft_rank.tex','paper/appendix/G.tex','W7_FT_RANK.md','W6_ADAM_ROOT.md','w6_checks/adam_certificate.py']
def hashes():return {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in INPUTS}
def enc(x):return F(str(x))
def make_iv(lo,hi):return hull(enc(lo),enc(hi))
def readable(s):return {k:show(v) for k,v in s.items()}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='y2_checks/region.json');args=parser.parse_args()
    start=time.monotonic();before=hashes();leaves=[];attempts=0;maxdepth=0
    def cover(al,ar,el,er,depth=0):
        global attempts,maxdepth
        a=make_iv(al,ar);e=make_iv(el,er)
        vals,err=integ(a,e);s=stats(a,vals);attempts+=1;maxdepth=max(maxdepth,depth)
        if s['N'][0]>0 and s['J'][0]>0:
            leaves.append(dict(a=[str(al),str(ar)],e=[str(el),str(er)],integrals=[show(v) for v in vals],bounds={k:show(s[k]) for k in ('N','J')},remainders=[show(v) for v in err]))
            if len(leaves)%100==0:print('leaves',len(leaves),'attempts',attempts,'elapsed',time.monotonic()-start,flush=True)
            return
        assert depth<14,('unresolved',al,ar,el,er,readable(s))
        # Absolute a width 1/32 and relative e width about 1/100 have comparable wrapping.
        if (ar-al)*Fraction(1,2)>(er-el)/(el+Fraction(1,10)):
            am=(al+ar)/2;cover(al,am,el,er,depth+1);cover(am,ar,el,er,depth+1)
        else:
            em=(el+er)/2;cover(al,ar,el,em,depth+1);cover(al,ar,em,er,depth+1)
    for j in range(16):
        for k in range(32):cover(Fraction(j,32),Fraction(j+1,32),Fraction(1,10)+Fraction(9*k,320),Fraction(1,10)+Fraction(9*(k+1),320))
    print('derivative region complete',len(leaves),time.monotonic()-start,flush=True)
    # At a=1/2 every e first reaches all targets u<=1/2.
    edge=[]
    for k in range(64):
        el=Fraction(1,10)+Fraction(9*k,640);er=el+Fraction(9,640)
        vals,err=integ(F('1/2'),make_iv(el,er));s=stats(F('1/2'),vals)
        H5=sub(sq(s['B']),mul(rat(1,4),s['q']))
        assert s['B'][0]>0 and H5[0]>0,('edge fails',el,er,show(H5))
        edge.append(dict(e=[str(el),str(er)],integrals=[show(v) for v in vals],B=show(s['B']),H5=show(H5)))
    assert before==hashes()
    result=dict(status='PASS',domain={'a':['0','1/2'],'e':['1/10','1']},beta1='9/10',beta2='99/100',u_interval=['0 (excluded)','1/2'],power=P,tail=show(TAIL),cells=64,order=4,leaves=leaves,edge=edge,attempts=attempts,maxdepth=maxdepth,global_N=show((min(int(l['bounds']['N']['lo']) for l in leaves),max(int(l['bounds']['N']['hi']) for l in leaves))),global_J=show((min(int(l['bounds']['J']['lo']) for l in leaves),max(int(l['bounds']['J']['hi']) for l in leaves))),inputs_before=before,inputs_after=hashes(),source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['y2_checks/interval_core.py','y2_checks/certify.py',__file__]},seconds=time.monotonic()-start,resources=resources())
    with open(args.output,'x') as f:json.dump(result,f,indent=2)
    print('PASS',args.output,'N',result['global_N']['display_only'],'J',result['global_J']['display_only'],'seconds',result['seconds'],'resources',result['resources'],flush=True)
