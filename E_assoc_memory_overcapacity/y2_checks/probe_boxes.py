from certify import *
start=time.monotonic()
for ee in ['.1','.2','.5','1']:
 for aa in ['0','.25','.5']:
  a=F(aa);e=F(ee);vals,err=integ(a,e)
  s=stats(a,vals)
  print(aa,ee,{k:show(s[k])['display_only'] for k in ['N','J']},flush=True)
print('seconds',time.monotonic()-start,resources())
