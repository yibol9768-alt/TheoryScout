from pathlib import Path
p=Path('supplement/two_step_adam_certificate/check_adam.py');s=p.read_text()
s=s.replace("def encode(x):return [str(x.a),str(x.b)]","def encode(x):\n    return [dict(sign=v[0],mantissa=str(v[1]),exponent=v[2]) for v in x._mpi_]")
pos=s.index('def root_check(')
s=s[:pos]+'''def certify_rectangle(row, audit, index, depth=0):
    al,ar=map(F,row['a']);el,er=map(F,row['e'])
    a=box(al,ar);e=box(el,er);v=reintegrate(a,e,64,4);st=signals(a,v)
    if st['N'].a>0 and st['J'].a>0:
        audit.write(json.dumps(dict(parent=index,a=list(map(str,(al,ar))),e=list(map(str,(el,er))),
            integrals=[encode(x) for x in v],N=encode(st['N']),J=encode(st['J'])))+'\\n')
        return 1
    assert depth<8, (index,depth,str(st['N']),str(st['J']))
    # Exact bisection certifies a complete subcover when dependency widens a box.
    if (ar-al)*2 > (er-el)/(el+F('1/10')):
        am=(al+ar)/2;children=[dict(a=[str(al),str(am)],e=row['e']),dict(a=[str(am),str(ar)],e=row['e'])]
    else:
        em=(el+er)/2;children=[dict(a=row['a'],e=[str(el),str(em)]),dict(a=row['a'],e=[str(em),str(er)])]
    return sum(certify_rectangle(child,audit,index,depth+1) for child in children)

'''+s[pos:]
s=s.replace("count=0;durations=[]", "count=0;durations=[];subcells=0\n    audit=args.output.with_name('independent_integral_envelopes.jsonl').open('x')")
s=s.replace("then=time.monotonic();v=reintegrate(a,e,64,4);fresh=signals(a,v)\n            pos(fresh['N']);pos(fresh['J'])\n            for x,y in zip(v,row['integrals']):overlap(x,stored(y))\n            durations.append(time.monotonic()-then);count+=1", "then=time.monotonic();subcells+=certify_rectangle(row,audit,index)\n            durations.append(time.monotonic()-then);count+=1")
s=s.replace("sample_cell_seconds=durations[:2]", "independent_region_subcells=subcells,sample_cell_seconds=durations[:2]")
s=s.replace("    result=dict(status='PASS'", "    audit.close()\n    result=dict(status='PASS'")
p.write_text(s)
