from pathlib import Path
p=Path('supplement/two_step_adam_certificate/check_adam.py');s=p.read_text()
s=s.replace("BASE=Path(__file__).resolve().parents[1]", "BASE=Path(__file__).resolve().parents[1]\nAUX_AUDIT=None")
s=s.replace("    return [x+iv.mpf([-tail.b,tail.b]) for x in answer]", "    result=[x+iv.mpf([-tail.b,tail.b]) for x in answer]\n    if AUX_AUDIT is not None:\n        AUX_AUDIT.write(json.dumps(dict(a=encode(a),e=encode(e),cells=cells,order=order,betas=list(betas),moment=moment,frozen=frozen,integrals=[encode(v) for v in result]))+'\\n')\n    return result")
s=s.replace("vv=reintegrate(a,e,192,10,betas);fresh=signals(a,vv);overlap(fresh['c0'],stored(row['c0']))", "vv=reintegrate(a,e,192,10,betas);fresh=signals(a,vv);overlap(fresh['c0'],stored(row['c0']));st=fresh")
s=s.replace("    all_integrals=args.reintegrate=='all'", "    global AUX_AUDIT\n    region_count=count\n    all_integrals=args.reintegrate=='all'\n    if all_integrals:AUX_AUDIT=args.output.with_name('independent_auxiliary_integral_envelopes.jsonl').open('x')")
s=s.replace("    audit.close()", "    audit.close()\n    if AUX_AUDIT is not None:AUX_AUDIT.close()")
s=s.replace('reintegrated_region_cells=min(count,1027)', 'reintegrated_region_cells=region_count')
p.write_text(s)
