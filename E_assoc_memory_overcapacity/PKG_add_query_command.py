from pathlib import Path
p=Path('supplement/reproduce.py');s=p.read_text().replace("'certificates'])","'certificates','query-tables'])")
s=s.replace("    a=p.parse_args();", "    p.add_argument('--replay-root',type=Path,default=Path('experiments_replay/reference/small_replay'))\n    a=p.parse_args();")
pos=s.index("    if a.task=='tables':")
s=s[:pos]+'''    if a.task=='query-tables':
        for design in ['matched','confirmation']:
            add('experiments_replay/measurements_from_queries.py','--input',a.replay_root/design,'--output',dest/(design+'_measurements.csv'))
        add('experiments_replay/recompute_tables.py','--matched-measurements',dest/'matched_measurements.csv','--confirmation-measurements',dest/'confirmation_measurements.csv','--output',dest/'tables')
'''+s[pos:]
p.write_text(s)
