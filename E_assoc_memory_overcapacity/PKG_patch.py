from pathlib import Path
base=Path('supplement')
for file in [base/'two_step_adam_certificate'/n for n in ['integer_intervals.py','adam_integrals.py','history_reversal.py']]+[base/'polar_certificate/polar_variance.py']:
    s=file.read_text()
    s=s.replace("paths=sorted(base.rglob('*.py'))+sorted((base/'statements').glob('*'))", "folder=Path(__file__).resolve().parent\n    names=['integer_intervals.py','adam_integrals.py','stabilizer_ordering.py','matched_chain.py','moment_history.py','history_reversal.py'] if folder.name=='two_step_adam_certificate' else ['polar_variance.py']\n    paths=[folder/name for name in names]+sorted((base/'statements').glob('*'))")
    s=s.replace('e_numerator=num,e_denominator=den','e_numerator=e_num,e_denominator=e_den')
    file.write_text(s)
