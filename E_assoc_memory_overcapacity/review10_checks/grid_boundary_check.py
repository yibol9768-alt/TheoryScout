"""A scalar exact-rational counterexample to the unrestricted grid claim."""
from fractions import Fraction
import json
from pathlib import Path

n = 2**27
i = n-1
x = i/n  # exact binary64
computed = 1-x*x
exact = 1-Fraction(i,n)**2
error = Fraction.from_float(computed)-exact
assert error == Fraction(1,2**54)
record = {'n':n, 'i':i, 'computed_p_hex':computed.hex(),
          'exact_p':str(exact), 'computed_minus_exact':str(error),
          'is_exact':False,
          'effect':'A claimed lower p endpoint can be rounded inward. This does not affect grid 4096.'}
Path(__file__).with_name('grid_boundary_result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
