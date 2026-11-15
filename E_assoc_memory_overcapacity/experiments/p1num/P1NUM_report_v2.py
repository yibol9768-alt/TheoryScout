"""Formatting-only correction: root report links to the root analysis note."""
from pathlib import Path
source=Path(__file__).with_name('P1NUM_report.py').read_text()
assert source.count('../../P1NUM_ANALYSIS_NOTE.md')==1
exec(compile(source.replace('../../P1NUM_ANALYSIS_NOTE.md','P1NUM_ANALYSIS_NOTE.md'),str(Path(__file__).with_name('P1NUM_report.py')),'exec'))
