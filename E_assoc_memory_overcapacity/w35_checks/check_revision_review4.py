from pathlib import Path
import re,json,subprocess,hashlib,zipfile
from collections import Counter
p=Path('paper');seen=set()
def expand(f):
 seen.add(f);s=re.sub(r'(?<!\\)%[^\n]*','',f.read_text())
 return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(p/(m[1]+'.tex')),s)
s=expand(p/'main.tex');labels=re.findall(r'\\label\{([^}]+)\}',s)
refs=re.findall(r'\\(?:eqref|ref|pageref)\{([^}]+)\}',s)
assert len(labels)==len(set(labels)),[k for k,v in Counter(labels).items() if v>1]
assert not set(refs)-set(labels),set(refs)-set(labels)
bib=set(re.findall(r'@\w+\{([^,]+),',(p/'refs.bib').read_text()));cit=set()
for m in re.finditer(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',s):cit.update(m[1].split(','))
assert not cit-bib,cit-bib
assert 'Assumption F.1' not in s and 'Assumption F.2' not in s
for f in seen:
 assert not re.search(r'\\TODO\{|% TODO|Source:|AUTHORERR',f.read_text()),f
for r in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',s):assert (p/r).is_file(),r
with zipfile.ZipFile(p/'icml2026.zip') as z:
 for name in ('icml2026.sty','icml2026.bst','fancyhdr.sty','algorithm.sty','algorithmic.sty'):
  assert (p/name).read_bytes()==z.read(name),name
cert=json.loads(Path('w35_checks/stabilizer_certificate_review4.json').read_text())
assert all(r['proposed_e_band_certified'] for r in cert['rows'])
assert cert['f5_sign_assertions_passed'] and cert['rows'][0]['tau_crossing']['certified']
print('PASS source references, labels, citations, graphics, official styles, theorem boundary and new interval certificate')
if (p/'build/main.aux').exists():
 aux=(p/'build/main.aux').read_text()
 page=lambda k:int(re.search(r'\\newlabel\{'+re.escape(k)+r'\}\{\{[^}]*\}\{(\d+)\}',aux)[1])
 body=page('references-start')-1
 assert body<=8,body
 log=(p/'build/main.log').read_text()
 assert not re.search(r'(?:LaTeX|Package natbib) Warning:[^\n]*(?:undefined|multiply defined)',log)
 assert 'Label(s) may have changed' not in log
 assert 'Overfull' not in log
 txt=subprocess.check_output(['pdftotext',str(p/'build/main.pdf'),'-'],text=True)
 assert '??' not in txt and 'AUTHORERR' not in txt
 assert 'Correspondence to: Anonymous Author' in txt
 pages={'main_text':body,'polar_theorem':page('thm:polar'),'uniform_theorem':page('thm:FT-uniform')}
 (p/'build/review4_pages.json').write_text(json.dumps(pages,indent=2)+'\n')
 print('PASS compiled PDF: '+json.dumps(pages)+'; undefined references/citations = 0; anonymous correspondence present')
 print(subprocess.check_output(['pdfinfo',str(p/'build/main.pdf')],text=True).split('Pages:')[1].splitlines()[0].strip()+' total PDF pages')
 for k in ('thm:polar','thm:FT-uniform','cor:FT-selected','thm:comparison'):print(k,'page',page(k))
# Hash only the current manuscript and new certificate inputs; historical manifests stay intact.
files=sorted(seen|{p/'refs.bib',Path('w35_checks/certify_stabilizer_review4.py'),Path('w35_checks/stabilizer_certificate_review4.json')})
(p/'revision_review4_manifest.sha256').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f)+'\n' for f in files))
