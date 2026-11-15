"""Post-freeze static/document audit; no new samples, fits, or gate changes."""
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/frontier'

class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts=[]
        self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ['script','style']:
            self.skip+=1
    def handle_endtag(self,tag):
        if tag in ['script','style']:
            self.skip-=1
        if tag in ['p','li','h1','h2','h3','h4','h5','h6','table','section']:
            self.parts.append('\n')
    def handle_data(self,data):
        if not self.skip:
            self.parts.append(data)

def main():
    start=time.monotonic()
    result={'purpose':'static audit only; no resampling, refitting, or relaxed gates','lemmas':{},'imports':{},'papers':[]}
    subprocess.run(['sha256sum','-c',str(BASE/'FROZEN.sha256')],cwd=ROOT,check=True,
                   stdout=(BASE/'final_freeze_check.txt').open('w'))
    for name in ['run.py','analyze.py']:
        tree=ast.parse((BASE/name).read_text())
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                imports.extend(x.name for x in node.names)
            if isinstance(node,ast.ImportFrom):
                imports.append(node.module)
        result['imports'][name]=imports
    for path in sorted((ROOT/'lemmas_v6').glob('L*.tex')):
        s=path.read_text()
        braces=0
        minimum=0
        for token in re.findall(r'(?<!\\)[{}]',s):
            braces+=1 if token=='{' else -1
            minimum=min(minimum,braces)
        envs=re.findall(r'\\(begin|end)\{([^}]+)\}',s)
        stack=[]
        for action,env in envs:
            if action=='begin':
                stack.append(env)
            else:
                assert stack.pop()==env
        ok=braces==0 and minimum>=0 and not stack and s.count('$')%2==0
        assert ok,path
        result['lemmas'][path.name]={'static_balanced':ok,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    for path in sorted((BASE/'papers').glob('*.html')):
        parser=Text()
        parser.feed(path.read_text())
        text='\n'.join(re.sub(r'\s+',' ',line).strip() for line in ''.join(parser.parts).splitlines() if line.strip())+'\n'
        path.with_suffix('.txt').write_text(text)
        result['papers'].append(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),text_lines=len(text.splitlines())))
    tex=BASE/'tmp'/'lemmas_v6.tex'
    tex.write_text('\\documentclass{article}\n\\usepackage{amsmath,amssymb,amsthm}\n\\newtheorem{problem}{Problem}\n\\begin{document}\n'+
                   '\n'.join(p.read_text() for p in sorted((ROOT/'lemmas_v6').glob('L*.tex')))+'\n\\end{document}\n')
    engine=shutil.which('pdflatex')
    result['tex_engine']=engine
    if engine:
        proc=subprocess.run([engine,'-interaction=nonstopmode','-halt-on-error','-output-directory',str(BASE/'tmp'),str(tex)],
                            cwd=ROOT,capture_output=True,text=True,timeout=60)
        (BASE/'tex_compile.log').write_text(proc.stdout+proc.stderr)
        result['tex_compiled']=proc.returncode==0
        assert proc.returncode==0
    else:
        result['tex_compiled']=None
    result.update(seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (BASE/'static_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'lemmas':len(result['lemmas']),'papers':len(result['papers']),'tex_compiled':result['tex_compiled']}))

if __name__=='__main__':
    main()
