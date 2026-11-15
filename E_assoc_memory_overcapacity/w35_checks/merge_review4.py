from pathlib import Path
root=Path.cwd()
p=root/'paper'
def edit(name,old,new):
    f=p/name;s=f.read_text();assert old in s,(name,old[:50]);f.write_text(s.replace(old,new))
f=p/'appendix/F.tex';s=f.read_text();s=s[:s.index('\\subsection{Locked external input')];s=s.replace('Reference stabilizers, external Muon inputs, and perturbations','Reference stabilizers, exact-polar variance, and perturbations');f.write_text(s+(root/'w35_checks/polar_proof_review4.tex').read_text())
f=p/'appendix/I_comparisons.tex';s=f.read_text();a=s.index('\\subsection{Numbered external');b=s.index('\\begin{corollary}',a);f.write_text(s[:a]+(root/'w35_checks/comparison_review4.tex').read_text()+s[b:])
f=p/'sections/main_results.tex';s=f.read_text();a=s.index('\\paragraph{Stabilizers and conditional');f.write_text(s[:a]+(root/'w35_checks/polar_statement_review4.tex').read_text())
edit('macros.tex',r'\newtheorem*{muoninput}{Assumption F.1 (external Muon input)}'+'\n'+r'\newtheorem*{muonlimit}{Assumption F.2 (additional deterministic variance limit)}'+'\n','')
edit('appendix/E.tex','None of these inputs uses F.1 or F.2.','The exact-polar proof is separate from these coordinate-method inputs.')
edit('main.tex',r'\icmlcorrespondingauthor{Anonymous Authors}{anonymous@example.com}'+'\n','')
edit('main.tex',r'\begin{document}',r'\icmlcorrespondingauthor{Anonymous Authors}{anonymous@example.com}'+'\n'+r'\begin{document}')
edit('sections/related_work.tex','Their Remark F.35 leaves a deterministic Muon variance limit unproved;\nour finite-load crossings retain that hypothesis.','We prove a deterministic exact-polar variance limit and a signed envelope,\nclosing the limiting-MSR stabilizer crossing for $\\gamma\\ge16$.')
edit('appendix/J_novelty.tex',"EXT's two corrections and FT's cross term; Muon crossings retain F.1/F.2", "EXT's two corrections, exact-polar variance and unique limiting-MSR stabilizer; FT's cross term and compact-uniform matching")
# Correct the source notes as well, preserving their historical numerical records.
f=root/'W35_STABILIZER.md';s=f.read_text().replace('浮点 enclosure τ_*∈[.52081,.65678]','浮点下界约 .52080753、上界约 .65677332（近似显示，非严格包围）');f.write_text(s)
f=root/'W35_MATCHING.md';s=f.read_text().replace('写 h_e(g)=g/(|g|+e)，A_e=E[G h_e(G)]、Q_e=E[h_e(G)²]，Sign 对应 e=0。','写 h_e(g)=g/(|g|+e)，A_e=E[G h_e(G)]、Q_e=E[h_e(G)²]，并定义 R_{3,e}=E[|G|²/(|G|+e)³]（e>0）；Sign 对应 e=0。');f.write_text(s)
f=p/'refs.bib';s=f.read_text();s+='''\n@misc{redelmeier2012,
  author = {Redelmeier, C. Emily I.},
  title = {Real Second-Order Freeness and the Asymptotic Real Second-Order Freeness of Several Real Matrix Models},
  year = {2012}, eprint = {1101.0422}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/1101.0422v3}
}
''';f.write_text(s)
