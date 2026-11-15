"""Check current active manuscript sources and the completed PDF build."""
from collections import Counter
from pathlib import Path
import re
import subprocess
import zipfile


def main():
    root = Path(__file__).resolve().parents[1] / "paper"
    visited = set()

    def expand(path):
        path = path.resolve()
        if not path.is_relative_to(root.resolve()):
            raise RuntimeError(f"Input outside paper: {path}")
        visited.add(path)
        source = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text())
        return re.sub(r"\\input\{([^}]+)\}",
                      lambda m: expand(root / (m[1]+".tex")), source)

    source = expand(root / "main.tex")
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    duplicates = [k for k, count in Counter(labels).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate labels: {duplicates}")
    refs = set(re.findall(r"\\(?:eqref|ref|pageref)\{([^}]+)\}", source))
    if refs-set(labels):
        raise RuntimeError(f"Missing references: {refs-set(labels)}")
    bib = set(re.findall(r"@\w+\{([^,]+),", (root / "refs.bib").read_text()))
    cited = set()
    for match in re.finditer(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}", source):
        cited.update(key.strip() for key in match[1].split(","))
    if cited-bib:
        raise RuntimeError(f"Missing citations: {cited-bib}")
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source):
        if not (root/name).is_file():
            raise RuntimeError(f"Missing graphic: {name}")
    with zipfile.ZipFile(root / "icml2026.zip") as archive:
        for name in ("icml2026.sty", "icml2026.bst", "fancyhdr.sty", "algorithm.sty", "algorithmic.sty"):
            if (root/name).read_bytes() != archive.read(name):
                raise RuntimeError(f"Modified style file: {name}")
    build = root / "build"
    aux = (build / "main.aux").read_text()
    def page(label):
        return int(re.search(r"\\newlabel\{"+re.escape(label)+r"\}\{\{[^}]*\}\{(\d+)\}", aux)[1])
    body = page("references-start")-1
    if body > 8:
        raise RuntimeError(f"Main text exceeds the current eight-page budget: {body}")
    log = (build / "main.log").read_text()
    if re.search(r"(?:LaTeX|Package natbib) Warning:[^\n]*(?:undefined|multiply defined)", log):
        raise RuntimeError("Unresolved or duplicate references/citations in build log")
    if "Label(s) may have changed" in log or "Overfull" in log:
        raise RuntimeError("Unconverged labels or overfull boxes")
    pdf = build / "main.pdf"
    if pdf.stat().st_mtime < max(p.stat().st_mtime for p in visited | {root / "refs.bib"}):
        raise RuntimeError("PDF predates manuscript sources")
    text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
    if "??" in text or "AUTHORERR" in text:
        raise RuntimeError("Unresolved references or author template errors in PDF")
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    pages = int(re.search(r"Pages:\s+(\d+)", info)[1])
    print(f"PASS {len(visited)} active TeX inputs; {len(labels)} unique labels; all references/citations/graphics resolve")
    print("PASS official style files unchanged; converged labels; no undefined references/citations or overfull boxes")
    print(f"PASS current PDF: {body} main-text pages, {pages} total pages")
    for label in ("eq:main-threshold-budget", "eq:main-path-budget", "sec:AC-threshold-diagnostic"):
        print(f"PASS {label}: page {page(label)}")
    print("Build/source validation only; this is not independent mathematical peer review.")


if __name__ == "__main__":
    main()
