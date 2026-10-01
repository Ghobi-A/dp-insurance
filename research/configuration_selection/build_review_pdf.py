"""Typeset the maintained manuscript with the official TMLR style as a draft."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    source = (root / 'paper/proxy_shift_manuscript.md').read_text()
    title, body = source.split('\n', 1)
    title = title.removeprefix('# ')
    body = body[body.index('## Abstract'):]
    # The editorial status belongs in the title header, not the abstract.
    abstract, remainder = body.removeprefix('## Abstract\n').split('## 1.', 1)
    replacements = {'10⁻⁵': r'$10^{-5}$', 'Aᵒ': r'$A^o$', 'Aᶜ': r'$A^c$',
                    'Aᵈ': r'$A^d$', 'α': r'$\alpha$', '≤': r'$\leq$ ', '≥': r'$\geq$ ',
                    '–': '-', '—': '-', '−': '-', 'ε': r'$\varepsilon$',
                    'δ': r'$\delta$', '≈': r'$\approx$', '’': "'", '“': '"', '”': '"'}
    abstract, remainder = abstract.strip(), '## 1.' + remainder
    for a, b in replacements.items():
        abstract, remainder, title = abstract.replace(a,b), remainder.replace(a,b), title.replace(a,b)
    remainder = remainder.replace('$50,000', r'\$50,000')
    remainder = re.sub(r'!\[([^]]*)\]\(([^)]+)\)',
                       lambda m: f'![{m[1]}]({(root / "paper" / m[2]).resolve()})', remainder)
    # Turn plain reference URLs into clickable links in the review copy.
    remainder = re.sub(r'(?m)^([ \t]*)(https://\S+)\s*$', r'\1<\2>', remainder)
    out = root / 'paper/submission'
    with tempfile.TemporaryDirectory(prefix='dp-review-pdf-') as directory:
        build = Path(directory)
        md = Path(directory) / 'body.md'
        md.write_text(remainder)
        abstract_path = Path(directory) / 'abstract.md'
        abstract_path.write_text(abstract)
        abstract_tex = subprocess.check_output(['pandoc', str(abstract_path), '-t', 'latex'], text=True)
        abstract_header = Path(directory) / 'abstract.tex'
        abstract_header.write_text(r'\AtBeginDocument{\maketitle\begin{abstract}'
                                   + abstract_tex + r'\end{abstract}}' + '\n')
        subprocess.run(['pandoc', str(md), '--standalone', '-t', 'latex',
                        '-V', 'documentclass=article', '-V', 'fontsize=10pt',
                        '-V', f'title={title}', '-V', 'author=Anonymous authors',
                        '--include-in-header', str(out/'header.tex'),
                        '--include-in-header', str(abstract_header),
                        '-o', str(build/'review_manuscript.tex')], check=True)
        shutil.copyfile(out/'tmlr.sty', build/'tmlr.sty')
        # Compile in isolation. A failed build must never truncate the review
        # PDF, leave corrupt auxiliaries, or reuse a stale partial aux file.
        for _ in range(2):
            result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                                     'review_manuscript.tex'], cwd=build, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout[-4000:])
        subprocess.run(['pdfinfo', str(build/'review_manuscript.pdf')], check=True,
                       stdout=subprocess.DEVNULL)
        tex = (build/'review_manuscript.tex').read_text().replace(str(root) + '/', '../../')
        tmp_tex = out/'review_manuscript.tex.tmp'
        tmp_tex.write_text(tex)
        tmp_tex.replace(out/'review_manuscript.tex')
        tmp_pdf = out/'review_manuscript.pdf.tmp'
        tmp_pdf.write_bytes((build/'review_manuscript.pdf').read_bytes())
        tmp_pdf.replace(out/'review_manuscript.pdf')
    for suffix in ['.aux', '.out', '.log']:
        (out/f'review_manuscript{suffix}').unlink(missing_ok=True)
    print(out/'review_manuscript.pdf')


if __name__ == '__main__':
    main()
