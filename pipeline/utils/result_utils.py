"""LaTeX tables and figure style for the result notebooks."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

from pipeline.utils.paper_style import RC

_LEAD0 = re.compile(r"(?<![\d.])([+-]?)0\.(\d+)")


def nz(s):
    """Drop the leading zero of every number in `s` (0.123 becomes .123); an exact zero becomes 0."""
    return _LEAD0.sub(lambda m: "0" if set(m[2]) == {"0"} else f"{m[1]}.{m[2]}", str(s))


def style(seaborn: bool = False):
    """Reset matplotlib to `RC`, optionally on top of the seaborn whitegrid theme."""
    plt.rcdefaults()
    if seaborn:
        sns.set_theme(style="whitegrid")
    plt.rcParams.update(RC)


def mark(sig):
    r"""Significance marker: `\lstar` if significant, `\lmins` if untested (None), else nothing."""
    return r"\lmins" if sig is None else r"\lstar" if sig else ""


def float_table(caption, label, colspec, head, body):
    """A booktabs `table` float; `head` and `body` are lists of LaTeX rows."""
    return "\n".join([r"\begin{table}", r"\centering\scriptsize", rf"\caption{{{caption}}}",
                      rf"\label{{{label}}}",
                      rf"\begin{{tabular*}}{{\columnwidth}}{{@{{\extracolsep{{\fill}}}}{colspec}}}",
                      r"\toprule", *head, r"\midrule", *body, r"\bottomrule",
                      r"\end{tabular*}", r"\end{table}"])


MACROS = Path(__file__).with_name("macros.tex")

# 17cm: the text width of the supplemental material.
STANDALONE = r"""\documentclass[border=6pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,booktabs,multirow,pifont,dsfont,caption,xcolor,algorithm,algpseudocode}
\input{%s}
\begin{document}
\begin{minipage}{17cm}
%s
\end{minipage}
\end{document}
"""


def render(tex):
    """Compile a LaTeX table and show it inline; print the source if pdflatex or pdftoppm is missing or fails."""
    if not (shutil.which("pdflatex") and shutil.which("pdftoppm")):
        print(tex)
        return
    body = re.sub(r"\\(begin|end)\{table\*?\}(\[[^\]]*\])?", "", tex).replace(r"\caption{", r"\captionof{table}{")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "t.tex").write_text(STANDALONE % (MACROS, body))
        for cmd in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
                    ["pdftoppm", "-png", "-r", "150", "-singlefile", "t.pdf", "page"]):
            if subprocess.run(cmd, cwd=tmp, capture_output=True).returncode:
                print(tex)
                return
        display(Image(filename=str(Path(tmp) / "page.png")))
